import streamlit as st
import os
import requests

import warnings
warnings.filterwarnings("ignore")

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, udf
from pyspark.sql.types import ArrayType, FloatType
import pyspark.sql.functions as F

# Creating spark context
spark = SparkSession.builder.appName("MovieRecommendations").getOrCreate()


# read metadata of movieLensDataset
metadata_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .json('data/metadata_with_popularity_rank.json')


metadata_df.printSchema()

def get_imdb_id(movie_id):
    """
    Given a movie_id, returns its imdbId.

    Parameters:
    - movie_id (int): The ID of the movie.

    Returns:
    - str: The imdbId of the movie.
    """
    print(movie_id)

    # Filter records for movieId
    filtered_df = metadata_df.filter(metadata_df.item_id == int(movie_id))

    # Extract the imdbId from the filtered record
    imdb_id = filtered_df.select("imdbId").first()[0]

    return imdb_id

def get_movie_name(movie_id):
    '''
    Given a movie_id, returns its movie name.

    Parameters:
    - movie_id (int): The ID of the movie.

    Returns:
    - str: The name of the movie.

    '''

    # Filter records for movieId
    filtered_df = metadata_df.filter(metadata_df.item_id == movie_id)

    # Extract the imdbId from the filtered record
    movie_name = filtered_df.select("title").first()[0]

    return movie_name


# Load the pre-trained ALS model
model = ALSModel.load("model/best_model")

def recommend(user_id, user_ratings, first_time_user=False, n_movies_to_recommend=10):

    # Create a Spark DataFrame for user ratings
    user_ratings_df = spark.createDataFrame([(user_id, int(movie_id), float(rating)) for movie_id, rating in user_ratings.items()], ["user_id", "movie_id", "rating"])

    # Extract item factors from the ALS model
    item_factors_df = model.itemFactors


    if first_time_user:
        # if user is first time user then use Fold-in technique to make recommendations

        def foldingIn(user_ratings_df, new_user_id=1111111, n_movies_to_recommend=5):
            '''
            Returns updated user vector after folding in technique.

            It will use pre-trained Recommendation model to make new recommendations.
            
            If user is unseen during training then it will use folding in technique to make recommendations.

            parametres
            -----------
            n_movies_to_recommend = movies to recommend to user (defautl: 5)
            user_ratings = python dictionary containing movies ratings
            
            
            exmple: {235929: '3', 4449: '4', 2350: '1'}
            where key is movie_id and value is rating
            '''

            # Join user ratings with item factors on movie_id
            joined_df = user_ratings_df.join(item_factors_df, user_ratings_df.movie_id == item_factors_df.id, "left")

            # Define a UDF for element-wise multiplication
            element_wise_multiply_udf = udf(lambda rating, features: [float(rating) * float(feature) for feature in features], ArrayType(FloatType()))

            # Calculate rating contribution for each feature
            projected_vector_on_item_space = joined_df.withColumn("rating_contribution", element_wise_multiply_udf(joined_df["rating"], joined_df["features"]))

            # Get the number of dimensions of the feature vector (k)
            k = len(projected_vector_on_item_space.select("rating_contribution").first()[0])

            # Aggregate by summing up the values in each dimension
            aggregation_exprs = [expr(f"SUM(rating_contribution[{i}]) as sum_{i}") for i in range(k)]
            aggregated_df = projected_vector_on_item_space.groupBy().agg(*aggregation_exprs)

            user_vector_df = aggregated_df.select(F.array([col(f"sum_{i}") for i in range(k)]).alias("user_vector"))

            # Generate collaborative filtering recommendations
            # Cross join user_preferences_df with item_latent_factors_df
            recommendations_df = user_vector_df.crossJoin(item_factors_df)

            # Calculate preference scores using dot product
            recommendations_df = recommendations_df.withColumn("preference_score", sum(
                col("user_vector")[i] * col("features")[i] for i in range(k)))

            recommendations_df = recommendations_df.select("id", "preference_score")

            # Rename the "id" column in recommendations_df to match the column name in user_ratings_df
            recommendations_df = recommendations_df.withColumnRenamed("id", "movie_id")

            # Filter recommendations that weren't part of users' ratings
            recommendations_df = recommendations_df.join(
                user_ratings_df.select("movie_id").distinct(),
                on="movie_id",
                how="left_anti").select("movie_id", "preference_score")

            
            # Rank recommendations by preference score in descending order
            recommended_items_df = recommendations_df.orderBy(col("preference_score").desc()).limit(n_movies_to_recommend)

            # Select only the columns you want in the result DataFrame
            
            return recommended_items_df.collect()
        
        # Generate collaborative filtering recommendations
        recommendations_for_user = foldingIn(user_ratings_df, new_user_id=user_id, n_movies_to_recommend=n_movies_to_recommend)

    else:
        # if user is not first-time user then use pre-trained model to make recommendations
        def recommendForExistingUser(user_id, user_ratings_df, n_movies_to_recommend):
            ''''
            It will use pre-trained Recommendation model to make new recommendations. 
            
            User has to at least recommned one movie in order to get recommendations.

            parametres
            -----------
            user_id = user id
            n_movies_to_recommend = movies to recommend to user (defautl: 5)
            user_ratings = python dictionary containing movies ratings
            
            exmple: {235929: '3', 4449: '4', 2350: '1'}
            where key is movie_id and value is rating
            '''

            # Generate collaborative filtering recommendations
            collaborative_recommendations = model.recommendForUserSubset(user_ratings_df, n_movies_to_recommend)

            # Check if the recommendations list is not empty
            if collaborative_recommendations is not None:
                recommendations_for_user = collaborative_recommendations.collect()
                return recommendations_for_user

            else:
                # Handle the case where recommendations are not available
                return None
        
        recommendations_for_user = recommendForExistingUser(user_id, user_ratings_df, n_movies_to_recommend)

         
    print('-----------------------------------------------------')
    if recommendations_for_user is not None:    
        # Extract movie IDs from the recommendations
        print(recommendations_for_user)
        movie_ids = [row[0] for row in recommendations_for_user]

        # print(movie_ids)

        recommend_movies_ids = []
        recommend_movies_names = []
        recommend_posters = []

        for movie_id in movie_ids:
            imdb_id = get_imdb_id(movie_id)

            # fetch poster
            poster_url = fetch_poster(imdb_id)
            if poster_url is not None: # appending only if their poster is found

                recommend_posters.append(poster_url)

                # extract movie names for that id
                m_name = get_movie_name(movie_id)
                recommend_movies_names.append(m_name)

                # adding movie id
                recommend_movies_ids.append(movie_id)

        return recommend_movies_ids, recommend_movies_names, recommend_posters
    
    else:
        print("No recommendations available for the given user ratings.")
        return [], [], []


def fetch_poster(imdb_id):
    '''
    It will fetch posters for a given imdb_id using TMDB API.
    '''
    try:
        api_key = "b0c6daf6553e5bf049c7a0dc27ae23bd"
        response = requests.get(f"https://api.themoviedb.org/3/find/tt{imdb_id}?api_key={api_key}&external_source=imdb_id")

        # Check if the request was successful
        response.raise_for_status()

        data = response.json()
        if 'movie_results' in data and data['movie_results']:
            poster_path = data['movie_results'][0]['poster_path']
            if poster_path:
                url = "https://image.tmdb.org/t/p/w500/" + poster_path
                return url

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching poster: {e}")

    return None
# Front End
# -------------------------------------------------------------


# Set the page's title and icon
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",)


# Custom CSS for the UI
st.markdown(
    """
    <style>
    body {
        background-color: #040606;  /* Dark background color */
        color: #FFD700;            /* Text color (Gold) */
        font-family: 'Comic Sans MS', cursive;
    }
    .stButton button {
        background-color: transparent; /* Remove the background color of the button */
        color: #FFFFFF;
        border-radius: 10px;
    }
    .stSelectbox {
        color: #FFD700;            /* Selectbox text color (Gold) */
        background-color: #0B0D17;  /* Selectbox background color */
    }
    .stRating {
        font-size: 24px;
    }
    .stSelectbox > div:first-child {
        background-color: #0B0D17;   /* Selectbox open button background color */
    }
    .stSelectbox > div:last-child {
        background-color: #0B0D17;   /* Selectbox dropdown background color */
    }
    .poster-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        width: 200px;  /* Adjust the width as needed */
        height: 250px;  /* Fixed height for the container */
    }
    .poster {
        max-width: 100%;  /* Max width for the poster */
        max-height: 80%;  /* Max height for the poster */
    }
    .caption {
        height: 20px;  /* Fixed height for the caption */
        overflow: hidden; /* Hide overflow text */
    }
    .header-text {
        text-align: left;
    }
    .text-container {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)# Custom layout to show tagline aligned to the left
st.markdown("<h2 class='header-text' style='color:#E50914;'>Discover your next favorite Movies 🍿</h2>", unsafe_allow_html=True)

st.markdown("<p style=#9C9D9F; font-size: 18px;'>Welcome to <b>Movie Recommender</b>, the free movie recommendation engine that suggests films based on your interest.</p>",
            unsafe_allow_html=True)



# ------
n_movies_to_display = 5  #( setting it to 5 for now, movie to display in one row)
n_random_movies = 10
top_k_popular_movies = 20
    

# Create a session state to store selected movies and posters (showing only top 1000 movies to rate)
if 'selected_movies' not in st.session_state:
    selected_movies = metadata_df.where(metadata_df.popularity_rank<=top_k_popular_movies)\
                        .orderBy(F.rand()).limit(n_random_movies).collect()
    st.session_state.selected_movies = selected_movies

# Emoji ratings with star ratings
emoji_ratings = {
    "Rate the movie": "❓ (Not Rated)",
    "1": "😠 (1 Star)",
    "2": "😕 (2 Stars)",
    "3": "😐 (3 Stars)",
    "4": "😃 (4 Stars)",
    "5": "😍 (5 Stars)",
}

# Display the selected movies and posters
selectbox_width = 200  # Adjust the width of the selectbox as needed
image_width = selectbox_width

# Display the selected movies and posters
user_ratings = {}

columns = st.columns(n_movies_to_display) # Display 5 movies in one row
cnt = 0

for i, movie in enumerate(st.session_state.selected_movies):
    if cnt >= 5:
        break

    movie_id = movie.item_id
    movie_name = movie.title
    imdb_id = movie.imdbId
    poster_url = fetch_poster(imdb_id)
    if poster_url is not None:  # Check if a movie poster is found
        with columns[cnt]:
            st.image(poster_url, caption="", output_format="PNG", width=image_width)
            with st.container():
                st.text(movie_name)
            rating = st.selectbox(f"", list(emoji_ratings.keys()), key=movie_name, format_func=lambda x: x.split(" (")[0])
            if rating != 'Rate the movie':
                user_ratings[movie_id] = rating
        cnt += 1

print(user_ratings)

#---------------------
# BackEnd

# Button to trigger recommendations
if st.button("Recommend"):
    # Check if the user has rated at least one movie
    if  len(user_ratings.values()) == 0:
        st.markdown("<p style='color:#FFBF00; font-size: 18px;'><b>Please rate at least one movie before proceeding.</b></p>",
                    unsafe_allow_html=True)
    else:
        st.write("<span style='color: #00BFFF;font-size: 18px;'><b>Personalizing your Recommendations...</b></span>", unsafe_allow_html=True)

        # Every user is first time user
        first_time_user = True

        recommended_movie_ids,recommended_movie_names, recommended_movie_posters = \
            recommend(1111111, user_ratings, first_time_user, n_movies_to_recommend=10)

        if len(recommended_movie_names) != 0:

            if len(recommended_movie_names) < 5:
                n_movies_to_display = len(recommended_movie_names)
            
            # I am displaying only 5 movies in one row
            n_movies_to_display = 5

            cnt = 0

            st.write("<span style='color: #00BFFF;font-size: 18px;'><b>Here are a few Recommendations..........</b></span>",\
                      unsafe_allow_html=True)
            columns = st.columns(n_movies_to_display)
            
            for i, movie_name in enumerate(recommended_movie_names):
                if cnt == 5:
                    break
                cnt += 1
                try:
                    with columns[i]:
                        st.image(recommended_movie_posters[i], caption=movie_name, output_format="PNG", width=200)
                except:
                    st.markdown(f"![{movie_name}]({recommended_movie_posters[i]})")

                # columns[i].image(recommended_movie_posters[i], caption=movie_name, output_format="PNG", width=150)
        else:
            st.markdown("<p style='color:#FFBF00; font-size: 18px;'><b>Sorry, We couldn't find any recommendations for you.</b></p>",
                        unsafe_allow_html=True)
            
            

