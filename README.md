# Movie Recommendation Engine


https://github.com/shant-kolekar/movieRecommender/assets/97169131/f630c6f6-46b2-4068-b15c-d99af3bca495


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movierecommendationengine.streamlit.app)



Movie Recommender is a collaborative filtering-based recommendation engine that provides personalized movie recommendations to users based on their preferences. This project utilizes __Apache Spark__ for model training and __Streamlit__ for building a user-friendly interface. The engine uses an ALS (Alternating Least Squares) model trained on a Spark cluster. The trained model is then used to provide movie recommendations to new users using the fold-in technique, instead of Re-training the entire model.

## Overview

The Movie Recommendation Engine offers the following features:

- **Collaborative Filtering:** The recommendation engine uses collaborative filtering techniques to suggest movies to users based on their ratings provided. The more you rate the more personalized content you get.

- **Streamlit Interface:** Users interact with the recommendation engine through a Streamlit web interface, making it easy and intuitive to use.

- **Personalized Recommendations:** Users can rate movies, and the engine generates personalized movie recommendations tailored to their individual tastes.

- **Spark Integration:** The ALS (Alternating Least Squares) model is trained using Apache Spark cluster, ensuring scalability and efficiency.

## Getting Started

To get started with the Movie Recommender system, follow these steps:

### Clone the repository to your local machine:

```shell
git clone https://github.com/shant-kolekar/movieRecommender.git
```

### Install the required dependencies:

```shell
pip install -r requirements.txt
```

### Run the Streamlit app:

```shell
streamlit run app.py
```

### Open a web browser and navigate to http://localhost:8501 to access the Movie Recommender interface.

>Demo file at demo/demo.mp4
