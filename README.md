# Movie Recommendation Engine

Movie Recommender is a collaborative filtering-based recommendation engine that provides personalized movie recommendations to users based on their preferences. This project utilizes __Apache Spark__ for model training and __Streamlit__ for building a user-friendly interface. The system uses an ALS (Alternating Least Squares) model trained on a Spark cluster. The trained model is then used to provide movie recommendations to new users using the fold-in technique, instead of Re-training entire model.

## Overview

The Movie Recommendation Engine offers the following features:

- **Collaborative Filtering:** The recommendation engine uses collaborative filtering techniques to suggest movies to users based on their historical ratings and preferences.

- **Streamlit Interface:** Users interact with the recommendation system through a Streamlit web interface, making it easy and intuitive to use.

- **Personalized Recommendations:** Users can rate movies, and the system generates personalized movie recommendations tailored to their individual tastes.

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

