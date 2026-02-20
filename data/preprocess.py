import pandas as pd
import os


def load_ml1m(data_path="data/ml-1m"):
    ratings_path = os.path.join(data_path, "ratings.dat")
    movies_path = os.path.join(data_path, "movies.dat")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"]
    )

    return ratings, movies


def create_user_sequences(ratings):
    ratings = ratings.sort_values(["user_id", "timestamp"])

    user_sequences = ratings.groupby("user_id")["movie_id"].apply(list)

    return user_sequences.to_dict()