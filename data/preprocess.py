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


def create_user_sequences(
    ratings,
    min_interactions=3,
    min_rating=0
):
    """
    Creates cleaned user sequences

    - Sorts by timestamp
    - Filters low-interaction users
    - Filters low ratings (optional)
    """

    # Filter by rating if needed
    if min_rating > 0:
        ratings = ratings[ratings["rating"] >= min_rating]

    # Sort chronologically
    ratings = ratings.sort_values(["user_id", "timestamp"])

    # Group into sequences
    user_sequences = ratings.groupby("user_id")["movie_id"].apply(list)

    # Filter short sequences
    user_sequences = user_sequences[
        user_sequences.apply(len) >= min_interactions
    ]

    return user_sequences.to_dict()


def train_val_test_split(user_sequences):
    """
    Leave-one-out split:
    - last item → test
    - second last → validation
    - rest → train
    """

    train_sequences = {}
    val_targets = {}
    test_targets = {}

    for user, items in user_sequences.items():

        if len(items) < 3:
            continue

        train_sequences[user] = items[:-2]
        val_targets[user] = items[-2]
        test_targets[user] = items[-1]

    return train_sequences, val_targets, test_targets
