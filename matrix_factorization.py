import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from recommender import mfv


def main():
    # Read the ratings
    ratings = pd.read_csv("MovieLens/rating.csv")
    ratings.drop(columns=["timestamp"], inplace=True)

    # Optional crop the data for speeding up the implementation
    TOP_K_USER = 1000
    TOP_K_MOVIE = 100

    # Find the top k users and movies
    top_k_users = ratings["userId"].value_counts().head(TOP_K_USER).index
    top_k_movies = ratings["movieId"].value_counts().head(TOP_K_MOVIE).index

    # Filter and pivot the data
    ratings = ratings[(ratings["userId"].isin(top_k_users)) & (
        ratings["movieId"].isin(top_k_movies))].reset_index(drop=True)

    # Shuffle the data
    df_filt = shuffle(ratings, random_state=42)
    cutoff = int(0.8*len(df_filt))
    df_train = df_filt[:cutoff]
    df_test = df_filt[cutoff:]

    # Number of hidden features to be extracted
    K = 10

    # Create ratings matrix
    train_rm = mfv.RatingsMatrix(df_train, K)
    test_rm = mfv.RatingsMatrix(df_test, K, train_rm.map)

    # factorize matrices
    mfv.factorize_matrices(train_rm)

    # Calculate Train Losses
    loss = mfv.calc_losses(train_rm, train_rm)
    print("Training loss is ", loss)

    # Calculate Test Losses
    loss = mfv.calc_losses(train_rm, test_rm)
    print("Test loss is ", loss)


if __name__ == '__main__':
    main()
