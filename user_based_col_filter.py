import pickle
import os
import pandas as pd
from recommender import ubcf

# Set the Data locations and names
folder = "Data/"
trained = folder+"Trained/"
train = "RatingsTrain.parquet"
test = "RatingsTest.parquet"


def main(clip=100):
    # Read and fix parquet columns
    X_train = pd.read_parquet(folder+train)
    X_train.columns = X_train.columns.astype(int)

    X_test = pd.read_parquet(folder+test)
    X_test.columns = X_test.columns.astype(int)

    # Read the pickle dictionaries
    with open(folder+"user2movie.json", "rb") as f:
        user2movie = pickle.load(f)

    with open(folder+"movie2user.json", "rb") as f:
        movie2user = pickle.load(f)

    # Determine if there are movies and users that are missing in the train
    X_test.columns.difference(X_train.columns)
    X_test.index.difference(X_train.index)

    # If there is a need to train
    if "userdetails.json" in os.listdir(trained):
        with open(trained+"userdetails.json", "rb") as f:
            user_details = pickle.load(f)
    else:
        # For the time being its safe to clip the amount of users
        X_train, user_details = ubcf.UserBasedTrain(
            X_train.head(clip), user2movie)
        X_test = X_test.head(clip)

    error = ubcf.RootMeanSquaredError(
        user_details, user2movie, X_train, X_train)
    print("User based Root Mean Squared Error for Train is: ", error)

    error = ubcf.RootMeanSquaredError(
        user_details, user2movie, X_train, X_test)
    print("User based Root Mean Squared Error for Test is: ", error)


if __name__ == '__main__':
    main()
