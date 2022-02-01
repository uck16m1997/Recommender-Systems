import pickle
import os
import pandas as pd
from recommender import ibcf

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

    # Transpose the Matrices so items become rows and users become columns
    X_train = X_train.transpose()
    X_test = X_test.transpose()

    # Read the pickle dictionaries
    with open(folder+"user2movie.json", "rb") as f:
        user2movie = pickle.load(f)

    with open(folder+"movie2user.json", "rb") as f:
        movie2user = pickle.load(f)

    # Determine if there are movies and users that are missing in the train
    X_test.columns.difference(X_train.columns)
    X_test.index.difference(X_train.index)

    # If there is a need to train
    if "item_details.json" in os.listdir(trained):
        with open(trained+"item_details.json", "rb") as f:
            item_details = pickle.load(f)
    else:
        # For the time being its safe to clip the amount of users
        X_train, item_details = ibcf.ItemBasedTrain(
            X_train.head(clip), movie2user)
        X_test = X_test.head(clip)

    error = ibcf.RootMeanSquaredError(
        item_details, movie2user, X_train, X_train)
    print("Item based Root Mean Squared Error for Train is: ", error)

    error = ibcf.RootMeanSquaredError(
        item_details, movie2user, X_train, X_test)
    print("Item based Root Mean Squared Error for Test is: ", error)


if __name__ == "__main__":
    main()
