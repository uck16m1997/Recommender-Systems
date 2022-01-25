import pickle
import pandas as pd
import numpy as np
from sklearn import neighbors
from sympy import li

# Set the Data locations and names
folder = "Data/"
train = "RatingsTrain.parquet"
test = "RatingsTest.parquet"

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

# Set the parameters for prediction
K_N = 25  # top k most similiar neighbours
limit = 5  # number of movies in common to
user_details = {}  # Will hold users neighbours, average value, etc
for i in X_train.index:

    if i not in user_details.keys():
        user_details[i] = {}
        user_details[i]["Neighbours"] = pd.DataFrame(
            {"NeighbourIndex": [], "NeighbourSimilarity": []})
        # Get the mean and mean center the user
        user_details[i]["avg"] = X_train.loc[i, :].mean()
        X_train.loc[i, :] = X_train.loc[i, :] - user_details[i]["avg"]

    # Get the movies this user watched
    movies_i = user2movie[i]

    # Find the users neighbours
    for j in X_train.index:

        # Skip the same user or a user that is known neighbour
        if j == i or j in user_details[i]["Neighbours"]["NeighbourIndex"]:
            continue

        # Get the potential neighbours movie list
        movies_j = user2movie[j]

        # Extract the movies in common for both users
        common_movies = (set(movies_i) & set(movies_j))

        # If there are enough common movies to calculate similarity
        if len(common_movies) > limit:

            if j not in user_details.keys():
                user_details[j] = {}
                user_details[j]["Neighbours"] = pd.DataFrame(
                    {"NeighbourIndex": [], "NeighbourSimilarity": []})
                # Get the mean and mean center the user
                user_details[j]["avg"] = X_train.loc[j, :].mean()
                X_train.loc[j, :] = X_train.loc[j, :] - user_details[j]["avg"]

            # Calculate Similarity
            numerator = np.dot(
                X_train.loc[i, common_movies], X_train.loc[j, common_movies])

            denominator = np.linalg.norm(
                X_train.loc[i, common_movies]) * np.linalg.norm(X_train.loc[j, common_movies])

            similarity = numerator/denominator

            # Add Neighbours if there are less than top k
            if len(user_details[i]["Neighbours"]) != 25:

                user_details[i]["Neighbours"] = user_details[i]["Neighbours"].append(pd.DataFrame(
                    {"NeighbourIndex": [j], "NeighbourSimilarity": [similarity]}), ignore_index=True).sort_values(by="NeighbourSimilarity")

            # If we have filled the quota of neighbours and if its more similar to our user
            # compared to current least similar neighbour then we should replace the least similiar
            elif np.abs(similarity) > user_details[i]["Neighbours"].iloc[0]["NeighbourSimilarity"]:
                user_details[i]["Neighbours"] = user_details[i]["Neighbours"].append(pd.DataFrame(
                    {"NeighbourIndex": [j], "NeighbourSimilarity": [similarity]}), ignore_index=True).sort_values(by="NeighbourSimilarity").iloc[1:]

            # Add Neighbours if there are less than top k
            if len(user_details[j]["Neighbours"]) != 25:

                user_details[j]["Neighbours"] = user_details[j]["Neighbours"].append(pd.DataFrame(
                    {"NeighbourIndex": [i], "NeighbourSimilarity": [similarity]}), ignore_index=True).sort_values(by="NeighbourSimilarity")

            # If we have filled the quota of neighbours and if its more similar to seconday user
            # compared to current least similar neighbour then we should replace the least similiar
            elif np.abs(similarity) > user_details[j]["Neighbours"].iloc[0]["NeighbourSimilarity"]:

                user_details[j]["Neighbours"] = user_details[j]["Neighbours"].append(pd.DataFrame(
                    {"NeighbourIndex": [i], "NeighbourSimilarity": [similarity]}), ignore_index=True).sort_values(by="NeighbourSimilarity").iloc[1:]
