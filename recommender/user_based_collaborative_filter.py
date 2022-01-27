import numpy as np
import pandas as pd
import pickle


def UserBasedTrain(X_train, user2movie, save=False):
    # Set the parameters for prediction
    tracking = 1
    K_N = 25  # top k most similiar neighbours
    limit = 5  # number of movies in common to
    user_details = {}  # Will hold users neighbours, average value, etc
    for i in X_train.index:

        if i not in user_details.keys():
            user_details[i] = {}
            user_details[i]["Neighbours"] = pd.DataFrame(
                {"NeighbourIndex": [], "NeighbourSimilarity": [], "AbsoluteSimilarity": []})
            # Get the mean and mean center the user
            user_details[i]["avg"] = X_train.loc[i, :].mean()
            X_train.loc[i, :] = X_train.loc[i, :] - user_details[i]["avg"]

        # Get the movies this user watched
        movies_i = user2movie[i]

        # Find the users neighbours
        for j in X_train.index:

            # Skip the same user or a user that is known neighbour
            if j == i or j in user_details[i]["Neighbours"]["NeighbourIndex"].tolist():
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
                        {"NeighbourIndex": [], "NeighbourSimilarity": [], "AbsoluteSimilarity": []})
                    # Get the mean and mean center the user
                    user_details[j]["avg"] = X_train.loc[j, :].mean()
                    X_train.loc[j, :] = X_train.loc[j, :] - \
                        user_details[j]["avg"]

                # Calculate Similarity
                numerator = np.dot(
                    X_train.loc[i, common_movies], X_train.loc[j, common_movies])

                denominator = np.linalg.norm(
                    X_train.loc[i, common_movies]) * np.linalg.norm(X_train.loc[j, common_movies])

                similarity = numerator/denominator

                # Add Neighbours if there are less than top k
                if len(user_details[i]["Neighbours"]) != K_N:

                    user_details[i]["Neighbours"] = user_details[i]["Neighbours"].append(pd.DataFrame(
                        {"NeighbourIndex": [j], "NeighbourSimilarity": [similarity], "AbsoluteSimilarity": [np.abs(similarity)]}), ignore_index=True).sort_values(by="AbsoluteSimilarity")

                # If we have filled the quota of neighbours and if its more similar to our user
                # compared to current least similar neighbour then we should replace the least similiar
                elif np.abs(similarity) > user_details[i]["Neighbours"].iloc[0]["AbsoluteSimilarity"]:
                    user_details[i]["Neighbours"] = user_details[i]["Neighbours"].append(pd.DataFrame(
                        {"NeighbourIndex": [j], "NeighbourSimilarity": [similarity], "AbsoluteSimilarity": [np.abs(similarity)]}), ignore_index=True).sort_values(by="AbsoluteSimilarity").iloc[1:]

                # Add Neighbours if there are less than top k
                if len(user_details[j]["Neighbours"]) != K_N and i not in user_details[j]["Neighbours"]["NeighbourIndex"].tolist():

                    user_details[j]["Neighbours"] = user_details[j]["Neighbours"].append(pd.DataFrame(
                        {"NeighbourIndex": [i], "NeighbourSimilarity": [similarity], "AbsoluteSimilarity": [np.abs(similarity)]}), ignore_index=True).sort_values(by="AbsoluteSimilarity")

                # If we have filled the quota of neighbours and if its more similar to seconday user
                # compared to current least similar neighbour then we should replace the least similiar
                elif np.abs(similarity) > user_details[j]["Neighbours"].iloc[0]["AbsoluteSimilarity"] and i not in user_details[j]["Neighbours"]["NeighbourIndex"].tolist():

                    user_details[j]["Neighbours"] = user_details[j]["Neighbours"].append(pd.DataFrame(
                        {"NeighbourIndex": [i], "NeighbourSimilarity": [similarity], "AbsoluteSimilarity": [np.abs(similarity)]}), ignore_index=True).sort_values(by="AbsoluteSimilarity").iloc[1:]

        print(f"{tracking/len(X_train)} % is done")
        tracking += 1
    if save:
        with open("Data/Trained/user_details.json", "wb") as f:
            pickle.dump(user_details,  f)

        X_train.columns = X_train.columns.astype(str)
        X_train.to_parquet("Data/RatingsTrain.parquet")

    return X_train, user_details


def Predict(u, m, user_details, user2movie, X_train):
    # Predicts the u th users m th movie rating
    # Needs X_train and user_details completed
    center_prediction = 0
    denom = 0
    for row in user_details[u]["Neighbours"].itertuples():
        ni = getattr(row, "NeighbourIndex")
        if m in user2movie[ni]:
            sim = getattr(row,
                          "NeighbourSimilarity")
            rating = X_train.loc[ni, m]
            center_prediction += sim * rating
            denom += sim
        else:
            continue
    try:
        pred = center_prediction/denom + user_details[u]["avg"]
        pred = max(0.5, pred)
        pred = min(5, pred)
        return pred
    except ZeroDivisionError:
        return user_details[u]["avg"]


def RootMeanSquaredError(user_details, user2movie, X_train, X_test):
    error = 0
    denom = 0
    for i in X_test.index:
        movies = X_test.columns[(~X_test.loc[i].isna())]
        for m in movies:
            denom += 1
            pred = Predict(i, m,  user_details, user2movie, X_train)
            actual = X_test.loc[i, m]
            error += np.square(pred - actual)

    return np.sqrt(error/denom)
