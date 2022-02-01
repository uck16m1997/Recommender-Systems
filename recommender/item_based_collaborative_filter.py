import numpy as np
import pandas as pd
import pickle
from recommender import ubcf


def ItemBasedTrain(X_train, movie2user, save=False):
    X_train, item_details = ubcf.UserBasedTrain(X_train, movie2user, False)
    if save:
        with open("Data/Trained/item_details.json", "wb") as f:
            pickle.dump(item_details,  f)

        X_train.columns = X_train.columns.astype(str)
        X_train.to_parquet("Data/RatingsTrain.parquet")

    return X_train, item_details


def Predict(i, u, item_details, movie2user, X_train):
    return ubcf.Predict(i, u, item_details, movie2user, X_train)


def RootMeanSquaredError(item_details, movie2user, X_train, X_test):
    error = 0
    denom = 0
    for i in X_test.index:
        users = X_test.columns[(~X_test.loc[i].isna())]
        for u in users:
            denom += 1
            pred = Predict(i, u,  item_details, movie2user, X_train)
            actual = X_test.loc[i, u]
            error += np.square(pred - actual)

    return np.sqrt(error/denom)
