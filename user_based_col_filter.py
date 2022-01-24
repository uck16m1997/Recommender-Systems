import pickle
import pandas as pd
import numpy as np
from sklearn import neighbors

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
    user_details[i] = {}

    # Get the mean and mean center the user
    user_details[i]["avg"] = X_train[i].mean()
    X_train[i] = X_train[i] - user_details[i]["avg"]
