import pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
folder = "MovieLens/"
rating = "rating.csv"
df_rate = pd.read_csv(folder+rating)

# Lower cast the rating (max is 5)
df_rate["rating"] = df_rate["rating"].astype(np.float32)


# Control the difference between size and maximum index
len(df_rate["userId"].unique()) - max(df_rate["userId"].unique())

# Lower cast the userId
df_rate["userId"] = df_rate["userId"].astype(np.int32)


# Control the difference between size and maximum index
# There are missing values for MovieIds
len(df_rate["movieId"].unique()) - max(df_rate["movieId"].unique())

# Lower cast the movieId
df_rate["movieId"] = df_rate["movieId"].astype(np.int32)

# Don't need this one
df_rate.drop(columns=["timestamp"], inplace=True)

# Create a smaller Dataset of most active users and movies for memory efficiency
TOP_K_USER = 10000
TOP_K_MOVIE = 1000

# Find the top k users and movies
top_k_users = df_rate["userId"].value_counts().head(TOP_K_USER).index
top_k_movies = df_rate["movieId"].value_counts().head(TOP_K_MOVIE).index

# Filter and pivot the data
df_filt = df_rate[(df_rate["userId"].isin(top_k_users)) & (
    df_rate["movieId"].isin(top_k_movies))].reset_index()

# Shuffle the data
df_filt = shuffle(df_filt, random_state=42)
cutoff = int(0.8*len(df_filt))
df_train = df_filt[:cutoff]
df_test = df_filt[cutoff:]


matrix = df_train.pivot_table(
    columns="movieId", values="rating", index="userId")

test_matrix = df_test.pivot_table(
    columns="movieId", values="rating", index="userId")


# We need to track which users have rated which movies and vice versa
user2movie = {}
for i in matrix.index:
    user2movie[i] = matrix.columns[(~matrix.loc[i].isna())]

movie2user = {}
for i in matrix.columns:
    movie2user[i] = matrix.index[(~matrix.loc[:, i].isna())]


# Save the Processed Data
matrix.columns = matrix.columns.astype(str)
matrix.to_parquet("Data/RatingsTrain.parquet")

test_matrix.columns = test_matrix.columns.astype(str)
test_matrix.to_parquet("Data/RatingsTest.parquet")

# Save the Dictionaries
with open("Data/user2movie.json", "wb") as f:
    pickle.dump(user2movie, f)

with open("Data/movie2user.json", "wb") as f:
    pickle.dump(movie2user,  f)
