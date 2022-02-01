import pandas as pd
import numpy as np


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

    # Number of hidden features to be extracted
    K = 10

    # Create ratings matrix
    rm = RatingsMatrix(ratings, K)

    # factorize matrices
    factorize_matrices(rm)

    # Calculate Losses
    loss = calc_losses(rm)
    print("Training loss is ", loss)


class RatingsMatrix():
    def __init__(self, ratings, K=10):
        # Create user matrix
        self.K = K
        self.N = len(ratings.userId.unique())
        self.u_b = np.zeros(self.N)  # User biases
        self.user_mat = np.random.rand(self.N, K)

        # Create item matrix
        self.M = len(ratings.movieId.unique())
        self.i_b = np.zeros(self.M)  # Item biases
        self.item_mat = np.random.rand(self.M, K)

        # Average rating
        self.mu = ratings.rating.mean()

        # Need to reset ids
        ratings["mapped_user"] = ratings["userId"].map(
            dict(zip(ratings["userId"].unique(), range(self.N))))
        ratings["mapped_item"] = ratings["movieId"].map(
            dict(zip(ratings["movieId"].unique(), range(self.M))))

        # Set mapped ids as index
        self.ratings = ratings.set_index(["mapped_user", "mapped_item"])


def calc_losses(rm):

    error = 0
    for u, i in rm.ratings.index:
        pred = rm.user_mat[u].dot(rm.item_mat[i]) + \
            rm.u_b[0] + rm.i_b[i] + rm.mu
        error += np.square(pred - rm.ratings.loc[u, i]["rating"])
    return error/len(rm.ratings)


def factorize_matrices(rm):
    # Set the parameters
    epochs = 25
    reg = 0.01

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        # For user_matrix and biases
        for u in range(rm.N):
            # For user matrix
            # Start with the already setted regularization
            mtrx = np.eye(rm.K) * reg

            # Will be vector representation of users movie tastes
            v = np.zeros(rm.K)

            # For user bias
            b_u = 0

            # for items rated by user u
            for i in rm.ratings.loc[u].index:
                r = rm.ratings.loc[u, i]["rating"]

                # Outer multipliaction is basically a line
                # represented as a matrix for the item i
                # different lines are added for each movie user watched
                # to get a combined representation of all movies
                mtrx += np.outer(rm.item_mat[i], rm.item_mat[i])

                # Get the combined representation of how much the user
                # has deviated in real life to his biases item i biases and general mean
                # considering item i's latent K features
                v += (r - rm.u_b[u] - rm.i_b[i] - rm.mu) * rm.item_mat[i]

                # Bias for user is used to fine tune how much user will
                # deviate from the mean of the movie and general mean rating
                # for a specific movie which will be further averaged later
                b_u += (r -
                        rm.user_mat[u].dot(rm.item_mat[i]) - rm.i_b[i] - rm.mu)

            rm.u_b[u] = b_u

            # The vector reprsentation of user u for latent k features
            # will be the linear mapping of combined movies in to vector with
            # information of the actual rating removed from biases and general mean
            # as they will be learned independently for the latent k features
            rm.user_mat[u] = np.linalg.solve(mtrx, v)

            if u % 100 == 0:
                print(u/rm.N, "% is complete for users")

        # For item_matrix and biases
        for i in range(rm.M):
            # Start with the already setted regularization
            mtrx = np.eye(rm.K) * reg

            # Will be vector representation of items latent features
            v = np.zeros(rm.K)

            # For item bias
            b_i = 0

            # for items rated by user u
            for u in rm.ratings.xs(i, level=1).index:
                r = rm.ratings.loc[u, i]["rating"]

                # Outer multipliaction is basically a line
                # represented as a matrix for the user u
                # different lines are added for each user who rated
                # to get a combined representation of all users
                mtrx += np.outer(rm.user_mat[u], rm.user_mat[u])

                # Get the combined representation of how much the item
                # has deviated in real life to its biases user u biases and general mean
                # considering users u tastes for the latent K features
                v += (r - rm.u_b[u] - rm.i_b[i] - rm.mu) * rm.user_mat[u]

                # Bias for item is used to fine tune how much item will
                # deviate from the mean of the users and general mean rating
                # for a specific user which will be further averaged later
                b_i += (r -
                        rm.user_mat[u].dot(rm.item_mat[i]) - rm.u_b[u] - rm.mu)

            rm.i_b[i] = b_i

            # The vector reprsentation of item i for latent k features
            # will be the linear mapping of combined users in to vector with
            # information of the actual rating removed from biases and general mean
            # as they will be learned independently for the latent k features
            rm.item_mat[i] = np.linalg.solve(mtrx, v)

            if i % 100 == 0:
                print(i/rm.M, "% is complete for items")


if __name__ == '__main__':
    main()
