import pandas as pd
import numpy as np


class RatingsMatrix():
    def __init__(self, ratings, K=10, map=False):
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
        if map:
            self.map = map
            ratings["mapped_user"] = ratings["userId"].map(
                map[0])
            ratings["mapped_item"] = ratings["movieId"].map(
                map[1])

        else:
            self.map = (dict(zip(ratings["userId"].unique(), range(self.N))), dict(
                zip(ratings["movieId"].unique(), range(self.M))))
            ratings["mapped_user"] = ratings["userId"].map(
                self.map[0])
            ratings["mapped_item"] = ratings["movieId"].map(
                self.map[1])

        # Set mapped ids as index
        self.ratings = ratings.set_index(["mapped_user", "mapped_item"])


def calc_losses(train_rm, test_rm):

    error = 0
    normalizer = 0

    for i in test_rm.ratings.reset_index()["mapped_item"].unique():

        # Get all the user ids and their ratings for the item i
        u_ids = test_rm.ratings.xs(i, level=1).index
        ratings = test_rm.ratings.xs(i, level=1)["rating"]

        # For every user that rated the item i represented by k latent dimensions each
        # how their latent dimensions interact with the item i's k latent dimensions
        # added by bias of each user ,  item i's bias and the mean rating
        pred = train_rm.user_mat[u_ids].dot(
            train_rm.item_mat[i]) + train_rm.u_b[u_ids] + train_rm.i_b[i] + train_rm.mu

        # What is the magnitude of the error vector
        error += np.linalg.norm(pred-ratings.values)  # Magnitude

        normalizer += len(ratings)

    return error/normalizer


def factorize_matrices(rm):
    # Set the parameters
    epochs = 25
    reg = 0.01

    for epoch in range(epochs):
        print("Epoch: ", epoch)

        # For user_matrix and biases
        for u in range(rm.N):
            # Get the item ids and ratings
            i_ids = rm.ratings.loc[u].index
            ratings = rm.ratings.loc[u]["rating"]

            # Relevant Items matrix will be multiplied with its Transpose
            # A.dot(A^T) Result is an K by K matrix
            # This creates a combined latent features matrix for the items
            # that this user has interacted with
            mtrx = rm.item_mat[i_ids].T.dot(
                rm.item_mat[i_ids]) + np.eye(rm.K) * reg

            # Will be vector representation of user u's movie tastes
            # First half is how much user actually deviates from his
            # own bias, all the individual item biases and general mean
            # second half will take that vector and project it in to the
            # latent dimensions of each item ending up representing the
            # deviation of users tastes for the latent features
            v = (ratings.values -
                 rm.u_b[u] - rm.i_b[i_ids] - rm.mu).dot(rm.item_mat[i_ids])

            # Biases for user is used to fine tune how much user will deviate
            # from the predicted rating of those items for user u
            # the biases of the items and general mean rating
            # for a specific user which will be further averaged later
            b_u = (ratings.values -
                   rm.item_mat[i_ids].dot(rm.user_mat[u]) - rm.i_b[i_ids] - rm.mu).sum()

            rm.u_b[u] = b_u/(len(i_ids) + reg)

            # The vector reprsentation of user u for latent k features
            # will be the linear mapping of combined movies in to vector with
            # information of the actual rating removed from biases and general mean
            # as they will be learned independently for the latent k features
            rm.user_mat[u] = np.linalg.solve(mtrx, v)

            if u % 100 == 0:
                print(u/rm.N, "% is complete for users at epoch", epoch)

        # For item_matrix and biases
        for i in range(rm.M):
            # Get the user ids and ratings
            u_ids = rm.ratings.xs(i, level=1).index
            ratings = rm.ratings.xs(i, level=1)["rating"]

            # Relevant Users matrix will be multiplied with its Transpose
            # A.dot(A^T) Result is an K by K matrix
            # This creates a combined latent features matrix for the users
            # that this item has been interacted with
            mtrx = rm.user_mat[u_ids].T.dot(
                rm.user_mat[u_ids]) + np.eye(rm.K) * reg

            # Will be vector representation of item i's feature qualities
            # First half is how much item actually deviates from its
            # own bias, all the individual users biases and general mean
            # second half will take that vector and project it in to the
            # latent dimension tastes of each user ending up representing the
            # deviation of items qualities for the latent features
            v = (ratings.values -
                 rm.i_b[i] - rm.u_b[u_ids] - rm.mu).dot(rm.user_mat[u_ids])

            # Biases for item is used to fine tune how much item will deviate
            # from the predicted rating of this item for interacted users
            # the biases of the users and general mean rating
            # for a specific item which will be further averaged later
            b_i = (ratings.values -
                   rm.user_mat[u_ids].dot(rm.item_mat[i]) - rm.u_b[u_ids] - rm.mu).sum()

            rm.i_b[i] = b_i/(len(u_ids) + reg)

            # The vector reprsentation of item i for latent k features
            # will be the linear mapping of combined users in to vector with
            # information of the actual rating removed from biases and general mean
            # as they will be learned independently for the latent k features
            rm.item_mat[i] = np.linalg.solve(mtrx, v)

            if i % 10 == 0:
                print(i/rm.M, "% is complete for items at epoch", epoch)
