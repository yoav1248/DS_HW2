import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.00001^2)
    """
    noise = np.random.normal(loc=0, scale=1e-5, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    df_relevant = df[features]
    df_scaled = min_max_scale(df_relevant)
    return add_noise(df_scaled).to_numpy()

def min_max_scale(df):
    return (df - df.min()) / (df.max() - df.min())

def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    prev_centroids = None
    labels = None
    current_centroids = choose_initial_centroids(data, k)

    while prev_centroids is None or not np.array_equal(prev_centroids, current_centroids):
        prev_centroids = current_centroids.copy()
        labels = assign_to_clusters(data, current_centroids)
        current_centroids = recompute_centroids(data, labels, k)

    return labels, current_centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    pass
    # plt.savefig(path)


def dist(vectors, reference):
    """
    Euclidean distances between some multidimensional array of vectors
     and a reference point.
    :param vectors: numpy array of shape (..., n)
    :param reference: numpy array of size n
    :return: numpy array of shape (...) as in vectors, containing distances
    """
    return np.linalg.norm(vectors - reference, axis = -1)

def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    closest_index = lambda point: dist(centroids, point).argmin()
    return np.apply_along_axis(closest_index, 1, data)

def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    return np.array([data[labels == label].mean(axis = 0) for label in range(k)])

