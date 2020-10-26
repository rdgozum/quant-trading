import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt


class DBSCANClustering:
    def __init__(self, epsilon, min_samples):
        self.dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

    def run(self, features, symbols):
        clustering = self.dbscan.fit(features)
        print(clustering.labels_)


def optimal_epsilon(features, min_samples):
    # Calculate the average distance between each point
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)

    # Sort distance values in ascending order and plot
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()


def optimal_min_samples(features):
    # Calculate the min_samples from data dimension
    min_samples = features.shape[1] * 2

    return min_samples
