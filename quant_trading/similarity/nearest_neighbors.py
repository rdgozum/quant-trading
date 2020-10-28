from sklearn.neighbors import NearestNeighbors


class KNN:
    def __init__(self, k):
        self.knn = NearestNeighbors(n_neighbors=k)

    def run(self, X):
        neighbors_fit = self.knn.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)

        return distances, indices
