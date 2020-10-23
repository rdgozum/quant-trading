import numpy as np

from sklearn.cluster import DBSCAN


class DBSCANClustering:
    def __init__(self):
        self.dbscan = DBSCAN(eps=5, min_samples=2)

    def run(self, features):
        features = np.asarray(features, dtype=np.float32)
        clustering = self.dbscan.fit(features)
        print(clustering.labels_)
