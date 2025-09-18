import numpy as np
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

from TSC.base_TSC import BaseTSC


class KNN_DTW(BaseTSC):
    NAME = "KNN_DTW"
    def __init__(self):
        super().__init__(self.NAME)
        self.trained = False
        self.tsc = KNeighborsTimeSeriesClassifier(n_neighbors=3, distance="dtw")

    def train(self, X:np.ndarray,y:np.ndarray, epochs: int)->None:
        self.tsc.fit(X,y)
        self.trained = True

    def predict(self, X:np.ndarray)->np.ndarray:
        assert self.trained, "Model must be trained before predicting"
        return self.tsc.predict(X)


