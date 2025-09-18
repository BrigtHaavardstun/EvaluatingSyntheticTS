import numpy as np
from sktime.classification.interval_based import TimeSeriesForestClassifier

from TSC.base_TSC import BaseTSC


class TSForest(BaseTSC):
    NAME = "TSForest"
    def __init__(self):
        super().__init__(self.NAME)
        self.trained = False
        self.tsc = TimeSeriesForestClassifier(n_estimators=50, random_state=42)

    def train(self, X:np.ndarray,y:np.ndarray, epochs: int)->None:
        self.tsc.fit(X,y)
        self.trained = True

    def predict(self, X:np.ndarray)->np.ndarray:
        assert self.trained, "Model must be trained before predicting"
        return self.tsc.predict(X)


