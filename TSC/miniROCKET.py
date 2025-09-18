from sktime.transformations.panel.rocket import MiniRocket as MiniROCKET_sktime
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from TSC.base_TSC import BaseTSC
import numpy as np

class MiniROCKET(BaseTSC):
    NAME = "MiniROCKET"
    def __init__(self):
        super().__init__(self.NAME)
        self.trained = False
        self.tsc = make_pipeline(
            MiniROCKET_sktime(),          # Transformer
            RidgeClassifierCV()            # Classifier
        )

    def train(self, X:np.ndarray,y:np.ndarray, epochs: int)->None:
        X = X[:, np.newaxis, :]
        self.tsc.fit(X,y)
        self.trained = True

    def predict(self, X:np.ndarray)->np.ndarray:
        assert self.trained, "Model must be trained before predicting"
        X = X[:, np.newaxis, :]
        return self.tsc.predict(X)