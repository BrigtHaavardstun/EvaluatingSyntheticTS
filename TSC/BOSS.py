import numpy as np
from sktime.classification.dictionary_based import BOSSEnsemble

from TSC.base_TSC import BaseTSC


class BOSS(BaseTSC):
    NAME = "BOSS"
    def __init__(self):
        super().__init__(self.NAME)
        self.trained = False
        self.tsc = BOSSEnsemble(max_ensemble_size=25, random_state=42)

    def train(self, X:np.ndarray,y:np.ndarray, epochs: int)->None:
        self.tsc.fit(X,y)
        self.trained = True

    def predict(self, X:np.ndarray)->np.ndarray:
        assert self.trained, "Model must be trained before predicting"
        return self.tsc.predict(X)


