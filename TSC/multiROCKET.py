from sktime.transformations.panel.rocket import MultiRocket as MultiRocket_sktime
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV

from TSC.base_TSC import BaseTSC
import numpy as np

class MultiROCKET(BaseTSC):
    NAME = "MultiROCKET"
    def __init__(self):
        super().__init__(self.NAME)
        self.trained = False
        self.tsc = make_pipeline(
            MultiRocket_sktime(),          # Transformer
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


if __name__ == "__main__":
    model = MultiROCKET()
    from utils.dataset import load_dataset

    dataset_name = "Chinatown"
    data = load_dataset(dataset_name=dataset_name, dataset_type="TRAIN")

    x = data.iloc[:, 1:].to_numpy().astype(float)

    y = data.iloc[:, 0].to_numpy()

    model.train(x, y, epochs=10)
    print(model.predict(x))
