from abc import ABC, abstractmethod
import numpy as np
class BaseTSC(ABC):

    def __init__(self,name):
        self.name = name

    @abstractmethod
    def train(self, X,y, epochs: int)->None:
        pass

    @abstractmethod
    def predict(self, X)->np.ndarray:
        pass

    def get_name(self)->str:
        return self.name

    def fit(self, *args, **kwargs)->None:
        self.train(*args, **kwargs)

