from abc import ABC, abstractmethod


class BaseGenerator(ABC):

    def __init__(self, name):
        self.name = name
    @abstractmethod
    def train(self, dataset_name: str, epochs: int, nr_samples: int):
        pass

    def get_name(self) -> str:
        return self.name