import pandas as pd
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import Metadata
import os
import numpy as np
from pathlib import Path


from matplotlib import pyplot as plt

from Generators.base_generator import BaseGenerator
from utils.dataset import load_data_by_class, get_metadata,get_metadata_no_label
from utils.save import save_synthetic_data, load_synthetic_data
from utils.viz import load_and_visualize
from sklearn.preprocessing import MinMaxScaler


class Copula_GAN(BaseGenerator):
    NAME = "COPULA_GAN"
    def __init__(self):
        super().__init__(name=self.NAME)

    def train_class(self,dataset_name,group, epochs=100):
        no_label = group.iloc[:, 1:]  # Keep as DataFrame
        metadata = get_metadata_no_label(dataset_name=dataset_name)
        synthesizer = CopulaGANSynthesizer(
            metadata,
            enforce_min_max_values=True,
            epochs=epochs,
            verbose=True,
            default_distribution = "gamma"
        )
        synthesizer.fit(no_label)  # Let CopulaGAN handle normalization
        return synthesizer



    def train(self,dataset_name = "Chinatown", epochs=1000, nr_samples=10)->None:
        # We will train one synth per class


        grouped_data = load_data_by_class(dataset_name=dataset_name, dataset_type="TRAIN")

        for class_name, group in grouped_data:
            synthesizer = self.train_class(dataset_name=dataset_name,group=group, epochs=epochs)
            synth_data = synthesizer.sample(nr_samples).to_numpy()
            save_synthetic_data(synth_data=synth_data,dataset_name=dataset_name,generator_name=self.NAME,class_label=class_name,epochs=epochs)



def run():
    epochs = 5
    dataset_name = "Beef"
    gan = Copula_GAN()
    gan.train(dataset_name=dataset_name, epochs=epochs)
    load_and_visualize(dataset_name=dataset_name, generator_name=Copula_GAN.NAME, epochs=epochs)


if __name__ == "__main__":
   run()
