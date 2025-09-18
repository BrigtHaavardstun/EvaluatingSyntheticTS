import pandas as pd
import os
from matplotlib import pyplot as plt
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata

from utils.dataset import load_data_by_class, get_metadata_no_label
from utils.save import save_synthetic_data,load_synthetic_data
from utils.viz import load_and_visualize

from Generators.base_generator import BaseGenerator

class TVAE(BaseGenerator):
    NAME = "TVAE"

    def __init__(self):
        super().__init__(name=self.NAME)

    def train_class(self,group, metadata, epochs=100):
        print("Training VAE synthesizer...")
        no_label = group.iloc[:,1:] # We don't want the labels in the synth
        synthesizer = TVAESynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False, epochs=epochs, verbose=True)
        synthesizer.fit(no_label)
        return synthesizer

    def train(self,dataset_name = "Chinatown", epochs=10,nr_samples=100):
        print("Loading data...")
        dataset_type = "TRAIN"
        metadata = get_metadata_no_label(dataset_name=dataset_name)
        by_class_group = load_data_by_class(dataset_name=dataset_name, dataset_type=dataset_type)

        for class_name, group in by_class_group:
            synthesizer = self.train_class(group=group, metadata=metadata, epochs=epochs)
            gen_data = synthesizer.sample(nr_samples).to_numpy()
            save_synthetic_data(synth_data=gen_data,dataset_name=dataset_name,generator_name=self.NAME,class_label=class_name,epochs=epochs)
            print("Done.")


def run():
    dataset_name = "Chinatown"
    epochs = 1000
    gen = TVAE()
    gen.train(dataset_name=dataset_name, epochs=epochs)
    load_and_visualize(dataset_name=dataset_name, generator_name=gen.NAME, epochs=epochs)

if __name__ == "__main__":
    run()
