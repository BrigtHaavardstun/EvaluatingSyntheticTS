from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition
import pandas as pd

from matplotlib import pyplot as  plt
from utils.dataset import load_dataset, get_metadata,load_labels
from utils.save import save_synthetic_data, load_synthetic_data
from utils.viz import load_and_visualize

from Generators.base_generator import BaseGenerator
import numpy as np
import os


class CT_GAN(BaseGenerator):
    NAME = "CT_GAN"
    def __init__(self):
        super().__init__(name=self.NAME)

    def train(self,dataset_name, epochs=1000, nr_samples=None):

        data = load_dataset(dataset_name=dataset_name, dataset_type="TRAIN")
        data.columns = data.columns.astype(str)

        if nr_samples is None:
            nr_samples = len(data)




        # Create metadata
        metadata = get_metadata(dataset_name)
        synthesizer = CTGANSynthesizer(
            metadata,  # required
            enforce_min_max_values=True,
            enforce_rounding=False,
            epochs=epochs,
            verbose=True,
        )
        synthesizer.fit(data)

        labels = load_labels(dataset_name, dataset_type="TRAIN")



        for label in labels:
            sample =  synthesizer.sample_from_conditions([Condition({data.columns[0]:label},nr_samples)])
            rows = sample.iloc[:,1:]
            rows = rows.values
            save_synthetic_data(synth_data=rows,dataset_name=dataset_name, generator_name=self.NAME, class_label=label,epochs=epochs)



def run():
    dataset_name = "Chinatown"
    epochs = 50
    gen = CT_GAN()
    gen.train(dataset_name=dataset_name, epochs=epochs)
    load_and_visualize(dataset_name=dataset_name, generator_name=gen.NAME, epochs=epochs)

if __name__ == "__main__":
    run()
