import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from timeGAN.timegan import  timegan
from utils.dataset import load_data_by_class, get_metadata_no_label
from utils.save import save_synthetic_data, load_synthetic_data
from utils.viz import load_and_visualize

from Generators.base_generator import BaseGenerator
from sklearn.preprocessing import MinMaxScaler

class TimeGAN(BaseGenerator):
    NAME = "timeGAN"

    def __init__(self):
        super().__init__(name=self.NAME)

    def train_class(self,group, epochs=100,nr_samples=100):
        no_label = group.iloc[:,1:] # We don't want the labels in the synth
        no_label = np.array(no_label)
        no_label = no_label.reshape(no_label.shape[0], no_label.shape[1],1)
        # === Normalize ===
        scaler = MinMaxScaler()
        n_samples = no_label.shape[0]
        x_flat = no_label.reshape(n_samples, -1)
        x_scaled_flat = scaler.fit_transform(x_flat)
        x_scaled = x_scaled_flat.reshape(n_samples, no_label.shape[1], 1)

        rows, ts_length, dims = no_label.shape
        parameters = {
            'hidden_dim': 3*ts_length,
            'num_layer': 3,
            'iterations': epochs,
            'batch_size':min(32,rows),
            'module': 'gru',

        }
        gen_data = timegan(x_scaled, parameters=parameters )
        gen_data = np.array(gen_data)

        flatten_gen_data = gen_data.reshape(-1,no_label.shape[1])
        gen_data = scaler.inverse_transform(flatten_gen_data)
        gen_data = gen_data.reshape(rows,ts_length,1)

        return gen_data

    def train(self,dataset_name, epochs=10, nr_samples=100):
        dataset_type = "TRAIN"
        by_class_group = load_data_by_class(dataset_name=dataset_name, dataset_type=dataset_type)
        for class_name, group in by_class_group:

            gen_data = self.train_class(group=group, epochs=epochs)

            # Revert scaling
            save_synthetic_data(synth_data=gen_data,dataset_name=dataset_name,generator_name=self.NAME,class_label=class_name,epochs=epochs)

def run():
    dataset_name = "Chinatown"
    epochs = 10
    gen = TimeGAN()
    gen.train(dataset_name=dataset_name, epochs=epochs)
    load_and_visualize(dataset_name=dataset_name, generator_name=gen.NAME, epochs=epochs)


if __name__ == "__main__":
    run()

