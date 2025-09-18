import pandas as pd
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import Metadata
import os
import numpy as np
from pathlib import Path


from matplotlib import pyplot as plt

from Generators.base_generator import BaseGenerator
from utils.dataset import load_data_by_class, get_metadata,get_metadata_no_label, load_dataset
from utils.save import save_synthetic_data, load_synthetic_data
from utils.viz import load_and_visualize
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import binary_crossentropy
from Generators.helperFiles.ConditionalGan import ConditionalGAN

import tsgm


def define_model(T, y):

    seq_len = T
    feat_dim = 1
    latent_dim = T//5
    output_dim = y  # Binary One Hot Encoded
    model_type = tsgm.models.architectures.zoo["cgan_lstm_3"]
    arch = model_type(
        seq_len=seq_len, feat_dim=feat_dim,
        latent_dim=latent_dim, output_dim=output_dim)
    arch_dict = arch.get()

    discriminator, generator = arch_dict["discriminator"], arch_dict["generator"]

    cGan = ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim, temporal=False)

    cGan.compile(d_optimizer=Adam(1e-4, beta_1=0.0, beta_2=0.9), g_optimizer=Adam(1e-4, beta_1=0.0, beta_2=0.9),
                 loss_fn=binary_crossentropy)
    return cGan

class CGAN(BaseGenerator):
    NAME = "CGAN"
    def __init__(self):
        super().__init__(name=self.NAME)
        self.model = None
        self.scaler_minmax = True
        self.label_encoder = None



    def _preprocess_label(self,dataset_name):
        data = load_dataset(dataset_name, "TRAIN")

        # Suppose labels is a 1D array of length N
        labels = data.iloc[:, 0].to_numpy()  # shape (N,)

        self.label_encoder = OneHotEncoder()
        labels = self.label_encoder.fit_transform(labels.reshape(-1, 1)).toarray()
        labels = np.array(labels, dtype=np.float32)

        return labels

    def _preprocess_ts(self,dataset_name):
        data = load_dataset(dataset_name, "TRAIN")
        ts_data = data.iloc[:, 1:]
        ts_data = ts_data.to_numpy(dtype=np.float32)
        org_data_shape = ts_data.shape
        N = org_data_shape[0]
        T = org_data_shape[1]


        if self.scaler_minmax:
            scaler = MinMaxScaler()
            ts_data = scaler.fit_transform(ts_data)
            ts_data = ts_data.reshape(N, T, 1)
        else:
            ts_data = ts_data.reshape(N, T, 1)
            scaler = tsgm.utils.TSFeatureWiseScaler()
            ts_data = scaler.fit_transform(ts_data)

        self.scaler = scaler
        return ts_data

    def sample(self, T, num_classes, nr_samples):
        import tensorflow as tf
        def distribute_items(n, k):
            """Distribute n items across k bags as evenly as possible"""
            return np.array_split(range(n), k)

        def distribute_counts(n, k):
            """Get just the counts per bag"""
            return [len(bag) for bag in distribute_items(n, k)]

        all_samples = []

        for i, samples in enumerate(distribute_counts(nr_samples, num_classes)):
            for _ in range(samples):
                one_hot = [0.0 for _ in range(num_classes)]
                one_hot[i] = 1.0
                all_samples.append(one_hot)

        tensor = tf.convert_to_tensor(all_samples)

        gen_samples = self.model.generate(tensor)

        gen_data = np.array(gen_samples)
        if self.scaler_minmax:
            gen_data = gen_data.reshape(nr_samples, T)
        gen_data = self.scaler.inverse_transform(gen_data)


        if not self.scaler_minmax:
            gen_data = gen_data.reshape(nr_samples, T)

        return gen_data, self.label_encoder.inverse_transform(tensor).flatten()

    def train(self,dataset_name = "Chinatown", epochs=1000, nr_samples=10)->None:

        local_epochs = epochs*10
        ts_data = self._preprocess_ts(dataset_name)
        labels = self._preprocess_label(dataset_name)

        N, T, F = ts_data.shape
        y = labels.shape[1]

        self.model = define_model(T,y)

        self.model.fit(x=ts_data, y=labels, epochs=local_epochs, batch_size=4,verbose=False)

        gen_data, gen_labels = self.sample(T=T, num_classes=y, nr_samples=nr_samples)
        unique_labels = np.unique(gen_labels, axis=0)
        for label in unique_labels:
            save_synthetic_data(synth_data=gen_data[gen_labels==label],dataset_name=dataset_name, generator_name=self.NAME, class_label=label,epochs=epochs)



def run():
    epochs = 5
    dataset_name = "Chinatown"
    gan = CGAN()
    gan.train(dataset_name=dataset_name, epochs=epochs,nr_samples=100)
    load_and_visualize(dataset_name=dataset_name, generator_name=gan.NAME, epochs=epochs)


if __name__ == "__main__":
   run()
