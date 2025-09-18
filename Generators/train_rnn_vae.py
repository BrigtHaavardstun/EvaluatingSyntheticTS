# rvae.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Lambda,
    RepeatVector, TimeDistributed, Bidirectional
)
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.viz import load_and_visualize


class TimeSeriesVAE:
    def __init__(self, timesteps=30, latent_dim=10, hidden_dim=64, beta=0.25):
        self.timesteps = timesteps
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.scaler = None
        self._build()

    def _build(self):
        # === Encoder ===
        inp = Input(shape=(self.timesteps, 1), name='encoder_input')
        h = LSTM(self.hidden_dim, return_sequences=True)(inp)
        h = LSTM(self.hidden_dim, return_sequences=True)(h)
        h = LSTM(self.hidden_dim, return_sequences=False)(h)

        z_mean = Dense(self.latent_dim, name='z_mean')(h)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(h)

        def sampling(args):
            mu, logv = args
            eps = K.random_normal(shape=(K.shape(mu)[0], self.latent_dim))
            return mu + K.exp(0.5 * logv) * eps

        z = Lambda(sampling, name='z')([z_mean, z_log_var])

        self.encoder = Model(inp, [z_mean, z_log_var, z], name='encoder')

        # === Decoder ===
        dec_input = RepeatVector(self.timesteps)(z)
        dec_h = LSTM(self.hidden_dim, return_sequences=True)(dec_input)
        dec_h = LSTM(self.hidden_dim, return_sequences=True)(dec_h)
        dec_h = LSTM(self.hidden_dim, return_sequences=True)(dec_h)
        out = TimeDistributed(Dense(1))(dec_h)


        decoder_input = Input(shape=(self.latent_dim,), name='z_sampling')
        dec_input2 = RepeatVector(self.timesteps)(decoder_input)
        dec_h2 = LSTM(self.hidden_dim, return_sequences=True)(dec_input2)
        dec_h2 = LSTM(self.hidden_dim, return_sequences=True)(dec_h2)
        dec_h2 = LSTM(self.hidden_dim, return_sequences=True)(dec_h2)

        dec_out2 = TimeDistributed(Dense(1))(dec_h2)
        self.generator = Model(decoder_input, dec_out2, name='decoder')

        vae = Model(inp, out, name='rvae')

        recon_loss = mse(K.flatten(inp), K.flatten(out))
        recon_loss = K.mean(recon_loss) * self.timesteps
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        vae.add_loss(recon_loss + self.beta * kl_loss)
        vae.compile(optimizer='adam')
        self.vae = vae

    def train(self, x, epochs=50, batch_size=64, validation_data=None):
        local_epochs = epochs*10
        # === Normalize ===
        self.scaler = StandardScaler()
        n_samples = x.shape[0]
        x_flat = x.reshape(n_samples, -1)
        x_scaled_flat = self.scaler.fit_transform(x_flat)
        x_scaled = x_scaled_flat.reshape(n_samples, self.timesteps, 1)

        # === Fit VAE ===
        self.vae.fit(
            x_scaled, None,
            epochs=local_epochs,
            batch_size=batch_size,
            validation_data=(validation_data, None) if validation_data is not None else None,
            verbose = False
        )

    def generate(self, n_samples=1):
        z = np.random.normal(size=(n_samples, self.latent_dim))
        generated_scaled = self.generator.predict(z)

        if self.scaler:
            flat_gen = generated_scaled.reshape(n_samples, -1)
            restored = self.scaler.inverse_transform(flat_gen)
            return restored.reshape(n_samples, self.timesteps, 1)
        else:
            raise RuntimeError("Scaler is not fitted. Call train() first.")

import numpy as np
from matplotlib import pyplot as plt
from utils.dataset import load_data_by_class
from utils.save import save_synthetic_data,load_synthetic_data
from Generators.base_generator import BaseGenerator
import numpy as np


class RNN_VAE(BaseGenerator):
    NAME = "RNN_VAE"
    def __init__(self):
        super().__init__(name=self.NAME)

    def train(self,dataset_name, epochs=1000, nr_samples=None):

        grouped_by_class = load_data_by_class(dataset_name=dataset_name)



        for class_label, group_df in grouped_by_class:
            no_label = group_df.iloc[:, 1:].to_numpy()  # We don't want the labels in the synth
            timesteps = len(no_label[0])
            vae = TimeSeriesVAE(timesteps=timesteps, latent_dim=min(3 * timesteps,50), hidden_dim=min(3 * timesteps,50), beta=1)
            vae.train(no_label, epochs=epochs, batch_size=32)

            # Generate synthetic sequences
            if nr_samples is None:
                nr_samples = len(no_label)
            synth_data = vae.generate(n_samples=nr_samples)
            save_synthetic_data(dataset_name=dataset_name,synth_data=synth_data,generator_name=self.NAME,class_label=class_label,epochs=epochs)


def run():
    dataset_name = "Chinatown"
    epochs = 5000
    gen = RNN_VAE()
    gen.train(dataset_name, epochs)
    load_and_visualize(dataset_name=dataset_name, generator_name=gen.NAME, epochs=epochs)

if __name__ == '__main__':
    run()