# timegan.py (TF2 version)
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from timeGAN.utils import extract_time, rnn_cell, random_generator, batch_generator

# --- Small helper layers / blocks ---
def make_rnn_stack(module_name, hidden_dim, num_layers, return_sequences=True):
    """Return a tf.keras RNN layer stacking the appropriate cells."""
    cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
    # Wrap cells in keras RNN. If cells are RNNCell objects (GRUCell/LSTMCell) this works.
    return layers.RNN(cells, return_sequences=return_sequences)

class Embedder(Model):
    def __init__(self, module, hidden_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.rnn = make_rnn_stack(module, hidden_dim, num_layers, return_sequences=True)
        self.dense = layers.Dense(hidden_dim, activation='sigmoid')

    def call(self, x, training=False):
        h = self.rnn(x)
        return self.dense(h)

class Recovery(Model):
    def __init__(self, module, hidden_dim, num_layers, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.rnn = make_rnn_stack(module, hidden_dim, num_layers, return_sequences=True)
        self.dense = layers.Dense(output_dim, activation='sigmoid')

    def call(self, h, training=False):
        r = self.rnn(h)
        return self.dense(r)

class Generator(Model):
    def __init__(self, module, hidden_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.rnn = make_rnn_stack(module, hidden_dim, num_layers, return_sequences=True)
        self.dense = layers.Dense(hidden_dim, activation='sigmoid')

    def call(self, z, training=False):
        g = self.rnn(z)
        return self.dense(g)

class Supervisor(Model):
    def __init__(self, module, hidden_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        # supervisor has num_layers-1 as in original paper
        num = max(1, num_layers - 1)
        self.rnn = make_rnn_stack(module, hidden_dim, num, return_sequences=True)
        self.dense = layers.Dense(hidden_dim, activation='sigmoid')

    def call(self, h, training=False):
        s = self.rnn(h)
        return self.dense(s)

class Discriminator(Model):
    def __init__(self, module, hidden_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.rnn = make_rnn_stack(module, hidden_dim, num_layers, return_sequences=True)
        self.dense = layers.Dense(1, activation=None)  # logits

    def call(self, h, training=False):
        d = self.rnn(h)
        return self.dense(d)

# --- Training function ---
def timegan(ori_data, parameters, verbose=False):
    """
    TF2 implementation of TimeGAN training loop.
    ori_data: list of numpy arrays, each shape (seq_len, dim)
    parameters: dict with keys:
      hidden_dim, num_layer, iterations, batch_size, module (e.g. 'gru'/'lstm'), z_dim (optional)
    """
    # extract some basic parameters
    ori_time, max_seq_len = extract_time(ori_data)
    no, _, dim = np.asarray(ori_data, dtype=object).shape if False else (len(ori_data), None, None)
    # we will infer dim from first sequence
    dim = ori_data[0].shape[1]
    no = len(ori_data)

    # MinMax scaler (same semantics as original)
    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    # Convert list-of-arrays into single padded numpy array for normalization
    # We'll create a big array (no, max_seq_len, dim) filled with zeros and then mask
    padded = np.zeros((no, max_seq_len, dim), dtype=np.float32)
    for i in range(no):
        L = ori_data[i].shape[0]
        padded[i, :L, :] = ori_data[i]
    padded_norm, min_val, max_val = MinMaxScaler(padded.copy())

    # Overwrite ori_data to be padded normalized arrays and keep ori_time
    # We'll keep ori_data as list of numpy arrays for other utils, but also keep padded_norm for testing
    ori_data_norm_list = [padded_norm[i, :ori_time[i], :] for i in range(no)]

    # Parameters
    hidden_dim = parameters.get('hidden_dim', 24)
    num_layers = parameters.get('num_layer', 3)
    iterations = parameters.get('iterations', 10000)
    batch_size = parameters.get('batch_size', 128)
    module = parameters.get('module', 'gru')
    z_dim = parameters.get('z_dim', dim)  # default z_dim = data dim
    gamma = 1.0

    # Build models
    embedder = Embedder(module, hidden_dim, num_layers)
    recovery = Recovery(module, hidden_dim, num_layers, dim)
    generator = Generator(module, hidden_dim, num_layers)
    supervisor = Supervisor(module, hidden_dim, num_layers)
    discriminator = Discriminator(module, hidden_dim, num_layers)

    # Optimizers and losses
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse_loss = tf.keras.losses.MeanSquaredError()
    opt_e = tf.keras.optimizers.legacy.Adam()
    opt_g = tf.keras.optimizers.legacy.Adam()
    opt_d = tf.keras.optimizers.legacy.Adam()

    # Helper for moment loss
    def two_moments_loss(x_real, x_fake):
        # both are [batch, time, dim]
        mean_real = tf.reduce_mean(x_real, axis=0)
        mean_fake = tf.reduce_mean(x_fake, axis=0)
        var_real = tf.math.reduce_variance(x_real, axis=0)
        var_fake = tf.math.reduce_variance(x_fake, axis=0)
        v1 = tf.reduce_mean(tf.abs(tf.sqrt(var_fake + 1e-6) - tf.sqrt(var_real + 1e-6)))
        v2 = tf.reduce_mean(tf.abs(mean_fake - mean_real))
        return v1 + v2

    # Training steps
    @tf.function
    def train_embedder_step(x_mb):
        with tf.GradientTape() as tape:
            h = embedder(x_mb, training=True)
            x_tilde = recovery(h, training=True)
            loss_t0 = mse_loss(x_mb, x_tilde)
            loss_e = 10.0 * tf.sqrt(loss_t0 + 1e-8)
        grads = tape.gradient(loss_e, embedder.trainable_variables + recovery.trainable_variables)
        opt_e.apply_gradients(zip(grads, embedder.trainable_variables + recovery.trainable_variables))
        return loss_t0

    @tf.function
    def train_supervised_generator_step(z_mb, x_mb):
        # minimize G_loss_S = mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
        with tf.GradientTape() as tape:
            e_hat = generator(z_mb, training=True)
            h_hat = supervisor(e_hat, training=True)
            # compute H from real X
            h = embedder(x_mb, training=False)
            h_hat_supervise = supervisor(h, training=True)
            # shift comparison
            loss_s = mse_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
        grads = tape.gradient(loss_s, generator.trainable_variables + supervisor.trainable_variables)
        opt_g.apply_gradients(zip(grads, generator.trainable_variables + supervisor.trainable_variables))
        return loss_s

    @tf.function
    def train_discriminator_step(x_mb, z_mb):
        with tf.GradientTape() as tape:
            # real
            h = embedder(x_mb, training=False)
            y_real = discriminator(h, training=True)
            # fake
            e_hat = generator(z_mb, training=False)
            h_hat = supervisor(e_hat, training=False)
            y_fake = discriminator(h_hat, training=True)
            # also E_hat discriminator
            y_fake_e = discriminator(e_hat, training=True)
            d_loss_real = bce_loss(tf.ones_like(y_real), y_real)
            d_loss_fake = bce_loss(tf.zeros_like(y_fake), y_fake)
            d_loss_fake_e = bce_loss(tf.zeros_like(y_fake_e), y_fake_e)
            d_loss = d_loss_real + d_loss_fake + gamma * d_loss_fake_e
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        opt_d.apply_gradients(zip(grads, discriminator.trainable_variables))
        return d_loss

    @tf.function
    def train_generator_step(x_mb, z_mb):
        with tf.GradientTape(persistent=True) as tape:
            # adversarial losses
            e_hat = generator(z_mb, training=True)
            h_hat = supervisor(e_hat, training=True)
            y_fake = discriminator(h_hat, training=False)
            y_fake_e = discriminator(e_hat, training=False)
            g_loss_u = bce_loss(tf.ones_like(y_fake), y_fake)
            g_loss_u_e = bce_loss(tf.ones_like(y_fake_e), y_fake_e)
            # supervised loss
            h = embedder(x_mb, training=False)
            h_hat_supervise = supervisor(h, training=True)
            g_loss_s = mse_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
            # synthetic data
            x_hat = recovery(h_hat, training=True)
            g_loss_v = two_moments_loss(x_mb, x_hat)
            g_loss = g_loss_u + gamma * g_loss_u_e + 100.0 * tf.sqrt(g_loss_s + 1e-8) + 100.0 * g_loss_v
        # update generator + supervisor variables
        vars_gs = generator.trainable_variables + supervisor.trainable_variables
        grads = tape.gradient(g_loss, vars_gs)
        opt_g.apply_gradients(zip(grads, vars_gs))
        # update embedder to help reconstruction (E_loss part)
        with tf.GradientTape() as tape2:
            h = embedder(x_mb, training=True)
            x_tilde = recovery(h, training=True)
            loss_t0 = mse_loss(x_mb, x_tilde)
            e_loss = 10.0 * tf.sqrt(loss_t0 + 1e-8) + 0.1 * g_loss_s
        grads_e = tape2.gradient(e_loss, embedder.trainable_variables + recovery.trainable_variables)
        opt_e.apply_gradients(zip(grads_e, embedder.trainable_variables + recovery.trainable_variables))
        return g_loss_u, g_loss_s, g_loss_v

    # --- Training schedule ---
    if verbose:
        print("Starting TF2 TimeGAN training...")
    # 1) Embedding network training (pretrain)
    if verbose:
        print("Embedding pretrain")
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # X_mb shape (batch, seq_len, dim)
        # pad/truncate X_mb to max_seq_len if needed (batch_generator returns padded)
        loss_t0 = train_embedder_step(tf.convert_to_tensor(X_mb, dtype=tf.float32))
        if itt % 1000 == 0 and verbose:
            print(f"embed step {itt}/{iterations}, e_loss_t0: {np.sqrt(float(loss_t0)):.5f}")

    # 2) Training with supervised loss only
    if verbose:
        print("Supervised generator training")
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # create Z_mb consistent with batch sizes and lengths
        Z_mb = random_generator(batch_size, z_dim, T_mb, X_mb.shape[1])
        loss_s = train_supervised_generator_step(tf.convert_to_tensor(Z_mb, dtype=tf.float32),
                                                 tf.convert_to_tensor(X_mb, dtype=tf.float32))
        if itt % 1000 == 0 and verbose:
            print(f"supervised step {itt}/{iterations}, s_loss: {np.sqrt(float(loss_s)):.5f}")

    # 3) Joint training
    if verbose:
        print("Joint adversarial training")
    for itt in range(iterations):
        # two generator updates per discriminator update
        for _ in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, X_mb.shape[1])
            g_u, g_s, g_v = train_generator_step(tf.convert_to_tensor(X_mb, dtype=tf.float32),
                                                 tf.convert_to_tensor(Z_mb, dtype=tf.float32))
        # train discriminator
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, X_mb.shape[1])
        d_loss = train_discriminator_step(tf.convert_to_tensor(X_mb, dtype=tf.float32),
                                          tf.convert_to_tensor(Z_mb, dtype=tf.float32))
        if itt % 1000 == 0 and verbose:
            print(f"joint step {itt}/{iterations}, d_loss: {float(d_loss):.5f}, g_u: {float(g_u):.5f}, g_s: {np.sqrt(float(g_s)):.5f}, g_v: {float(g_v):.5f}")

    # --- Synthetic data generation ---
    if verbose:
        print("Generating synthetic data")
    # generate Z for the whole dataset size 'no'
    # We'll use padded z for the largest sequence length in ori_time (which is max_seq_len)
    Z_mb_full = random_generator(no, z_dim, ori_time, max_seq_len)
    X_hat_full = generator(tf.convert_to_tensor(Z_mb_full, dtype=tf.float32), training=False)
    H_hat = supervisor(X_hat_full, training=False)
    X_hat_full = recovery(H_hat, training=False).numpy()  # shape (no, max_seq_len, dim)

    # convert back to list of variable-length numpy arrays and renormalize
    generated = []
    for i in range(no):
        L = int(ori_time[i])
        seq = X_hat_full[i, :L, :]
        # renormalize
        seq = seq * max_val + min_val
        generated.append(seq.astype(np.float32))

    return generated