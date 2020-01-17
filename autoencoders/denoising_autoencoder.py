import tensorflow as tf
import numpy as np
from .base_autoencoder import BaseAutoencoder


class DenoisingAutoencoder(BaseAutoencoder):
  def __init__(self, input_dims, latent_dim, corruption_proportion=0.1):
    super(DenoisingAutoencoder, self).__init__(input_dims, latent_dim)
    self.corruption_proportion = corruption_proportion

    self.encoder_net = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=input_dims),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(corruption_proportion),
      tf.keras.layers.Dense(1000, activation='relu'),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dense(250, activation='relu'),
      tf.keras.layers.Dense(latent_dim)
    ])

    self.decoder_net = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
      tf.keras.layers.Dense(250, activation='relu'),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dense(1000, activation='relu'),
      tf.keras.layers.Dense(np.prod(input_dims), activation='sigmoid'),
      tf.keras.layers.Reshape(target_shape=input_dims)
    ])

  @tf.function
  def encode(self, x):
    return self.encoder_net(x, training=True)

  def decode(self, z):
    return self.decoder_net(z)

  @tf.function
  def compute_loss(self, x):
    z = self.encode(x)
    x_reconstructed = self.decode(z)
    return tf.reduce_mean(tf.square(x - x_reconstructed))

  def reconstruct(self, x):
    z = self.encoder_net(x, training=True)
    x_reconstructed = self.decoder_net(z)
    return x_reconstructed