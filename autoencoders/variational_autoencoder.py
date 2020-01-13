import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
from .base_autoencoder import BaseAutoencoder

class VariationalAutoencoder(BaseAutoencoder):
  def __init__(self, input_dims, latent_dim):
    super(VariationalAutoencoder, self).__init__(input_dims, latent_dim)
    
    self.inference_net = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=input_dims),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1000, activation='relu'),
      # tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dense(500, activation='relu'),
      # tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dense(250, activation='relu'),
      # tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dense(latent_dim + latent_dim)
    ])

    self.generative_net = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
      tf.keras.layers.Dense(250, activation='relu'),
      # tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dense(500, activation='relu'),
      # tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dense(1000, activation='relu'),
      # tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dense(np.prod(input_dims)),
      tf.keras.layers.Reshape(input_dims)
    ])

  @tf.function
  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.nn.sigmoid(logits)
      return probs
    return logits

  def reconstruct(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    z = self.reparameterize(mean, logvar)
    return self.decode(z, apply_sigmoid=True)

  @tf.function
  def compute_loss(self, x):
    mean, logvar = self.encode(x)
    z = self.reparameterize(mean, logvar)
    x_logits = self.decode(z)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_logits)
    logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    
    p_z = tfp.distributions.Normal(loc=np.zeros(self.latent_dim, dtype=np.float32), scale=np.ones(self.latent_dim, dtype=np.float32))
    q_z = tfp.distributions.Normal(loc=mean, scale=tf.exp(logvar * 0.5))
    
    kl = tf.reduce_sum(tfp.distributions.kl_divergence(q_z, p_z), axis=1)
    return tf.reduce_mean(kl - logpx_z)

  # def compute_reconstruction_error(self, x):
  #   mean, logvar = self.encode(x)
  #   z = self.reparameterize(mean, logvar)
  #   x_reconstruction = self.decode(z, apply_sigmoid=True)
  #   return tf.reduce_mean(tf.reduce_sum(tf.square(x - x_reconstruction), axis=[1, 2, 3]))