from abc import abstractmethod
from .base_autoencoder import BaseAutoencoder
import tensorflow as tf
import numpy as np
import time
from .utils import compute_mmd

class BaseInfoVariationalAutoencoder(BaseAutoencoder):
  def __init__(self, input_dims, latent_dim, hidden_dim=1024, alpha=0.1):
    super(BaseInfoVariationalAutoencoder, self).__init__(input_dims, latent_dim)
    self.hidden_dim = hidden_dim
    self.alpha = alpha

    self.inference_net = self.make_inference_net()
    self.generative_net = self.make_generative_net()

  @abstractmethod
  def make_inference_net(self):
    raise NotImplementedError()

  @abstractmethod
  def make_generative_net(self):
    raise NotImplementedError()

  @tf.function
  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar
    # return self.inference_net(x)

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.nn.sigmoid(logits)
      return probs
    return logits

  @tf.function
  def compute_loss(self, x):
    mean, logvar = self.encode(x)
    z = self.reparameterize(mean, logvar)
    # z = self.encode(x)
    probs = self.decode(z, apply_sigmoid=True)

    # loss_nll = tf.reduce_mean(tf.reduce_sum(tf.square(x + 0.0 - probs), axis=[1, 2, 3]))
    loss_nll = tf.reduce_mean(tf.square(x + 0.0 - probs))

    true_samples = tf.random.normal(z.shape)
    loss_mmd = compute_mmd(true_samples, z)

    return loss_nll + loss_mmd

  def reconstruct(self, x):
    mean, logvar = self.encode(x)
    z = self.reparameterize(mean, logvar)
    # z = self.encode(x)
    return self.decode(z, apply_sigmoid=True)

class MnistInfoVariationalAutoencoder(BaseInfoVariationalAutoencoder):
  def make_inference_net(self):
    return tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=self.input_dims),
      tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2)),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      tf.keras.layers.Conv2D(128, kernel_size=3, strides=(2, 2)),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(self.hidden_dim),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      # No activation
      tf.keras.layers.Dense(2 * self.latent_dim)
    ])

  def make_generative_net(self):
    return tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(self.latent_dim)),
      tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
      tf.keras.layers.Dense(7*7*128, activation='relu'),
      tf.keras.layers.Reshape(target_shape=(7, 7, 128)),
      tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='SAME', activation='relu'),
      # No activation
      tf.keras.layers.Conv2DTranspose(filters=self.input_dims[-1], kernel_size=3, strides=(2, 2), padding='SAME')
    ])

class Cifar10InfoVariationalAutoencoder(BaseInfoVariationalAutoencoder):
  def make_inference_net(self):
    return tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=self.input_dims),
      tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2)),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      tf.keras.layers.Conv2D(128, kernel_size=3, strides=(2, 2)),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(self.hidden_dim),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      # No activation
      tf.keras.layers.Dense(2 * self.latent_dim)
    ])

  def make_generative_net(self):
    return tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(self.latent_dim)),
      tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
      tf.keras.layers.Dense(8*8*128, activation='relu'),
      tf.keras.layers.Reshape(target_shape=(8, 8, 128)),
      tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='SAME', activation='relu'),

      # No activation
      tf.keras.layers.Conv2DTranspose(filters=self.input_dims[-1], kernel_size=3, strides=(2, 2), padding='SAME')
    ])