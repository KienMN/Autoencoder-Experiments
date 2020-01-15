from __future__ import absolute_import, division, print_function
from abc import abstractmethod
import tensorflow as tf
import time
from keras.callbacks import CallbackList
from .ae_callbacks import SaveLossesCallback
import numpy as np


class BaseAutoencoder(tf.keras.Model):
  def __init__(self, input_dims, latent_dim):
    super(BaseAutoencoder, self).__init__()
    self.input_dims = input_dims
    self.latent_dim = latent_dim

  @abstractmethod
  def compute_loss(self, x):
    pass

  def compute_reconstruction_error(self, x, raxis=[1, 2, 3]):
    x_reconstruction = self.reconstruct(x)
    return tf.reduce_mean(tf.reduce_sum(tf.square(x - x_reconstruction), axis=raxis))

  def reconstruct(self, x):
    pass

  def compute_apply_gradients(self, x, optimizer):
    with tf.GradientTape() as tape:
      loss = self.compute_loss(x)
    gradients = tape.gradient(loss, self.trainable_variables)
    # gradients = [gradient if gradient is not None else tf.zeros_like(var) for var, gradient in zip(self.trainable_variables, gradients)]
    optimizer.apply_gradients(zip(gradients, self.trainable_variables))

  def _check_tf_dataset_instance(self, x, batch_size=32):
    if not isinstance(x, tf.data.Dataset):
      if isinstance(x, np.ndarray):
        x = tf.data.Dataset.from_tensor_slices(x).shuffle(x.shape[0]).batch(batch_size)
    return x

  def fit(self,
          train_dataset,
          test_dataset=None,
          batch_size=32,
          epochs=1,
          optimizer=tf.keras.optimizers.Adam(),
          callbacks=None):

    # log_dir = './logs/test/'
    # writer = tf.summary.create_file_writer(log_dir)
    train_loss = tf.keras.metrics.Mean()
    train_reconstruction_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()
    test_reconstruction_loss = tf.keras.metrics.Mean()

    train_dataset = self._check_tf_dataset_instance(train_dataset, batch_size=batch_size)

    callbacks = CallbackList(callbacks)
    callback_model = self._get_callback_model()
    callbacks.set_model(callback_model)

    for epoch in range(1, epochs + 1):  
      callbacks.on_epoch_begin(epoch)
      epoch_logs = {}

      for x_train in train_dataset:
        self.compute_apply_gradients(x_train, optimizer=optimizer)
        train_loss(self.compute_loss(x_train))
        train_reconstruction_loss(self.compute_reconstruction_error(x_train))

      epoch_logs['train_loss'] = train_loss.result().numpy()
      epoch_logs['train_reconstruction_error'] = train_reconstruction_loss.result().numpy()
      train_loss.reset_states()
      train_reconstruction_loss.reset_states()

      if test_dataset is not None:
        test_dataset = self._check_tf_dataset_instance(test_dataset, batch_size=batch_size)
        for x_test in test_dataset:
          test_loss(self.compute_loss(x_test))
          test_reconstruction_loss(self.compute_reconstruction_error(x_test))
        
        epoch_logs['test_loss'] = test_loss.result().numpy()
        epoch_logs['test_reconstruction_error'] = test_reconstruction_loss.result().numpy()
        test_loss.reset_states()
        test_reconstruction_loss.reset_states()

      callbacks.on_epoch_end(epoch, logs=epoch_logs)