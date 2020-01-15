from .base_autoencoder import BaseAutoencoder
import tensorflow as tf
import numpy as np
import time
from keras.callbacks import CallbackList

class AdversarialAutoencoder(BaseAutoencoder):
  def __init__(self,
              input_dims,
              latent_dim,
              hidden_dim=1000,
              alpha=0.3,
              drop_rate=0.5,
              ae_loss_weight=1.0,
              gen_loss_weight=1.0,
              dc_loss_weight=1.0):

    super(AdversarialAutoencoder, self).__init__(input_dims, latent_dim)
    
    self.hidden_dim = hidden_dim
    self.alpha = alpha
    self.drop_rate = drop_rate

    self.ae_loss_weight = ae_loss_weight
    self.gen_loss_weight = gen_loss_weight
    self.dc_loss_weight = dc_loss_weight

    self.encoder_model = self.make_encoder_model()
    self.decoder_model = self.make_decoder_model()
    self.discriminator_model = self.make_discriminator_model()

    self.accuracy = tf.keras.metrics.BinaryAccuracy()

    # Cyclic learning rate
    self.base_lr = 0.00025
    self.max_lr = 0.0025

    self.ae_optimizer = tf.keras.optimizers.Adam(self.base_lr)
    self.dc_optimizer = tf.keras.optimizers.Adam(self.base_lr)
    self.gen_optimizer = tf.keras.optimizers.Adam(self.base_lr)

  def make_encoder_model(self):
    return tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(self.input_dims)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(self.hidden_dim),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      tf.keras.layers.Dropout(rate=self.drop_rate),
      tf.keras.layers.Dense(self.hidden_dim),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      tf.keras.layers.Dropout(rate=self.drop_rate),
      tf.keras.layers.Dense(self.latent_dim)
    ])

  def make_decoder_model(self):
    return tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
      tf.keras.layers.Dense(self.hidden_dim),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      tf.keras.layers.Dropout(rate=self.drop_rate),
      tf.keras.layers.Dense(self.hidden_dim),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      tf.keras.layers.Dropout(rate=self.drop_rate),
      tf.keras.layers.Dense(np.prod(self.input_dims), activation='sigmoid'),
      tf.keras.layers.Reshape(target_shape=self.input_dims)
    ])

  def make_discriminator_model(self):
    return tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
      tf.keras.layers.Dense(self.hidden_dim),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      tf.keras.layers.Dropout(rate=self.drop_rate),
      tf.keras.layers.Dense(self.hidden_dim),
      tf.keras.layers.LeakyReLU(alpha=self.alpha),
      tf.keras.layers.Dropout(rate=self.drop_rate),
      tf.keras.layers.Dense(1)
    ])

  @tf.function
  def encode(self, x, training=False):
    return self.encoder_model(x, training=training)

  def decode(self, z, training=False):
    return self.decoder_model(z, training=training)

  def autoencoder_loss(self, inputs, reconstruction):
    # mse = tf.keras.metrics.MeanSquaredError()
    # return self.ae_loss_weight * mse(inputs, reconstruction)
    # return self.ae_loss_weight * tf.reduce_mean(tf.reduce_sum(tf.square(inputs - reconstruction), axis=[1, 2, 3]))
    return self.ae_loss_weight * tf.reduce_mean(tf.square(inputs - reconstruction))

  def reconstruct(self, x):
    x_encoded = self.encoder_model(x, training=False)
    x_decoded = self.decoder_model(x_encoded, training=False)
    return x_decoded

  def discriminator_loss(self, real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return self.dc_loss_weight * (real_loss + fake_loss)

  def generator_loss(self, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return self.gen_loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)

  @tf.function
  def train_step(self, x):
    # Autoencoder
    with tf.GradientTape() as ae_tape:
      x_encoded = self.encoder_model(x, training=True)
      x_decoded = self.decoder_model(x_encoded, training=True)
      ae_loss = self.autoencoder_loss(x, x_decoded)

    ae_gradients = ae_tape.gradient(ae_loss, self.encoder_model.trainable_variables + self.decoder_model.trainable_variables)
    self.ae_optimizer.apply_gradients(zip(ae_gradients, self.encoder_model.trainable_variables + self.decoder_model.trainable_variables))

    # Discriminator
    with tf.GradientTape() as dc_tape:
      x_encoded = self.encoder_model(x, training=True)
      real_distribution = tf.random.normal(x_encoded.shape, mean=0., stddev=1.)

      dc_fake = self.discriminator_model(x_encoded, training=True)
      dc_real = self.discriminator_model(real_distribution, training=True)

      dc_loss = self.discriminator_loss(dc_real, dc_fake)
      dc_acc = self.accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                              tf.concat([dc_real, dc_fake], axis=0))

    dc_gradients = dc_tape.gradient(dc_loss, self.discriminator_model.trainable_variables)
    self.dc_optimizer.apply_gradients(zip(dc_gradients, self.discriminator_model.trainable_variables))

    # Generator
    with tf.GradientTape() as gen_tape:
      x_encoded = self.encoder_model(x, training=True)
      gen_fake = self.discriminator_model(x_encoded, training=True)

      gen_loss = self.generator_loss(gen_fake)

    gen_gradients = gen_tape.gradient(gen_loss, self.encoder_model.trainable_variables)
    self.gen_optimizer.apply_gradients(zip(gen_gradients, self.encoder_model.trainable_variables))

    return ae_loss, dc_loss, dc_acc, gen_loss

  def fit(self, X_train, epochs=1, batch_size=256, callbacks=None):
    global_step = 0
    step_size = 2 * np.ceil(X_train.shape[0] / batch_size)
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(X_train.shape[0]).batch(batch_size)

    callbacks = CallbackList(callbacks)
    callback_model = self._get_callback_model()
    callbacks.set_model(callback_model)

    for epoch in range(1, epochs + 1):
      callbacks.on_epoch_begin(epoch)
      epoch_logs = {}

      # Learning rate schedule
      if epoch in [60, 100, 300]:
        self.base_lr = self.base_lr / 2
        self.max_lr = self.max_lr / 2
        step_size = step_size / 2

        # print('learning rate changed!')

      epoch_ae_loss_avg = tf.metrics.Mean()
      epoch_dc_loss_avg = tf.metrics.Mean()
      epoch_dc_acc_avg = tf.metrics.Mean()
      epoch_gen_loss_avg = tf.metrics.Mean()

      for batch, (x_batch) in enumerate(train_dataset):
        # -------------------------------------------------------------------------------------------------------------
        # Calculate cyclic learning rate
        global_step = global_step + 1
        cycle = np.floor(1 + global_step / (2 * step_size))
        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        clr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x_lr)
        self.ae_optimizer.lr = clr
        self.dc_optimizer.lr = clr
        self.gen_optimizer.lr = clr

        ae_loss, dc_loss, dc_acc, gen_loss = self.train_step(x_batch)

        epoch_ae_loss_avg(ae_loss)
        epoch_dc_loss_avg(dc_loss)
        epoch_dc_acc_avg(dc_acc)
        epoch_gen_loss_avg(gen_loss)
      
      epoch_logs['ae_loss'] = epoch_ae_loss_avg.result().numpy()
      epoch_logs['dc_loss'] = epoch_dc_loss_avg.result().numpy()
      epoch_logs['dc_acc'] = epoch_dc_acc_avg.result().numpy()
      epoch_logs['gen_loss'] = epoch_gen_loss_avg.result().numpy()

      callbacks.on_epoch_end(epoch, logs=epoch_logs)

class AdversarialAutoencoder2(AdversarialAutoencoder):
  def __init__(self,
              input_dims,
              latent_dim,
              hidden_dim=1000,
              alpha=0.3,
              drop_rate=0.5,
              ae_loss_weight=1.0,
              gen_loss_weight=1.0,
              dc_loss_weight=1.0):

    super(AdversarialAutoencoder2, self).__init__(input_dims=input_dims,
                                                  latent_dim=latent_dim,
                                                  hidden_dim=hidden_dim,
                                                  alpha=alpha,
                                                  drop_rate=drop_rate,
                                                  ae_loss_weight=ae_loss_weight,
                                                  gen_loss_weight=gen_loss_weight,
                                                  dc_loss_weight=dc_loss_weight)

    self.ae_optimizer = tf.keras.optimizers.Adam(1e-4)
    self.dc_optimizer = tf.keras.optimizers.Adam(1e-4)
    self.gen_optimizer = tf.keras.optimizers.Adam(1e-4)

  def fit(self,
          train_dataset,
          test_dataset=None,
          batch_size=32,
          epochs=1,
          callbacks=None):

    epoch_ae_loss_avg = tf.metrics.Mean()
    epoch_dc_loss_avg = tf.metrics.Mean()
    epoch_dc_acc_avg = tf.metrics.Mean()
    epoch_gen_loss_avg = tf.metrics.Mean()

    train_reconstruction_loss = tf.keras.metrics.Mean()
    test_reconstruction_loss = tf.keras.metrics.Mean()

    train_dataset = self._check_tf_dataset_instance(train_dataset, batch_size=batch_size)

    callbacks = CallbackList(callbacks)
    callback_model = self._get_callback_model()
    callbacks.set_model(callback_model)

    for epoch in range(1, epochs + 1):  
      callbacks.on_epoch_begin(epoch)
      epoch_logs = {}

      for x_train in train_dataset:
        ae_loss, dc_loss, dc_acc, gen_loss = self.train_step(x_train)

        epoch_ae_loss_avg(ae_loss)
        epoch_dc_loss_avg(dc_loss)
        epoch_dc_acc_avg(dc_acc)
        epoch_gen_loss_avg(gen_loss)

        train_reconstruction_loss(self.compute_reconstruction_error(x_train))

      epoch_logs['ae_loss'] = epoch_ae_loss_avg.result().numpy()
      epoch_logs['dc_loss'] = epoch_dc_loss_avg.result().numpy()
      epoch_logs['dc_acc'] = epoch_dc_acc_avg.result().numpy()
      epoch_logs['gen_loss'] = epoch_gen_loss_avg.result().numpy()
      epoch_logs['train_reconstruction_error'] = train_reconstruction_loss.result().numpy()
      epoch_ae_loss_avg.reset_states()
      epoch_dc_loss_avg.reset_states()
      epoch_dc_acc_avg.reset_states()
      epoch_gen_loss_avg.reset_states()
      train_reconstruction_loss.reset_states()

      if test_dataset is not None:
        test_dataset = self._check_tf_dataset_instance(test_dataset, batch_size=batch_size)
        for x_test in test_dataset:
          test_reconstruction_loss(self.compute_reconstruction_error(x_test))
        
        epoch_logs['test_reconstruction_error'] = test_reconstruction_loss.result().numpy()
        test_reconstruction_loss.reset_states()

      callbacks.on_epoch_end(epoch, logs=epoch_logs)

  def reconstruct(self, x):
    x_encoded = self.encoder_model(x, training=False)
    x_decoded = self.decoder_model(x_encoded, training=False)
    return x_decoded