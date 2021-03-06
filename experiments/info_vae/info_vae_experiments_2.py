from autoencoders import MnistInfoVariationalAutoencoder, Cifar10InfoVariationalAutoencoder
from autoencoders.ae_callbacks import SaveLossesCallback, LogCallback, ReconstructionErrorCallback
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist', 'cifar10'])
args = parser.parse_args()

if args.dataset == 'mnist':
  # MNIST Dataset
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
  input_dims = [28, 28, 1]
  train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
  test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')

  train_images /= 255.
  test_images /= 255.

  batch_size = 200
  train_buffer_size = 60000
  epochs = 0
  min_latent_dim = 2
  max_latent_dim = 3

  base_logdir = 'mnist_info_vae'

elif args.dataset == 'fashion_mnist':
  # FASHION MNIST Dataset
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
  input_dims = [28, 28, 1]
  train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
  test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')

  train_images /= 255.
  test_images /= 255.

  batch_size = 200
  train_buffer_size = 60000
  epochs = 3
  min_latent_dim = 2
  max_latent_dim = 32

  base_logdir = 'fashion_mnist_info_vae'

elif args.dataset == 'cifar10':
  # CIFAR10 Dataset
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  input_dims = [32, 32, 3]
  train_images = train_images.astype('float32')
  train_labels = train_labels.ravel()
  test_images = test_images.astype('float32')
  test_labels = test_labels.ravel()

  train_images /= 255.
  test_images /= 255.

  batch_size = 200
  train_buffer_size = 50000
  epochs = 3
  min_latent_dim = 2
  max_latent_dim = 32

  base_logdir = 'cifar10__info_vae'

acc_logdir = base_logdir + '/logs/accuracy'
train_mse_logdir = base_logdir + '/logs/mse/train'
test_mse_logdir = base_logdir + '/logs/mse/test'
acc_writer_1 = tf.summary.create_file_writer(acc_logdir + '/gaussian_nb')
acc_writer_2 = tf.summary.create_file_writer(acc_logdir + '/svm')
acc_writer_3 = tf.summary.create_file_writer(acc_logdir + '/random_forest')
train_mse_writer = tf.summary.create_file_writer(train_mse_logdir)
test_mse_writer = tf.summary.create_file_writer(test_mse_logdir)


train_mse = tf.keras.metrics.Mean()
test_mse = tf.keras.metrics.Mean()

for latent_dim in range(min_latent_dim, max_latent_dim+1):
  start_time = time.time()
  if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
    model = MnistInfoVariationalAutoencoder(input_dims=input_dims, latent_dim=latent_dim)
  elif args.dataset == 'cifar10':
    model = Cifar10InfoVariationalAutoencoder(input_dims=input_dims, latent_dim=latent_dim)

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_buffer_size).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)

  model.fit(train_dataset,
            test_dataset,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=tf.keras.optimizers.Adam(1e-4),
            callbacks=[SaveLossesCallback(logdir=base_logdir+'/logs/info_vae_{}/'.format(str(latent_dim))),
                      ReconstructionErrorCallback(logdir=base_logdir+'/logs/info_vae_{}/'.format(str(latent_dim)))]
            )
  
  model.save_weights(base_logdir+'/weights/info_vae_{}.ckpt'.format(str(latent_dim)))

  # Classification
  train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
  
  X_train = np.zeros((1, latent_dim))
  X_test = np.zeros((1, latent_dim))
  y_train = np.zeros((1,))
  y_test = np.zeros((1,))

  for (x_batch, y_batch) in train_dataset:
    mean, logvar = model.encode(x_batch)
    z = model.reparameterize(mean, logvar)
    X_train = np.append(X_train, z.numpy(), axis=0)
    y_train = np.append(y_train, y_batch.numpy())
    train_mse(model.compute_reconstruction_error(x_batch))

  for (x_batch, y_batch) in test_dataset:
    mean, logvar = model.encode(x_batch)
    z = model.reparameterize(mean, logvar)
    X_test = np.append(X_test, z.numpy(), axis=0)
    y_test = np.append(y_test, y_batch.numpy())
    test_mse(model.compute_reconstruction_error(x_batch))
  
  X_train = X_train[1:]
  X_test = X_test[1:]
  y_train = y_train[1:]
  y_test = y_test[1:]
  
  ae_end_time = time.time()

  classifier_1 = GaussianNB()
  classifier_1.fit(X_train, y_train)
  acc1 = classifier_1.score(X_test, y_test)

  classifier_1_time = time.time()

  classifier_2 = SVC(gamma='scale')
  classifier_2.fit(X_train, y_train)
  acc2 = classifier_2.score(X_test, y_test)

  classifier_2_time = time.time()

  classifier_3 = RandomForestClassifier(n_estimators=100)
  classifier_3.fit(X_train, y_train)
  acc3 = classifier_3.score(X_test, y_test)
  
  classifier_3_time = time.time()

  with acc_writer_1.as_default():
    tf.summary.scalar('Accuracy', acc1, step=latent_dim)
  
  with acc_writer_2.as_default():  
    tf.summary.scalar('Accuracy', acc2, step=latent_dim)

  with acc_writer_3.as_default():  
    tf.summary.scalar('Accuracy', acc3, step=latent_dim)

  with train_mse_writer.as_default():
    tf.summary.scalar('Train MSE', train_mse.result().numpy(), step=latent_dim)

  with test_mse_writer.as_default():
    tf.summary.scalar('Test MSE', test_mse.result().numpy(), step=latent_dim)
  
  train_mse.reset_states()
  test_mse.reset_states()

  end_time = time.time()
  print('Latent dim: {}, AE Time: {}, NB time: {}, SVM time: {}, Forest time: {}, Accuracy: {}'.format(
    latent_dim,
    ae_end_time - start_time,
    classifier_1_time - ae_end_time,
    classifier_2_time - classifier_1_time,
    classifier_3_time - classifier_2_time,
    [acc1, acc2, acc3]))