import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.naive_bayes import GaussianNB
from argparse import ArgumentParser
from autoencoders import DenoisingAutoencoder
from autoencoders.ae_callbacks import SaveLossesCallback, LogCallback, ReconstructionErrorCallback

parser = ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist', 'cifar10'])
parser.add_argument('-c', '--corruption_rate', type=float, default=0.2)

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
  epochs = 30
  min_latent_dim = 2
  max_latent_dim = 32
  corruption_proportion = args.corruption_rate

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
  base_logdir = 'mnist_denoising_ae_prop_{}'.format(corruption_proportion)

elif args.dataset == 'fashion_mnist':
  # FASHION MNIST Dataset
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
  input_dims = [28, 28, 1]
  train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
  test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')

  train_images /= 255.
  test_images /= 255.

  batch_size = 200
  epochs = 30
  min_latent_dim = 2
  max_latent_dim = 50
  corruption_proportion = args.corruption_rate

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
  base_logdir = 'fashion_mnist_denoising_ae_prop_{}'.format(corruption_proportion)

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
  epochs = 30
  min_latent_dim = 2
  max_latent_dim = 100
  corruption_proportion = args.corruption_rate

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(50000).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
  base_logdir = 'cifar10_denoising_ae_prop_{}'.format(corruption_proportion)

acc_logdir = base_logdir + '/logs/accuracy'
train_mse_logdir = base_logdir + '/logs/mse/train'
test_mse_logdir = base_logdir + '/logs/mse/test'
acc_writer = tf.summary.create_file_writer(acc_logdir)
train_mse_writer = tf.summary.create_file_writer(train_mse_logdir)
test_mse_writer = tf.summary.create_file_writer(test_mse_logdir)
train_dataset_unshuffle = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size)

train_mse = tf.keras.metrics.Mean()
test_mse = tf.keras.metrics.Mean()

for latent_dim in range(min_latent_dim, max_latent_dim+1):
  start_time = time.time()

  model = DenoisingAutoencoder(input_dims=input_dims, latent_dim=latent_dim, corruption_proportion=corruption_proportion)

  model.fit(train_dataset,
            test_dataset,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=tf.keras.optimizers.Adam(1e-4),
            callbacks=[SaveLossesCallback(logdir=base_logdir+'/logs/denoising_ae_' + str(latent_dim) + '/'),
                      ReconstructionErrorCallback(logdir=base_logdir+'/logs/denoising_ae_' + str(latent_dim) + '/')]
            )

  model.save_weights(base_logdir + '/weights/denoising_ae_{}.ckpt'.format(str(latent_dim)))
  
  X_train = np.zeros((1, latent_dim))
  X_test = np.zeros((1, latent_dim))

  for x_batch in train_dataset_unshuffle:
    z = model.encode(x_batch)
    X_train = np.append(X_train, z.numpy(), axis=0)
    train_mse(model.compute_reconstruction_error(x_batch))

  for x_batch in test_dataset:
    z = model.encode(x_batch)
    X_test = np.append(X_test, z.numpy(), axis=0)
    test_mse(model.compute_reconstruction_error(x_batch))
  
  X_train = X_train[1:]
  X_test = X_test[1:]

  classifier = GaussianNB()
  classifier.fit(X_train, train_labels)
  acc = classifier.score(X_test, test_labels)
  
  with acc_writer.as_default():
    tf.summary.scalar('Accuracy', acc, step=latent_dim)

  with train_mse_writer.as_default():
    tf.summary.scalar('Train MSE', train_mse.result().numpy(), step=latent_dim)

  with test_mse_writer.as_default():
    tf.summary.scalar('Test MSE', test_mse.result().numpy(), step=latent_dim)
  
  train_mse.reset_states()
  test_mse.reset_states()
  end_time = time.time()
  print('Latent dim: {}, Time: {}, Accuracy: {}'.format(latent_dim, end_time - start_time, acc))