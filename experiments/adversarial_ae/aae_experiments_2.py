from autoencoders import AdversarialAutoencoder2
from autoencoders.ae_callbacks import AAESaveLossesCallback, LogCallback, ReconstructionErrorCallback
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
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
  epochs = 30
  min_latent_dim = 2
  max_latent_dim = 32

  base_logdir = 'mnist_aae'

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
  epochs = 30
  min_latent_dim = 2
  max_latent_dim = 50

  base_logdir = 'fashion_mnist_aae'

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
  epochs = 30
  min_latent_dim = 2
  max_latent_dim = 100

  base_logdir = 'cifar10_aae'

acc_logdir = base_logdir + '/logs/accuracy'
train_mse_logdir = base_logdir + '/logs/mse/train'
test_mse_logdir = base_logdir + '/logs/mse/test'
acc_writer = tf.summary.create_file_writer(acc_logdir)
train_mse_writer = tf.summary.create_file_writer(train_mse_logdir)
test_mse_writer = tf.summary.create_file_writer(test_mse_logdir)


train_mse = tf.keras.metrics.Mean()
test_mse = tf.keras.metrics.Mean()

for latent_dim in range(min_latent_dim, max_latent_dim+1):
  start_time = time.time()
  model = AdversarialAutoencoder2(input_dims=input_dims, latent_dim=latent_dim)

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_buffer_size).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)

  model.fit(train_dataset,
            test_dataset,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[AAESaveLossesCallback(logdir=base_logdir+'/logs/aae_{}/'.format(str(latent_dim))),
                      ReconstructionErrorCallback(logdir=base_logdir+'/logs/aae_{}/'.format(str(latent_dim)))]
            )
  
  model.save_weights(base_logdir+'/weights/aae_{}.ckpt'.format(str(latent_dim)))

  # Classification
  train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
  
  X_train = np.zeros((1, latent_dim))
  X_test = np.zeros((1, latent_dim))
  y_train = np.zeros((1,))
  y_test = np.zeros((1,))

  for (x_batch, y_batch) in train_dataset:
    z = model.encode(x_batch)
    X_train = np.append(X_train, z.numpy(), axis=0)
    y_train = np.append(y_train, y_batch.numpy())
    train_mse(model.compute_reconstruction_error(x_batch))

  for (x_batch, y_batch) in test_dataset:
    z = model.encode(x_batch)
    X_test = np.append(X_test, z.numpy(), axis=0)
    y_test = np.append(y_test, y_batch.numpy())
    test_mse(model.compute_reconstruction_error(x_batch))
  
  X_train = X_train[1:]
  X_test = X_test[1:]
  y_train = y_train[1:]
  y_test = y_test[1:]

  classifier = GaussianNB()
  classifier.fit(X_train, y_train)
  acc = classifier.score(X_test, y_test)
  
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