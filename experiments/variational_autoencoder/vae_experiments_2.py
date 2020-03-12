import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
from autoencoders import VariationalAutoencoder
from autoencoders.ae_callbacks import SaveLossesCallback, LogCallback, ReconstructionErrorCallback

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

  batch_size = 100
  epochs = 30
  min_latent_dim = 5
  max_latent_dim = 15

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
  base_logdir = 'mnist_vae'

elif args.dataset == 'fashion_mnist':
  # FASHION MNIST Dataset
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
  input_dims = [28, 28, 1]
  train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
  test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')

  train_images /= 255.
  test_images /= 255.

  batch_size = 100
  epochs = 30
  min_latent_dim = 5
  max_latent_dim = 20

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
  base_logdir = 'fashion_mnist_vae'

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

  batch_size = 100
  epochs = 30
  min_latent_dim = 15
  max_latent_dim = 30

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(50000).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
  base_logdir = 'cifar10_vae'

acc_logdir = base_logdir + '/logs/accuracy'
train_mse_logdir = base_logdir + '/logs/mse/train'
test_mse_logdir = base_logdir + '/logs/mse/test'
train_mse_writer = tf.summary.create_file_writer(train_mse_logdir)
test_mse_writer = tf.summary.create_file_writer(test_mse_logdir)

classifiers = [
  GaussianNB(),
  SVC(gamma='scale'),
  RandomForestClassifier(n_estimators=100)]

classifier_names = [
  'gaussian_nb',
  'svm',
  'random_forest']

acc_writer_files = [tf.summary.create_file_writer(acc_logdir + '/{}'.format(classifier_name)) for classifier_name in classifier_names]

for latent_dim in range(min_latent_dim, max_latent_dim+1):
  model = VariationalAutoencoder(input_dims=input_dims, latent_dim=latent_dim)
  
  start_time = time.time()
  model.fit(train_dataset,
            test_dataset,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=tf.keras.optimizers.Adam(1e-4),
            callbacks=[SaveLossesCallback(logdir=base_logdir+'/logs/vae_' + str(latent_dim) + '/'),
                      ReconstructionErrorCallback(logdir=base_logdir+'/logs/vae_' + str(latent_dim) + '/')]
            )
  end_time = time.time()

  model.save_weights(base_logdir + '/weights/vae_{}.ckpt'.format(str(latent_dim)))

  mean, logvar = model.encode(train_images)
  X_train = model.reparameterize(mean, logvar).numpy()
  train_mse = model.compute_reconstruction_error(train_images)

  mean, logvar = model.encode(test_images)
  X_test = model.reparameterize(mean, logvar).numpy()
  test_mse = model.compute_reconstruction_error(test_images)
  
  for classifier, acc_writer_file in zip(classifiers, acc_writer_files):
    classifier_start_time = time.time()
    classifier.fit(X_train, train_labels)
    acc = classifier.score(X_test, test_labels)
    classifier_end_time = time.time()
    with acc_writer_file.as_default():
      tf.summary.scalar('Accuracy', acc, step=latent_dim)
      tf.summary.scalar('Time elapsed', classifier_end_time - classifier_start_time, step=latent_dim)

  with train_mse_writer.as_default():
    tf.summary.scalar('Train MSE', train_mse, step=latent_dim)

  with test_mse_writer.as_default():
    tf.summary.scalar('Test MSE', test_mse, step=latent_dim)

  print('Latent dim: {}, Training Time: {}'.format(latent_dim, end_time - start_time))