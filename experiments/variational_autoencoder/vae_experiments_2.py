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
  max_latent_dim = 10

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
  max_latent_dim = 15

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
  max_latent_dim = 25

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(50000).batch(batch_size)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
  base_logdir = 'cifar10_vae'

acc_logdir = base_logdir + '/logs/accuracy'
train_mse_logdir = base_logdir + '/logs/mse/train'
test_mse_logdir = base_logdir + '/logs/mse/test'
acc_writer_1 = tf.summary.create_file_writer(acc_logdir + '/gaussian_nb')
acc_writer_2 = tf.summary.create_file_writer(acc_logdir + '/svm')
acc_writer_3 = tf.summary.create_file_writer(acc_logdir + '/random_forest')
train_mse_writer = tf.summary.create_file_writer(train_mse_logdir)
test_mse_writer = tf.summary.create_file_writer(test_mse_logdir)

for latent_dim in range(min_latent_dim, max_latent_dim+1):
  start_time = time.time()
  model = VariationalAutoencoder(input_dims=input_dims, latent_dim=latent_dim)

  model.fit(train_dataset,
            test_dataset,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=tf.keras.optimizers.Adam(1e-4),
            callbacks=[SaveLossesCallback(logdir=base_logdir+'/logs/vae_' + str(latent_dim) + '/'),
                      ReconstructionErrorCallback(logdir=base_logdir+'/logs/vae_' + str(latent_dim) + '/')]
            )

  model.save_weights(base_logdir + '/weights/vae_{}.ckpt'.format(str(latent_dim)))

  mean, logvar = model.encode(train_images)
  X_train = model.reparameterize(mean, logvar)
  train_mse = model.compute_reconstruction_error(train_images)

  mean, logvar = model.encode(test_images)
  X_test = model.reparameterize(mean, logvar)
  test_mse = model.compute_reconstruction_error(test_images)
  
  ae_end_time = time.time()

  classifier_1 = GaussianNB()
  classifier_1.fit(X_train, train_labels)
  acc1 = classifier_1.score(X_test, test_labels)

  classifier_1_time = time.time()

  classifier_2 = SVC(gamma='scale')
  classifier_2.fit(X_train, train_labels)
  acc2 = classifier_2.score(X_test, test_labels)

  classifier_2_time = time.time()

  classifier_3 = RandomForestClassifier(n_estimators=100)
  classifier_3.fit(X_train.numpy(), train_labels)
  acc3 = classifier_3.score(X_test.numpy(), test_labels)
  
  classifier_3_time = time.time()

  with acc_writer_1.as_default():
    tf.summary.scalar('Accuracy', acc1, step=latent_dim)
  
  with acc_writer_2.as_default():  
    tf.summary.scalar('Accuracy', acc2, step=latent_dim)

  with acc_writer_3.as_default():  
    tf.summary.scalar('Accuracy', acc3, step=latent_dim)

  with train_mse_writer.as_default():
    tf.summary.scalar('Train MSE', train_mse, step=latent_dim)

  with test_mse_writer.as_default():
    tf.summary.scalar('Test MSE', test_mse, step=latent_dim)
  
  end_time = time.time()
  print('Latent dim: {}, AE Time: {}, NB time: {}, SVM time: {}, Forest time: {}, Accuracy: {}'.format(
    latent_dim,
    ae_end_time - start_time,
    classifier_1_time - ae_end_time,
    classifier_2_time - classifier_1_time,
    classifier_3_time - classifier_2_time,
    [acc1, acc2, acc3]))