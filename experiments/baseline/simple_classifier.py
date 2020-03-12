import time
import datetime
import csv
import os
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

path = os.path.dirname(__file__) + 'simple_classifier_logs'
if not os.path.exists(path):
  os.mkdir(path)

classifiers = [
  GaussianNB(),
  SVC(gamma='scale'),
  RandomForestClassifier(n_estimators=100)]
classifier_name = [
  'gaussian_nb',
  'svm',
  'random_forest']

for d in ['mnist', 'fashion_mnist', 'cifar']:
  if d == 'mnist':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape([-1, 28 * 28]).astype('float32')
    test_images = test_images.reshape([-1, 28 * 28]).astype('float32')
    TRAINING_EXAMPLES = [5000, 10000, 30000, 60000]
  elif d == 'fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape([-1, 28 * 28]).astype('float32')
    test_images = test_images.reshape([-1, 28 * 28]).astype('float32')
    TRAINING_EXAMPLES = [5000, 10000, 30000, 60000]
  elif d == 'cifar':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.reshape([-1, 32 * 32 * 3]).astype('float32')
    test_images = test_images.reshape([-1, 32 * 32 * 3]).astype('float32')
    TRAINING_EXAMPLES = [5000, 10000, 30000, 50000]

  train_images /= 255.
  test_images /= 255.

  for n_examples in TRAINING_EXAMPLES:
    print('Dataset: {}, Number of exampls: {}'.format(d, n_examples))
    X_train = train_images[:n_examples]
    y_train = train_labels[:n_examples].ravel()

    for name, classifier in zip(classifier_name, classifiers):
      start_time = time.time()
      classifier.fit(X_train, y_train)
      prediction_start_time = time.time()
      accuracy = classifier.score(test_images, test_labels.ravel())
      time_elapsed = time.time() - start_time
      prediction_time = time.time() - prediction_start_time
      print('Classifier: {}, Accuracy: {}, Training time: {}, Prediction time: {}'.format(name, accuracy, time_elapsed, prediction_time))
      with open(path + '/{}_{}_baseline.csv'.format(d, name), 'a') as csvfile:
        fieldnames = ['time', 'n_examples', 'accuracy', 'time_elapsed', 'prediction_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
          writer.writeheader()
        writer.writerow({
          'time': datetime.datetime.now(),
          'n_examples': n_examples,
          'accuracy': accuracy,
          'time_elapsed': time_elapsed,
          'prediction_time': prediction_time})