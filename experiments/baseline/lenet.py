import tensorflow as tf
import time
import datetime
import csv
import os

def create_lenet_for_mnist():
  return tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=train_images.shape[1:]),
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Conv2D(filters=120, kernel_size=5, strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

def create_lenet_for_cifar():
  return tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=train_images.shape[1:]),
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Conv2D(filters=120, kernel_size=5, strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

path = os.path.dirname(__file__) + 'lenet_logs'
if not os.path.exists(path):
  os.mkdir(path)

for d in ['mnist', 'fashion_mnist', 'cifar']:
  if d == 'mnist':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
    test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')
    EPOCHS = 15
    BATCH_SIZE = 256
    TRAINING_EXAMPLES = [5000, 10000, 30000, 60000]
  elif d == 'fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
    test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')
    EPOCHS = 15
    BATCH_SIZE = 256
    TRAINING_EXAMPLES = [5000, 10000, 30000, 60000]
  elif d == 'cifar':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.reshape([-1, 32, 32, 3]).astype('float32')
    test_images = test_images.reshape([-1, 32, 32, 3]).astype('float32')
    EPOCHS = 30
    BATCH_SIZE = 256
    TRAINING_EXAMPLES = [5000, 10000, 30000, 50000]

  train_images /= 255.
  test_images /= 255.

  for n_examples in TRAINING_EXAMPLES:
    print('Dataset: {}, Number of exampls: {}'.format(d, n_examples))
    train_dataset = tf.data.Dataset.from_tensor_slices(
      (train_images[:n_examples], train_labels[:n_examples])
    ).shuffle(n_examples).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)

    if d == 'mnist' or d == 'fashion_mnist':
      model = create_lenet_for_mnist()
    elif d == 'cifar':
      model = create_lenet_for_cifar()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    prediction_time = tf.keras.metrics.Mean(name='prediction_time')

    time_elapsed = 0
    for epoch in range(EPOCHS):
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()
      
      start_time = time.time()
      for (x_batch, y_batch) in train_dataset:
        with tf.GradientTape() as tape:
          y_pred = model(x_batch)
          loss = loss_object(y_batch, y_pred)
        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        train_loss(loss)
        train_accuracy(y_batch, y_pred)

      prediction_start_time = time.time()
      for (x_batch, y_batch) in test_dataset:
        y_pred = model(x_batch)
        loss = loss_object(y_batch, y_pred)
        test_loss(loss)
        test_accuracy(y_batch, y_pred)
      time_elapsed += (prediction_start_time - start_time)
      prediction_epoch_time = time.time() - prediction_start_time
      prediction_time(prediction_epoch_time)

      print('Epoch {}: Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Time: {}'.format(
        epoch + 1,
        train_loss.result(),
        train_accuracy.result(),
        test_loss.result(),
        test_accuracy.result(),
        time_elapsed + prediction_epoch_time
      ))

    with open('lenet_logs/{}_lenet_baseline.csv'.format(d), 'a') as csvfile:
      fieldnames = ['time', 'n_examples', 'accuracy', 'time_elapsed', 'prediction_time']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      if csvfile.tell() == 0:
        writer.writeheader()
      writer.writerow({
        'time': datetime.datetime.now(),
        'n_examples': n_examples,
        'accuracy': test_accuracy.result().numpy(),
        'time_elapsed': time_elapsed,
        'prediction_time': prediction_time.result().numpy()})
