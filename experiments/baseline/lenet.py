import tensorflow as tf
import time

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')
train_images /= 255.
test_images /= 255.

EPOCHS = 10
BATCH_SIZE = 256
TRAINING_EXAMPLES = [60000, 30000, 10000, 5000]

for n_examples in TRAINING_EXAMPLES:

  train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images[:n_examples], train_labels[:n_examples])
  ).shuffle(n_examples).batch(BATCH_SIZE)
  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)

  model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=train_images.shape[1:]),
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(filters=120, kernel_size=5, strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam()

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

  for epoch in range(EPOCHS):
    start_time = time.time()
    for (x_batch, y_batch) in train_dataset:
      with tf.GradientTape() as tape:
        y_pred = model(x_batch)
        loss = loss_object(y_batch, y_pred)
      gradient = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradient, model.trainable_variables))

      train_loss(loss)
      train_accuracy(y_batch, y_pred)

    for (x_batch, y_batch) in test_dataset:
      y_pred = model(x_batch)
      loss = loss_object(y_batch, y_pred)
      test_loss(loss)
      test_accuracy(y_batch, y_pred)
    
    end_time = time.time()

    print('Epoch {}: Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Time: {}'.format(
      epoch + 1,
      train_loss.result(),
      train_accuracy.result(),
      test_loss.result(),
      test_accuracy.result(),
      end_time - start_time
    ))

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()