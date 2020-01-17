from autoencoders import DenoisingAutoencoder
from autoencoders.ae_callbacks import SaveLossesCallback, LogCallback, ReconstructionErrorCallback
import tensorflow as tf
import matplotlib.pyplot as plt

model = DenoisingAutoencoder(input_dims=[28, 28, 1], latent_dim=9, corruption_proportion=0.1)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')

train_images /= 255.
test_images /= 255.

batch_size = 100
epochs = 10

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)

model.fit(train_dataset,
          test_dataset,
          batch_size=batch_size,
          epochs=epochs,
          optimizer=tf.keras.optimizers.Adam(1e-4),
          callbacks=[
            # SaveLossesCallback(logdir='logs/vae/test/'),
            LogCallback(),
            # ReconstructionErrorCallback(logdir='logs/vae/test/')
          ])

from sklearn.naive_bayes import GaussianNB
X_train = model.encode(train_images)
X_test = model.encode(test_images)
classifier = GaussianNB()
classifier.fit(X_train, train_labels)
print('Classification accuracy: {}'.format(classifier.score(X_test, test_labels)))