from autoencoders import MnistInfoVariationalAutoencoder
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import numpy as np
from autoencoders.ae_callbacks import LogCallback

model = MnistInfoVariationalAutoencoder(input_dims=[28, 28, 1], latent_dim=2, hidden_dim=1000)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')

train_images /= 255.
test_images /= 255.

batch_size = 1000
epochs = 3

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)

model.fit(train_dataset, test_dataset, epochs=epochs, batch_size=batch_size, optimizer=tf.keras.optimizers.Adam(1e-4),
          callbacks=[LogCallback()])

# print(model.inference_net.summary())
# print(model.generative_net.summary())

# mean, logvar = model.encode(test_images[:100])
# z = model.reparameterize(mean, logvar)

# x_recon = model.decode(z, apply_sigmoid=True)

# fig = plt.figure(figsize=(10, 10))

# for i in range(x_recon.shape[0]):
#     plt.subplot(10, 10, i+1)
#     plt.imshow(x_recon[i, :, :, 0], cmap='gray')
#     plt.axis('off')

# # tight_layout minimizes the overlap between 2 sub-plots
# # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
# plt.show()