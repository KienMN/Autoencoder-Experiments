from autoencoders import AdversarialAutoencoder2
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import numpy as np
from autoencoders.ae_callbacks import LogCallback

model = AdversarialAutoencoder2(input_dims=[28, 28, 1], hidden_dim=1000, latent_dim=2)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')

train_images /= 255.
test_images /= 255.

batch_size = 100
epochs = 5

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)

model.fit(train_dataset, test_dataset, epochs=epochs, batch_size=batch_size, callbacks=[LogCallback()])

# Latent space of test set
# x_test_encoded = model.encoder_model(test_images, training=False)
# label_list = list(test_labels)

# fig = plt.figure(figsize=(10, 10))
# classes = set(label_list)
# colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
# kwargs = {'alpha': 0.8, 'c': [colormap[i] for i in label_list]}
# ax = plt.subplot(111, aspect='equal')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# handles = [mpatches.Circle((0, 0), label=class_, color=colormap[i])
#             for i, class_ in enumerate(classes)]
# ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45), fancybox=True, loc='center left')
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], s=2, **kwargs)
# ax.set_xlim([-3, 3])
# ax.set_ylim([-3, 3])

# plt.show()