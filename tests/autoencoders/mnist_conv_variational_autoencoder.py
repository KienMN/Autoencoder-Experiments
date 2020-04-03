from autoencoders import MnistConvolutionalVariationalAutoencoder
from autoencoders.ae_callbacks import SaveLossesCallback, LogCallback, ReconstructionErrorCallback
import tensorflow as tf
import matplotlib.pyplot as plt

model = MnistConvolutionalVariationalAutoencoder(input_dims=[28, 28, 1], latent_dim=3)

print(model.inference_net.summary())
print(model.generative_net.summary())

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')

train_images /= 255.
test_images /= 255.

batch_size = 200
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

# model.save_weights('my_model_weights.ckpt')
# new_model = VariationalAutoencoder(input_dims=[28, 28, 1], latent_dim=3)
# new_model.load_weights('my_model_weights.ckpt')

# mean, logvar = model.encode(test_images[:100])
# z = model.reparameterize(mean, logvar)

# x_recon = model.decode(z, apply_sigmoid=True)

# fig = plt.figure(figsize=(10, 10))

# for i in range(x_recon.shape[0]):
#     plt.subplot(10, 10, i+1)
#     plt.imshow(x_recon[i, :, :, 0], cmap='gray')
#     plt.axis('off')

# plt.show()