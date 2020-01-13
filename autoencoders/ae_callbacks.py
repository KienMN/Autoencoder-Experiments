import tensorflow as tf
import time
import os

class SaveLossesCallback(tf.keras.callbacks.Callback):
  def __init__(self, logdir=None):
    super(SaveLossesCallback, self).__init__()
    if logdir is not None:
      self.train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
      self.test_writer = tf.summary.create_file_writer(os.path.join(logdir, 'test'))

  def on_epoch_end(self, epoch, logs=None):
    
    with self.train_writer.as_default():
      tf.summary.scalar('Loss', data=logs.get('train_loss', 0), step=epoch)
    
    if logs.get('test_loss') is not None:
      with self.test_writer.as_default():
        tf.summary.scalar('Loss', data=logs.get('test_loss', 0), step=epoch)

class LogCallback(tf.keras.callbacks.Callback):
  def on_epoch_begin(self, epoch, logs=None):
    self.start_time = time.time()

  def on_epoch_end(self, epoch, logs=None):
    self.end_time = time.time()
    print('Epoch: {} Time: {}'.format(epoch, self.end_time - self.start_time), end='')
    for (name, value) in logs.items():
      print(' ' + name + ': ' + str(value), end='')
    print()

class ReconstructionErrorCallback(tf.keras.callbacks.Callback):
  def __init__(self, logdir=None):
    super(ReconstructionErrorCallback, self).__init__()
    if logdir is not None:
      self.train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
      self.test_writer = tf.summary.create_file_writer(os.path.join(logdir, 'test'))

  def on_epoch_end(self, epoch, logs=None):
    
    with self.train_writer.as_default():
      tf.summary.scalar('Reconstruction error', data=logs.get('train_reconstruction_error', 0), step=epoch)

    if logs.get('test_reconstruction_error') is not None:
      with self.test_writer.as_default():
        tf.summary.scalar('Reconstruction error', data=logs.get('test_reconstruction_error', 0), step=epoch)
    
class AAESaveLossesCallback(tf.keras.callbacks.Callback):
  def __init__(self, logdir=None):
    super(AAESaveLossesCallback, self).__init__()
    if logdir is not None:
      self.train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
      self.test_writer = tf.summary.create_file_writer(os.path.join(logdir, 'test'))

  def on_epoch_end(self, epoch, logs=None):
    
    with self.train_writer.as_default():
      tf.summary.scalar('AE Loss', data=logs.get('ae_loss', 0), step=epoch)
      tf.summary.scalar('DC Loss', data=logs.get('dc_loss', 0), step=epoch)
      tf.summary.scalar('GEN Loss', data=logs.get('gen_loss', 0), step=epoch)
      tf.summary.scalar('DC Acc', data=logs.get('dc_acc', 0), step=epoch)