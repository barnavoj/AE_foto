from enum import auto
import matplotlib as matplt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Reshape, InputLayer, Conv2DTranspose, UpSampling2D


matplt.use('Agg')

img_size = 256
batch_size = 64
train_dir = 'database\\places_256\\a'

input_shape = (img_size, img_size, 3)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
train_ds = image_generator.flow_from_directory( 
    train_dir, 
    class_mode='input', target_size=(img_size,img_size), batch_size=batch_size,
)

# # NN params
# num_blocks = 4  # number of feature extraction blocks [conv, conv, pool]
# num_conv = [2, 2, 3, 3]        # number of convolutin layers per block
# filters = [[32, 32], [64, 64], [128, 128, 128], [256,256,256]]
# kernel_s = [[3, 3], [3, 3], [3, 3, 3], [3, 3, 3]]
# num_dense = 2       # number of fully conected layers
# dense = [256, 32]  # number of neurons per fully conected layer
# dense_decoder = int(img_size/16)*int(img_size/16)*256


# class Autoencoder(tf.keras.models.Model):
#   def __init__(self):
#     super(Autoencoder, self).__init__()
#     self.encoder = tf.keras.Sequential()
#     self.encoder.add(InputLayer(input_shape=(img_size, img_size, 3)))
#     for i in range(num_blocks):
#         for j in range(num_conv[i]):
#             self.encoder.add(
#                 Conv2D(
#                     filters=filters[i][j],
#                     kernel_size=(kernel_s[i][j], kernel_s[i][j]),
#                     padding="same", activation="relu"))
#         self.encoder.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
#     self.encoder.add(Flatten())
#     for i in range(num_dense):
#         self.encoder.add(Dense(units=dense[i], activation="relu"))

#     filters.reverse()
#     kernel_s.reverse()
#     num_conv.reverse()

#     self.decoder = tf.keras.Sequential()
#     self.decoder.add(InputLayer(input_shape=(dense[1])))
#     self.decoder.add(Dense(units=dense[0], activation="relu"))
#     self.decoder.add(Dense(units=dense_decoder, activation="relu"))
#     self.decoder.add(Reshape((int(img_size/16), int(img_size/16), 256)))
#     for i in range(num_blocks):
#         for j in range(num_conv[i]):
#             self.decoder.add(
#                 Conv2DTranspose(
#                     filters=filters[i][j],
#                     kernel_size=(kernel_s[i][j], kernel_s[i][j]),
#                     padding="same", activation="relu"))
#         self.decoder.add(UpSampling2D(size=(2, 2)))
#     self.decoder.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))

#   def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded



class Autoencoder(tf.keras.models.Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential()
    self.encoder.add(InputLayer(input_shape=(img_size, img_size, 3)))
    self.encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    self.encoder.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    self.encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    self.encoder.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    self.encoder.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    self.encoder.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    self.encoder.add(Flatten())
    self.encoder.add(Dense(units=32, activation="relu"))


    self.decoder = tf.keras.Sequential()
    self.decoder.add(InputLayer(input_shape=(32)))
    self.decoder.add(Dense(units=4096, activation="relu"))
    self.decoder.add(Reshape((32, 32, 4)))
    self.decoder.add(UpSampling2D(size=(2, 2)))
    self.decoder.add(Conv2DTranspose(4, (3, 3), activation='relu', padding='same'))
    self.decoder.add(UpSampling2D(size=(2, 2)))
    self.decoder.add(Conv2DTranspose(8, (3, 3), activation='relu', padding='same'))
    self.decoder.add(UpSampling2D(size=(2, 2)))
    self.decoder.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    self.decoder.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        plt.figure(figsize=(15, 5))
        clear_output(wait=True)
            
        plt.plot(range(1, epoch + 2), self.metrics["loss"])
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Progress of loss function during training")
        plt.tight_layout()
        plt.savefig("training_progress.png")

        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()


autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.encoder.summary()
autoencoder.decoder.summary()

checkpoint_path = "ae.ckpt"
#model.load_weights(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    period=10,
    verbose=1,
    monitor='loss',
    mode='min',
    save_best_only=True)

history = autoencoder.fit(train_ds,
                          epochs=1000,
                          batch_size=batch_size, 
                          shuffle=True,
                          callbacks=[cp_callback, PlotLearning()],
                          verbose=1
                          )
                          
            