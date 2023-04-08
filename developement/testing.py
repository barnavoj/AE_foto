from enum import auto
from pickletools import uint8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Reshape, InputLayer, Conv2DTranspose, UpSampling2D

import cv2 as cv2
import os
from glob import glob

img_size = 256
batch_size = 32
train_dir = 'database\\places_256\\a'

input_shape = (img_size, img_size, 3)

# image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
# train_ds = image_generator.flow_from_directory(
#     train_dir, 
#     class_mode='input', target_size=(img_size,img_size), batch_size=batch_size,
# )

# # NN params
# num_blocks = 4  # number of feature extraction blocks [conv, conv, pool]
# num_conv = [2, 2, 3, 3]        # number of convolutin layers per block
# filters = [[32, 32], [64, 64], [128, 128, 128], [256,256,256]]
# kernel_s = [[3, 3], [3, 3], [3, 3, 3], [3, 3, 3]]
# num_dense = 2       # number of fully conected layers
# dense = [256, 32]  # number of neurons per fully conected layer
# dense_decoder = 14*14*256


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
#     self.decoder.add(Reshape((14, 14, 256)))
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





autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss='mae')

autoencoder.encoder.summary()
autoencoder.decoder.summary()

checkpoint_path = "ae.ckpt"
autoencoder.load_weights(checkpoint_path)


filenames = glob(os.path.join("database\\places_256\\a\\airfield" ,"*.jpg"))
for i, filename in enumerate(filenames[::10]):
    img = cv2.imread(filename)
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
   
    img = img.astype(float) * 1.0/255
    input_img = np.reshape(img, (1,256,256,3))
    latent_vec = autoencoder.encoder.predict(input_img)
    print(latent_vec)
    res_img = autoencoder.decoder.predict(latent_vec)
    output_img = np.reshape(res_img, (256,256,3))
    output_img = np.round(output_img * 255)
    output_img = output_img.astype("uint8")
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    plt.savefig("output" + str(i) + ".png")
    #plt.show()
