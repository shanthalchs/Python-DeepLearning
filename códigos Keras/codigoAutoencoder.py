#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:29:41 2021

@author: promidat04
"""

## muestra el codificador que comprime el dígito MNIST 
# en un vector latente de 16 dim. El codificador es una pila de dos Conv2D. 
## La etapa final es una capa densa con 16 unidades para generar el vector latente.

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# Carga el conjunto de datos MNIST
(x_train, _), (x_test, _) = mnist.load_data()

# remodelar a (28, 28, 1) y normalizar las imágenes de entrada
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# parametros de la red
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
latent_dim = 16

# codificador / decodificador número de capas CNN y filtros por capa
layer_filters = [32, 64]
# construye el modelo autoencoder 
# # primero construye el modelo del codificador
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# stack of Conv2D(32)-Conv2D(64)
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)
   # información de forma necesaria para construir el modelo de decodificador
    # para que no hagamos cálculos a mano
    # la entrada al primer decodificador
    # Conv2DTranspose tendrá esta forma
    # forma es (7, 7, 64) que es procesada por
    # el decodificador vuelve a (28, 28, 1)
shape = K.int_shape(x)
   # genera el vector latente
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)
      # # instanciar modelo de codificador
encoder = Model(inputs,
                latent,
                name='encoder')
encoder.summary()
plot_model(encoder,
           to_file='encoder.png',
           show_shapes=True)
 # construir el modelo de decodificador
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
  # usa la forma (7, 7, 64) que se guardó anteriormente
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
   # del vector a la forma adecuada para la conversión transpuesta
x = Reshape((shape[1], shape[2], shape[3]))(x)
   # stack of Conv2DTranspose(64)-Conv2DTranspose(32)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
# reconstruye la salida
outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)
#instanciar modelo de decodificador
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='decoder.png', show_shapes=True)
# autoencoder = encoder + decoder
# instanciar modelo de codificador automático
autoencoder = Model(inputs,
                    decoder(encoder(inputs)),
                    name='autoencoder')
autoencoder.summary()
plot_model(autoencoder,
           to_file='autoencoder.png',
           show_shapes=True)
# Función de pérdida de error cuadrático medio (MSE), optimizador de Adam
autoencoder.compile(loss='mse', optimizer='adam')
# entrena el modelo codificador
autoencoder.fit(x_train,
                x_train,
                validation_data=(x_test, x_test),
                epochs=1,
                batch_size=batch_size)
# predecir la salida del codificador automático a partir de los datos de prueba
x_decoded = autoencoder.predict(x_test)
# mostrar las primeras 8 entradas de prueba e imágenes decodificadas
imgs = np.concatenate([x_test[:8], x_decoded[:8]])
imgs = imgs.reshape((4, 4, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('input_and_decoded.png')
plt.show()

# El decodificador descomprime el vector latente para 
# recuperar el dígito MNIST. La etapa de entrada del decodificador es una capa
# densa que aceptará el vector latente. El número de unidades es igual al 
# producto de las dimensiones de salida de Conv2D guardadas del codificador. 
# Esto se hace para que podamos cambiar fácilmente el tamaño de la salida de 
# la capa Densa para Conv2DTranspose para finalmente recuperar las dimensiones
# originales de la imagen MNIST.