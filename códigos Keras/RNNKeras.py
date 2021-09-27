#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:37:37 2021

@author: promidat04
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist
   # carga los datos mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
   # calcula el número de etiquetas
num_labels = len(np.unique(y_train))
   # convierte a one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
   # cambia el tamaño y normaliza
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size])
x_test = np.reshape(x_test,[-1, image_size, image_size])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
   # parámetros de la red
input_shape = (image_size, image_size)
batch_size = 128
units = 256
dropout = 0.2
   # el modelo es RNN con 256 unidades, la entrada es un vector de 28 dim 28 pasos de tiempo
model = Sequential()
model.add(SimpleRNN(units=units,
                       dropout=dropout,
                       input_shape=input_shape))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
plot_model(model, to_file='rnn-mnist.png', show_shapes=True)
   # función de pérdida para un vector one-hot
   # uso del optimizador sgd
   # la precisión es una buena métrica para las tareas de clasificación
model.compile(loss='categorical_crossentropy',
                 optimizer='sgd',
                 metrics=['accuracy'])
   # entrena la red
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)
_, acc = model.evaluate(x_test,
                           y_test,
                           batch_size=batch_size,
                           verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))