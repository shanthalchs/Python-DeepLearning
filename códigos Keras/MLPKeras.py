#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:26:14 2021

@author: promidat04
"""
# pip install pydot
# pip install pydotplus
# pip install graphviz
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist
# carga la tabla de datos mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# calcula el número de etiquetas
num_labels = len(np.unique(y_train))

# conviete a código disyuntivo
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# dimensiones de la imagen
image_size = x_train.shape[1]
input_size = image_size * image_size

# cambia el tamaño y normaliza
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# parámetros de la red
batch_size = 128
hidden_units = 256
dropout = 0.45

# el modelo es un MLP de 3 capas con ReLU y “abandono” después de cada capa
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))

# salida de cada vector one-hot (disyuntivo)
model.add(Activation('softmax'))
model.summary()
plot_model(model, to_file='mlp-mnist.png', show_shapes=True)
# función de pérdida para un vector one-hot
# uso de adam optimizer
# la precisión es una buena métrica para las tareas de clasificación 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# entrena la red
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)
# valida el modelo en el conjunto de datos de prueba para determinar la generalización
_, acc = model.evaluate(x_test,
                        y_test,
		       batch_size=batch_size,
		       verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))
