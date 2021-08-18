#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:03:52 2021

@author: promidat04
"""

import numpy as np  
from scipy import misc  
from PIL import Image  
import glob  
import matplotlib.pyplot as plt  
import scipy.misc  
from matplotlib.pyplot import imshow  
from IPython.display import SVG  
import cv2  
import seaborn as sn  
import pandas as pd  
import pickle  
from keras import layers  
from keras.layers import Flatten, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout  
from keras.models import Sequential, Model, load_model  
from keras.preprocessing import image  
from keras.preprocessing.image import load_img  
from keras.preprocessing.image import img_to_array  
from keras.applications.imagenet_utils import decode_predictions  
from keras.utils import layer_utils, np_utils  
from keras.utils.data_utils import get_file  
from keras.applications.imagenet_utils import preprocess_input  
from keras.utils.vis_utils import model_to_dot  
from keras.utils import plot_model  
from keras.initializers import glorot_uniform  
from keras import losses  
import keras.backend as K  
from keras.callbacks import ModelCheckpoint  
from sklearn.metrics import confusion_matrix, classification_report  
import tensorflow as tf  





from keras.datasets import cifar100

# x_train_original y x_test_original son los conjuntos de datos con las imágenes de entrenamiento y validación respectivamente, mientras que y_train_original y y_test_original son los datasets con las etiquetas.
# (x_train_original, y_train_original), (x_test_original, y_test_original) = cifar100.load_data(label_mode='fine')

x_train_original = np.load('imagenes/imagenes_x_train.npy')

y_train_original = np.load('imagenes/imagenes_y_train.npy')


x_test_original = np.load('imagenes/imagenes_x_test.npy')
y_test_original = np.load('imagenes/imagenes_y_test.npy')


# vamos a pasar a forma one-hot-encoding 

y_train = np_utils.to_categorical(y_train_original, 10)  
y_test = np_utils.to_categorical(y_test_original, 10)  

# visualizamos imagen
imgplot = plt.imshow(x_train_original[0])
plt.show()  

# normalizar las imágenes dividiéndo cada elemento por el numero de píxeles, es decir, 255. Con lo que obteníamos el array con valores de entre 0 y 1:
x_train = x_train_original/255  
x_test = x_test_original/255  

#Preparamos entorno
#Especificábamos la situación de los canales de las imágenes y la fase del experimento:
# K.set_image_data_format('channels_last')  
# K.set_learning_phase(1)  


#Entranamos la red convolucional

#la instrucción Conv2D introduce una capa convolucional y la instrucción MaxPooling, la capa de pooling
# Para cada convolución usamos como función de activación ReLu.
# Dropout función de regularización Dropout.
def create_simple_cnn():  
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu'))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse']) 

  return model

#función de optimización: stochactic gradient descent 
#función de costo: cross entropy 
# métricas: accuracy y mse (media de los errores cuadráticos).
scnn_model = create_simple_cnn()  
#scnn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])  

#resumen del modelo
scnn_model.summary()

# entrenamos la red
# usaremos bloques de 32
# se darán 10 vueltas completas (epochs)
# Usaremos los datos para validar x_test e y_test.
# El resultado del entrenamiento se guarda en la variable scnn, de la cual, extraeremos el histórico de los datos del entrenamiento.
scnn=scnn_model.fit(x=x_train, y=y_train, batch_size=32, epochs=5, verbose=1, validation_data=(x_test, y_test), shuffle=True)  

plt.figure(0)  
plt.plot(scnn.history['acc'],'r')  
plt.plot(scnn.history['val_acc'],'g')  
plt.xticks(np.arange(0, 6, 1.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(scnn.history['loss'],'r')  
plt.plot(scnn.history['val_loss'],'g')  
plt.xticks(np.arange(0, 6, 1.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])

plt.show()  

#Matriz de confusión

#Vamos a hacer una predicción sobre el dataset de validación y, a partir de ésta, generamos la matriz de confusión y mostramos las métricas mencionadas anteriormente:
scnn_pred = scnn_model.predict(x_test, batch_size=32, verbose=1)  
scnn_predicted = np.argmax(scnn_pred, axis=1)

#Con la librería Scikit Learn
# generamos la matriz de confusión y la dibujamos (aunque el gráfico no es muy bueno debido al numero de etiquetas):

#Creamos la matriz de confusión
scnn_cm = confusion_matrix(np.argmax(y_test, axis=1), scnn_predicted)

# Visualiamos la matriz de confusión
scnn_df_cm = pd.DataFrame(scnn_cm, range(10), range(10))  
plt.figure(figsize = (20,14))  
sn.set(font_scale=1.4) #for label size  
sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12}) # font size  
plt.show()  

# métricas

scnn_report = classification_report(np.argmax(y_test, axis=1), scnn_predicted)  
print(scnn_report)  


# visualizando algunos resultados

imgplot = plt.imshow(x_train_original[0])  
plt.show()  
print('class for image 1: ' + str(np.argmax(y_test[0])))  
print('predicted:         ' + str(scnn_predicted[0])) 


# Salvamos el histórico

#Histórico
with open(path_base + '/scnn_history.txt', 'wb') as file_pi:  
  pickle.dump(scnn.history, file_pi)

#cargandolo
with open(path_base + '/simplenn_history.txt', 'rb') as f:  
  snn_history = pickle.load(f)
  
  
# gráficamos comparación  
# plt.figure(0)  
plt.plot(snn_history['val_acc'],'r')  
plt.plot(scnn.history['val_acc'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Simple NN Accuracy vs simple CNN Accuracy")  
plt.legend(['simple NN','CNN'])  


indices = np.array([])

for i in range(10):
    indices = np.append(indices,(y_train_original==i).nonzero()[0])