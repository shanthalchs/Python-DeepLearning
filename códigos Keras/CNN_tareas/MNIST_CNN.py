#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 09:43:37 2021

@author: promidat04
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import math
import seaborn as sn  
from sklearn.metrics import confusion_matrix, classification_report  



np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

train = pd.read_csv('MNIST/ZipDataTestCod.csv',delimiter=";",decimal=".")
test = pd.read_csv('MNIST/ZipDataTrainCod.csv',delimiter=";",decimal="." )

Y_train = train["Numero"]
Y_test = test["Numero"]



# Removemos la columna "Numero"
X_train = train.drop(labels = ["Numero"],axis = 1) 
X_test = test.drop(labels = ["Numero"],axis = 1) 

# Gráfico de la distribución de la variable a predecir
g = sns.countplot(Y_train)

Y_train.value_counts()


#preparando los datos

# Normalizamos los datos
# define standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.reshape(-1,16,16,1)
X_test = X_test.reshape(-1,16,16,1)

# vamos a pasar a forma one-hot-encoding 
Y_train = to_categorical(Y_train, num_classes = 10)
Y_test = to_categorical(Y_test, num_classes = 10)


# visualizamos imagen
imgplot = plt.imshow(X_train[2])
plt.show()  


random_seed = 2

# Se define una arquitectura de dos capas convul con maxpool, una red
# neuronal con dos capas ocultas y como método de regulariz Dropout.
# como es un problema de clasificación multiclase, la función de activación
# para la capa final es softmax y la función de activación para las capas
# intermedias es ReLU

# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out


def create_simple_cnn():  

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (16,16,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
    
    # Compile the model
    model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

    return model


cnn_model = create_simple_cnn()  


#resumen del modelo
cnn_model.summary()

cnn=cnn_model.fit(x=X_train, y=Y_train, batch_size=86, epochs=10, verbose=1, validation_data=(X_test, Y_test), shuffle=True)  

cnn_pred = cnn_model.predict(X_test, batch_size=86, verbose=1)  
cnn_predicted = np.argmax(cnn_pred, axis=1)

#Con la librería Scikit Learn
# generamos la matriz de confusión y la dibujamos (aunque el gráfico no es muy bueno debido al numero de etiquetas):

#Creamos la matriz de confusión
cnn_cm = confusion_matrix(np.argmax(Y_test, axis=1), cnn_predicted)

# Visualiamos la matriz de confusión
cnn_df_cm = pd.DataFrame(cnn_cm, range(10), range(10))  
plt.figure(figsize = (20,14))  
sn.set(font_scale=1.4) #for label size  
sn.heatmap(cnn_df_cm, annot=True, annot_kws={"size": 12}) # font size  
plt.show()  

# métricas

cnn_report = classification_report(np.argmax(Y_test, axis=1), cnn_predicted)  
print(cnn_report)  


# visualizando algunos resultados

imgplot = plt.imshow(X_train[0])  
plt.show()  
print('class for image 1: ' + str(np.argmax(Y_test[0])))  
print('predicted:         ' + str(cnn_predicted[0])) 

