#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:44:31 2021

@author: promidat04
"""

import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Al visualizar el contenido de este dataset podemos 
# observar que cada registro contiene la información 
# de los valores más altos y más bajos alcanzados por 
# la acción, así como los valores de apertura y cierre
#  y el volumen de las transacciones:
dataset = pd.read_csv('AAPL.csv',delimiter=",",decimal=".", index_col=0,parse_dates=['Date'])
dataset.head()


# entrenaremos la Red LSTM usando únicamente el valor más alto 
# de la acción (columna High en el set de datos).

# para entrenamiento de la Red LSTM para training usaremos los datos de Enero de 2009 hasta Diciembre 2019
# para testing usaremos los datos del año 2020 

training = dataset[:'2019'].iloc[:,1:2]
testing = dataset['2020'].iloc[:,1:2]



#escalamos los datos
scaler = StandardScaler()
sc_training = scaler.fit_transform(training)


#Para entrenar la Red LSTM tomaremos bloques de 60 datos consecutivos,
# y la idea es que cada uno de estos permita predecir el siguiente valor:
    
# Los bloques de 60 datos serán almacenados en la variable X,
# mientras que el dato que se debe predecir (el dato 61 dentro de 
# cada secuencia) se almacenará en la variable Y y será usado como 
# la salida de la Red LSTM:
    
time_step = 5

X_train = []
Y_train = []

m = len(sc_training)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(sc_training[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(sc_training[i,0])
    
X_train, Y_train = np.array(X_train), np.array(Y_train)

#set de testing normalizamos los datos
x_test = testing.values
x_test = scaler.transform(x_test)

# Recordemos que el modelo fue entrenado para tomar 60 
# y generar un dato como predicción. Así que debemos
# reorganizar el set de validación (x_test) para que 
# tenga bloques de 60 datos:
    
X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))



X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 50

#creamos el modelo
#optmizador Gradiente Descendente y función de error error cuadrático medio

def create_simple_rnn():  

    modelo = Sequential()
    
    modelo.add(LSTM(units=na, input_shape=dim_entrada))
    modelo.add(Dense(units=dim_salida, activation='relu'))
    modelo.compile(optimizer='tanh', loss='mse')

    return modelo

rnn_modelo = create_simple_rnn()  


#resumen del modelo
rnn_modelo.summary()


rnn_modelo.fit(X_train,Y_train,epochs=100,batch_size=32)

prediccion = rnn_modelo.predict(X_test)

# aplicamos la normalización inversa de dicha predicción
prediccion = scaler.inverse_transform(prediccion)



plt.plot(training.index, training.values, label = " Training")
plt.plot(testing.index, testing.values, label = " Testing")
plt.plot(testing.index[0:len(X_test)], prediccion, label = "Prediccion")

plt.legend()


