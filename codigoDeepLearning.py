import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from matplotlib import colors as mcolors
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

x = np.array(12) 
x
x.ndim

x = np.array([12, 3, 6, 14])
x
x.ndim

x = np.array([[7, 80, 4, 38, 2],[4, 77, 1, 33, 3],[5, 78, 2, 34, 1]])
x.ndim

x = np.array([[[7, 80, 4, 38, 2],[4, 77, 1, 33, 3],[5, 78, 2, 34, 1]],
[[7, 80, 4, 38, 2],[4, 77, 1, 33, 3],[5, 78, 2, 34, 1]],
[[7, 80, 4, 38, 2],[4, 77, 1, 33, 3],[5, 78, 2, 34, 1]]])

x.ndim

batch = train_images[:128]

batch = train_images[128:256]

batch = train_images[128 * n:128 * (n + 1)]

def naive_add(x, y):
  assert len(x.shape) == 2 
  assert x.shape == y.shape
  
  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]): x[i, j] += y[i, j]
  return x

import numpy as np 

z=x+y

z = np.maximum(z, 0.)

import numpy as np

z = np.dot(x, y)

def naive_vector_dot(x, y):
  assert len(x.shape) == 1
  assert len(y.shape) == 1
  assert x.shape[0] == y.shape[0]
  z = 0.
  for i in range(x.shape[0]):
    z += x[i] * y[i]
  return z
  
import numpy as np

def naive_matrix_vector_dot(x, y): 
  assert len(x.shape) == 2
  assert len(y.shape) == 1
  assert x.shape[1] == y.shape[0]
  
  z = np.zeros(x.shape[0])
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      z[i] += x[i, j] * y[j]
  return z
  
f(x + epsilon_x) = y + epsilon_y

f(x + epsilon_x) = y + a * epsilon_x

y_pred = dot(W, x)
loss_value = loss(y_pred, y)

loss_value = f(W)


A = [0.5, 1]

import pandas as pd

datos = pd.read_csv('iris.csv',delimiter=';',decimal=".")

print(datos.ndim)

