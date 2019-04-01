import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from tf_utils import load_dataset, convert_to_one_hot
from backwardPropagation import model
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = load_dataset()








# Take transpose of the input data and also normalize it

X_train = X_train.T

X_train = (X_train - X_train.mean()) / (X_train.max() - X_train.min())
X_train = X_train.fillna(0)
X_test = X_test.T

X_test = (X_test - X_test.mean()) / (X_test.max() - X_test.min())
X_test = X_test.fillna(0)








# Convert training and test labels to one hot matrices


y_train = to_categorical(y_train,9)
y_train = y_train.T

#print(y_train)
#print(y_train.shape)






y_test = to_categorical(y_test,9)
y_test = y_test.T


#print(X_train)

#print(y_train)
#print(X_test)
#print(y_test)





parameters = model(X_train,y_train,X_test,y_test)

print(parameters["W1"])









