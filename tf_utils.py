import pandas
# from pattern.en import sentiment
# import HTMLParser
import re
import pandas as pd
import tensorflow as tf
from collections import Counter
from nltk.corpus import stopwords
import string
from collections import OrderedDict
from nltk import bigrams
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
# import plotly.plotly as py

# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
import requests
from bs4 import BeautifulSoup
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
import os


def load_dataset():
  X = pd.read_csv("C:\Shashank Reddy\DataSet_Final.csv", encoding="ISO-8859-1").fillna(0)

  Y = pd.read_csv("C:\Shashank Reddy\FinalTreatment.csv", encoding="ISO-8859-1").fillna(0)

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


  # test_size = 20 percent

  return X_train,X_test,y_train,y_test



def convert_to_one_hot(labels,C):
    """
       Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                        corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                        will be 1.

       Arguments:
       labels -- vector containing the labels
       C -- number of classes, the depth of the one hot dimension

       Returns:
       one_hot -- one hot matrix
       """
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name='C')

    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    # Create the session (approx. 1 line)
    sess = tf.Session()

    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()



    return one_hot



def create_placeholders(n_x,n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X,Y


def initialize_parameters():
    """
        Initializes parameters to build a neural network with tensorflow. The shapes are:
                            W1 : [12, 14]
                            b1 : [12, 1]
                            W2 : [12, 12]
                            b2 : [12, 1]
                            W3  : [10,12]
                            b3  : [10,1]
                            W4 : [9, 12]
                            b4 : [9, 1]

        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """


    W1 = tf.get_variable("W1", [12, 14], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [12, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 12], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [10, 12], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [10, 1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [9, 12], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [9, 1], initializer=tf.zeros_initializer())


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4":W4,
                  "b4":b4}

    return parameters



def forward_propagation(X, parameters):
    """
       Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

       Arguments:
       X -- input dataset placeholder, of shape (input size, number of examples)
       parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                     the shapes are given in initialize_parameters

       Returns:
       Z3 -- the output of the last LINEAR unit
       """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

                                                       # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                     # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                   # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                    # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                   # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
# Z3 = np.dot(W3,Z2) + b3


    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (9, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)


    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


    return cost


def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """


    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X.iloc[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((9,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X.iloc[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X.iloc[:, (m - (mini_batch_size * math.floor(m / mini_batch_size))): m]
        mini_batch_Y = shuffled_Y[:, (m - (mini_batch_size * math.floor(m / mini_batch_size))): m]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

