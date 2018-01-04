# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:44:56 2017

Deep Learning - Programming Assignment 1

Logistic Regression

@author: Alex
"""
# %% Import Libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

# h5py doesn't seem to be able to read in file paths that have capital letters.
# Set file path:
os.chdir('C:/Users/Alex/Documents/Python/DataSets/DeepLearningCoursera/')

# %matplotlib inline

# %% Read in Data


def load_dataset():
    '''
    Function taken from the Coursera Jupyter Notebook Folders ('lr_utils.py')
    '''
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return(train_set_x_orig, train_set_y_orig, test_set_x_orig,
           test_set_y_orig, classes)


train_set_x_orig, train_set_y, test_set_x_orig, \
    test_set_y, classes = load_dataset()

# the 'orig' denotes that the set is pre-processed.

# %% Assignment Instructions


# Problem Statement:

# You are given a dataset containing:
#    - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
#    - a test set of m_test images labeled as cat or non-cat
#    - each image is of shape (num_pixels, num_pixels, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_pixels) and (width = num_pixels).
#
# You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.


# %% = Load an Image


# Example of a picture (copied code)
index = 9

# matplotlib knows how to read (m*n*3) RBG intensity arrays
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" +
      classes[np.squeeze(train_set_y[:, index])].decode("utf-8")
      + "' picture.")


# %% Data Processing

# %% = Explore Data Structure


# Dimensions of each data set:

print('train_set_x_orig:' + str(train_set_x_orig.shape))
print('train_set_y:' + str(train_set_y.shape))
print('test_set_x_orig:' + str(test_set_x_orig.shape))
print('test_set_y:' + str(test_set_y.shape))

# We have 209 training images that are 64x64 px with 3 channels (RGB)
# We have 50 test images.



# As an exercise, we will define values for:

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_pixels = train_set_x_orig.shape[1:3]    # Actually indexes 1:2



# The y sets are just row vectors.

print(classes)

# %% = Image To Vector Transformation


# For convenience, we will now convert our raw images into a matrix containing
# m_train in rows and examples n_x features.

# In general, when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b∗c∗d, a) is to use:
#X_flatten =  X.reshape(X.shape[0], -1).T
# Setting one dimension as -1 tells Numpy to infer the dimension.

train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T

train_set_x_flatten2 = train_set_x_orig.reshape(
        m_train, num_pixels[0] * num_pixels[1] * 3).T

(train_set_x_flatten == train_set_x_flatten2).all()
# These two data structures are the same


# We should also do a quick sanity check to make sure the im2vec was correct:

# Read the first observation, all 3 channels, first 2x2 pixel intensities
train_set_x_orig[0, 0:2, 0:2, :]
train_set_x_flatten[0:12, 0]


test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

# %% = Standardizing Data

# Since the pixel intensities range from 0 to 255, we will express them as a
# fraction of 1.

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# %% = Recap of what we have done

# So far we have:
# - Found the dimensions for each of our data sets
# - Reshaped our data set from the dimensions (m, px, px, 3) to (px*px*3, m)
# - Standardized our data set


# %% Building the Algorithm

# %% = General Architechture of the Learning Algorithm

# Logistic regression can be viewed as a simple neural network. Each pixel has
# become a feature of our logistic regession, which feeds in to our 'z'
# function (W^T*X + B). We then run the neuron through an 'activation function'
# which is the logistic function sigma(z) = 1 / (1 + e^-z). The output is a
# probability, which we then use to make a binary classification guess.

# Finally, we can compute a loss function and backprop to adjust the weights.

# The main steps for building a Neural Net are:
# 1. Define the model structure (such as the number of input features)
# 2. Initialize the model's parameters
# 3. Loop:
#    - Calculate the current loss (forward prop)
#    - Calculate the current gradient (back prop)
#    - Update the parameters (gradient descent)

# Often we build steps 1-3 separately and then integrate them into 1 function
# called 'model()'


# Note: We actually don't need explicitly code the cost function, only the
# derivative for it. As a fun exercise, you can calculate the cost function
# if you want to model performance over time. (Can you think of how you might
# implement this?)

# %% = Helper Functions

def sigmoid(z):
    '''
    Returns the logistic function (i.e., sigmoid function) of z.

    Z is a numpy array or single number
    '''
    s = 1 / (1 + np.exp(-z))
    return s


# Test
sigmoid(np.array([0, 2]))

# %% = Initialize Parameters


def initialize_with_zeros(dim):
    '''
    Input: dim is the length of the w vector (i.e., equal to m)

    Initializes logistic regression parameters w and b as empty data structures
    w is a column vector with dims (dim, 1) and b is a scalar 0.
    '''
    w = np.zeros((dim, 1))    # Column Vector
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, int))

    return w, b


# Test
w, b = initialize_with_zeros(3)
print(w, b)

# %% = Forward Propagation


def forward_prop(w, X, b):
    '''
    Inputs:
        w is the weight matrix (or vector in this example, (m * 1)
        X is the matrix of features (n_x * m)
        b is the bias parameter (vector with length m or int)

    Computes a vector of (length m) which is
    the linear combination of W.T * X + B and
    subjects it to the logistic function.

    Returns the classification (A)
    '''
    assert(w.T.shape[1] == X.shape[0]), \
        "w must be an array with dimensions [(m, 1)]."
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    return A


# Test: n_x = 2, m = 3
forward_prop(w=np.array([[2],
                         [-1]]),
             X=np.array([[1, 2, 3],
                         [0, 5, 6]]),
             b=-0.1)

# %% = Back Propagation


def back_prop(A, X, Y):
    '''
    Inputs:
        A, a row vector full of predicted labels
        X, the feature matrix (same as used in forward_prop()
        Y, a row vector of true object labels

    Computes the cost of the function and back performs
    backward propagation to compute derivatives
    '''
    assert A.shape == Y.shape
    m = X.shape[1]
    dZ = A - Y
    dB = 1/m * np.sum(dZ)
    dW = 1/m * np.dot(X, dZ.T)
    return dW, dB


# Test: n_x = 2, m = 3
back_prop(A=np.array([[ 0.86989153, 0.24973989, 0.47502081]]),
          X=np.array([[1, 2, 3],
                      [0, 5, 6]]),
          Y=np.array([[1, 0, 1]]))

# %% = Gradient Descent


# We now update out parameters

def gradient_descent(dW, dB, w, b, alpha=0.1):
    '''
    Inputs:
        dW, dB from back_prop()
        W, B are the weight matrix and bias
        alpha is the learning rate (0 to 1)

    Outputs: Updates values for W and B for the next round of forward prop
    '''
    w = np.subtract(w, (alpha * dW))
    b = np.subtract(b, (alpha * dB))

    return w, b


# Test
gradient_descent(dW=np.array([[-0.40185542], [-0.63372523]]),
                 dB=-0.13511592333333336,
                 w=np.array([[2.],
                            [-1.]]),
                 b=-0.1,
                 alpha=0.1)

# %% = Propagate (Wrapper for all three)


def propagate(w, X, b, Y, alpha=0.1):
    '''
    Inputs:
        w is the weight vector with dimensions m * 1.
        X is the feature matrix wih dimensions n_x * m.
        b is the bias vector (scalar)
        Y is the vector of true labels
        alpha is the learning rate

    propagate() is a wrapper for forward_prop(), backprop() and
    gradient_descent().

    Returns:
        Updated w and b values
        A, a vector of predicted probabilities
    '''
    A = forward_prop(w, X, b)
    dW, dB = back_prop(A, X, Y)
    w, b = gradient_descent(dW, dB, w, b, alpha)

    return w, b, A

# Test
propagate(w=np.array([[2],
                      [-1]]),
          X=np.array([[1, 2, 3],
                      [0, 5, 6]]),
          b=-0.1,
          Y=np.array([[1, 0, 1]]),
          alpha=0.1)

# %% = Predict

def predict(A, Y=None):
    '''
    Takes vector A. If probability >= 0.5, assigns 1. Else assigns 0.
    '''
    assert(A.max() <= 1)
    assert(A.min() >= 0)

    m = A.shape[1]
    labels = (A >= 0.5).astype(int)
    performance = None

    # Performance: predicted label vs true label
    if Y is not None:
        performance = 1/m * np.sum(Y == labels)
    return labels, performance

# Test
predict(A=np.array([[ 0.86989153, 0.24973989, 0.47502081]]))
predict(A=np.array([[ 0.86989153, 0.24973989, 0.47502081]]),
        Y=[1, 0, 1])

# %% = Model

def model(X, Y, alpha=0.1, iterations=10):
    '''
    Inputs:
        X is the feature matrix wih dimensions n_x * m.
        Y is the vector of true labels.
        alpha is the learning rate
        iterations is the number of iterations of gradient descent to perform

    model() is a wrapper for initialize_with_zeros, propagate(), and predict()
    At each iteration, it also compares predicted labels to the true labels to
    show model performance.

    Returns:
        Model performance (overall accuracy) per iteration (printed)
        w and b parameters (to use to make separate predictions)
    '''
    n_x = X.shape[0]
    m = X.shape[1]
    w, b = initialize_with_zeros(n_x)

    # Just for the performance graph
    performance_vector = np.zeros(iterations)

    for i in range(iterations):
        w, b, A = propagate(w, X, b, Y, alpha)
        labels, performance = predict(A, Y)
#        print('Iteration ' + str(i+1) + ': ' + str(performance*100) + '%.')
        performance_vector[i] = performance
    return w, b, labels, performance_vector


# %% Running the Model

w, b, labels, performance_vector = model(train_set_x,
                                         train_set_y,
                                         alpha=0.001,
                                         iterations=10000)

# Running the model 10000 times at alpha=0.001 gives training accuracy of 99%
# and a test accuracy of 70%

# Test set
A_test=forward_prop(w, test_set_x, b)
test_labels, test_performance = predict(A_test, test_set_y)


# %% Plotting Performance

plt.plot(performance_vector)
plt.ylabel('accuracy')
plt.xlabel('iterations')
plt.show()

# %% Viewing Misclassified Images

# Training misclassifications
misclassified_labels= (labels != train_set_y).reshape(m_train)
misclassified_images = train_set_x_orig[misclassified_labels]

plt.imshow(misclassified_images[0])


# Test mmisclassifications

misclassified_labels = (test_labels != test_set_y).reshape(m_test)
misclassified_images = test_set_x_orig[misclassified_labels]

plt.imshow(misclassified_images[14])


# %% Running the model on outside images

# I have cat files 1-4. Must be 64x64 px

cat_file = "cat4.jpg"

image = np.array(ndimage.imread(cat_file, flatten=False))
my_image = scipy.misc.imresize(image,
                               size=(num_pixels[0], num_pixels[1])).reshape(
                                       (1, num_pixels[0]*num_pixels[1]*3)).T

A_myimage = forward_prop(w, my_image, b)
my_label= predict(A_myimage)

plt.imshow(image)

if my_label[0] == 1:
    print("This is a cat")
else:
    print("This is not a cat")