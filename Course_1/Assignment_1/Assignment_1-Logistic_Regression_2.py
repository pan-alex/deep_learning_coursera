# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:51:51 2017

Deep Learning - Programming Assignment 1

Logistic Regression

@author: Alex
"""

# Second attempt at The Deep Learning Coursera Assignment 1
# This is the same as Assignment_1-Logistic_Regression.py

# You are given a dataset containing:
#    - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
#    - a test set of m_test images labeled as cat or non-cat
#    - each image is of shape (num_pixels, num_pixels, 3) where 3 is for the 3
#      channels (RGB). Thus, each image is square (height = num_pixels) and
#      (width = num_pixels).
#
# You will build a simple image-recognition algorithm that can correctly
# classify pictures as cat or non-cat.


# %% Import Libraries

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

# %matplotlib inline

# %% Read in Data

os.chdir('C:/Users/Alex/Documents/Python/DataSets/DeepLearningCoursera')

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


# Load in data
train_set_x_orig, train_set_y, test_set_x_orig, \
    test_set_y, classes = load_dataset()

# the 'orig' denotes that the set is pre-processed.


# %%


def im_to_vec(image):
    '''
    Input: im is an array of images with dimensions (m, px1, px2, 3)
        Where m is the number observations
        px1 and px2 are the image dimensions
        3 is the number of colour channels (RBG from 0 to 255)

    Outputs:
        X, ndarray of image with dimensions (px1*px2*3, m)
        standardizes values from 0 to 1.
    '''
    if len(image.shape) > 3:
        m = image.shape[0]
    else:
        m = 1

    assert(np.amin(image) >= 0 and np.amax(image) <= 255)

    X = image.reshape(m, -1).T
    X = X / 255

    return X

# %%

def initialize_parameters(X):
    '''
    Input: X,the matrix of feature input with dimensions (n_x, m)
        from in_to_vec()

    Output: Initializes w and b. w is an array with dimensions (n_x, 1),
        b is a scalar
    '''
    w = np.zeros(shape=(X.shape[0], 1))
    b = 0

    return w, b


# %%

def activation(z):
    '''
    Input: Z, an np array

    Output: Sigmoid activation function
    '''
    s = 1 / (1 + np.exp(-z))

    return s

# %%

def forward_prop(X, w, b):
    '''
    Input:
        X from im_to_vec()
        w and b from initialize_parameters

    Output:
        A, Predicted values (not labels)
    '''
    Z = np.dot(w.T, X) + b
    A = activation(Z)

    return A


# %%

def back_prop(A, X, Y):
    '''
    Input: A from forward_prop(), which is a vector of floats with dims (1, m)
        X, the (n_x, m) input matrix from im_to_vec()
        Y, the vector of true labels with dims (1, m)

    Output: Values to dW and dB that will minimize the cost of the logistic
        function
    '''
    assert(A.shape == Y.shape)
    m = A.shape[1]

    dZ = A - Y
    dB = 1/m * np.sum(dZ)
    dW = 1/m * np.dot(X, dZ.T)

    return dW, dB

# %%

def gradient_descent(w, b, dW, dB, alpha=0.01):
    '''
    Inputs:
        w and b that are previously initialized
        dW and dB from back_prop()
        alpha is the learning rate

    Outputs:
        Updates w and b by the amounts dW or dB times alpha.
        Returns w and b
    '''
    w -= (alpha * dW)
    b -= (alpha * dB)

    return w, b

# %%

def probs_to_labels(A):
    '''
    Inputs:
        A, the (1, m) vector of predicted label probabilities

    Ouputs:
        label, a (1, m) vector of predicted labels. 1 if a >= 0.5, or else 0.

    This function is intended to be wrapped.
    '''
    label = (A >= 0.5).astype(int)
    return label



#%%


def train_model(image, Y, alpha=0.01, iterations=100, metrics=False):
    '''
    Inputs:
        image, an array of m images with RBG channels
        Y, The true labels (1, m)
        alpha is the learning rate
        iterations is the number of iterations of gradient descent to perform
        performance; If you want the function to return a vector of your
            model's performance

    Outputs:
        Vector of predicted classifications (1, m).
        If metrics = True, s
    '''
    performance = np.zeros(iterations)

    X = im_to_vec(image)
    w, b = initialize_parameters(X)

    for i in range(iterations):
            A = forward_prop(X, w, b)
            dW, dB = back_prop(A, X, Y)
            w, b = gradient_descent(w, b, dW, dB, alpha = alpha)
            if metrics == True:
                label = probs_to_labels(A)
                performance[i] = np.mean((label == Y))

    if metrics == False:
        return w, b
    else:
        return w, b, performance



# %%


def predict(w, b, image=None, X=None, Y=None):
    '''
    Inputs:
        w, b are the LEARNED weights after training the model.
        image, an array of m images with RBG channels
        Y, The true labels (1, m)

    Outputs:
        Vector of predicted classifications (1, m).
        If Y is supplied, computes prediction accuracy.
    '''
    assert(image is not None or X is not None), 'You must supply either an image or X'


    if image is None:
        A = forward_prop(X, w, b)
        label = probs_to_labels(A)
    else:
        X = im_to_vec(image)
        A = forward_prop(X, w, b)
        label = probs_to_labels(A)

    if Y is None:
        return label
    else:
        metrics = np.mean((label == Y))
        return label, metrics

# %% Run Model

w, b, metrics = train_model(train_set_x_orig,
                            train_set_y,
                            iterations=5000,
                            metrics=True)

predict(w, b, test_set_x_orig, Y=test_set_y)


# %% Running the model on outside images

# I have cat files 1-4. Must be 64x64 px

cat_file = "cat1.jpg"
image = np.array(ndimage.imread(cat_file, flatten=False))

label = predict(w, b, image, 0)

plt.imshow(image)

if label[0] == 1:
    print("This is a cat")
else:
    print("This is not a cat")