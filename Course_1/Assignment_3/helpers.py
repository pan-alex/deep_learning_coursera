# Helper functions for Assignment 3

import h5py
import numpy as np
import matplotlib.pyplot as plt


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
    if len(image.shape) < 3: m = 1
    else: m = image.shape[0]

    vec = image.reshape(m, -1).T
    return vec


def initialize_parameters(dims):
    """
    :param layer_dims: Python array (list) containing the number of nodes in each
      layer in the neural network.

    :return:
        * parameters: a dictionary containing all parameters "W1", "b1" ... "WL", "BL"
        * Wl: the weight matrix of shape (layer_dims[1], layer_dims[l-1]
        * b1: a bias vector of shape (layer_dims[1], 1)
    """
    parameters = {}

    for l in range(1, len(dims)):
        W = np.random.randn(dims[l], dims[l-1]) * 0.01
        b = np.random.randn(dims[l], 1) * 0.01
        parameters[l] = {'W': W, 'b': b}

    return parameters


def sigmoid(z):
    '''
    :param z: an np array

    :return: Sigmoid activation function
    '''
    A = 1 / (1 + np.exp(-z))
    return A, z


def relu(z):
    """
    :param z: an np array

    :return: ReLU activation function

    Note: building the ReLU directly into the outer function with the form
      A = np.maximum(Z, 0, A)
      is the fastest way to implement ReLU. 3rd argument specifies inplace --
      ignores return value and writes answer directly to A.
    """
    A = np.maximum(z, 0)

    return A, z


def relu_backward(dA, cache_z):
    """
    Provided in the course
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- Dict containing Z, W, and b for layer `l`. 'Z' is in the first
    index position where we store for computing backward propagation efficiently
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache_z
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid_backward(dA, cache_z):
    """
    Provided in the course
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- Dict containing Z, W, and b for layer `l`. 'Z' is in the first
    index position where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache_z

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)
    return dZ


def show_image(data, Y, index=None,):
    """
    Plots an image and prints whether it is a cat image or not. Default
    arguments rely on data that are loaded-in in main.py
    :param data: Array of raw RBG image data to plot
    :param index: Which image to plot
    :param Y: Ground truth label
    :return: None
    """
    if index == None:
        index = np.random.randint(0, len(data))
    plt.imshow(data[index])

    if Y.squeeze()[index] == 1: is_cat = "a cat"
    else: is_cat = "not a cat"
    print("Image " + str(index) + ". It's " + is_cat + " picture.")