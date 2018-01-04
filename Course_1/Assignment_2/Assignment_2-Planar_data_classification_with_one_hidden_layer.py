# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:37:10 2017

@author: Alex
"""
# %% Directory


# %% File Imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

# Helper functions that come with assignment.
from Assignment_2.planar_utils import plot_decision_boundary, sigmoid,\
                         load_planar_dataset, load_extra_datasets

# %matplotlib inline

np.random.seed(1)    # set a seed so that the results are consistent

# %% Load Data
X, Y = load_planar_dataset()


# %% Plot the Data

plt.scatter(x=X[0, :],
            y=X[1, :],
            c=Y.ravel(),   # Colour of points matches Y
            s=40,    # point size
            cmap=plt.cm.Spectral)    # Colour theme


# %% Logistic Regression

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.ravel())

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y.ravel())

# Print accuracy
LR_predictions = clf.predict(X.T)
print('Logistic Regression Error Rate: ' +
      str(100
          - (float((np.dot(Y, LR_predictions) + np.dot(1-Y, 1-LR_predictions))
                   / float(Y.size)*100))) +
      '%')

# The present classification problem is not linearly separable.

# %% Implementing a One-layer neural network

# Steps to make a Neural Net:
#   1. Define Parameter shapes
#   2. Initialize Parameters
#   3. Gradient Descent. Loop:
#       - Forward Prop
#       - Back Prop
#       - Update values
#   4. Combine 1-3- to train the model
#   5. Make predictions with parameters from 4.

# %%

def parameter_shapes(X, Y, n_1):
    '''
    Generates values for n_0 and n_2 (i.e., n_x and n_y) for a 1 layer neural
    neural net.

    X is the training data with 2 features dims = (2, m)
    Y is a column vector of labels; dims = (1, m)
    n_1 is the number of units that should be in layer 1 (i.e., hidden layer)
    '''
    n_0 = X.shape[0]
    n_1 = n_1
    n_2 = Y.shape[0]

    return n_0, n_1, n_2


parameter_shapes(X, Y, n_1=4)

# %%

def initialize_parameters(n_0, n_1, n_2):
    '''
    Returns a dictionary 'parameters' containing W1, W2, b1, and b2
    W1 and W2 have randomly initialized values
    B1 and B2 are zeroes
    '''

    W1 = np.random.randn(n_1, n_0) * 0.01
    b1 = np.zeros((n_1, 1))
    W2 = np.random.randn(n_2, n_1) * 0.01
    b2 = np.zeros((n_2, 1))

    parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2}

    return parameters


test_params = initialize_parameters(2, 4, 1)

# %%


def forward_prop(X, parameters):
    '''
    Takes
        X: a (2, m) matrix of features
        parameters (W1, b1, W2, b2) previously initialized

    Computes forward propagation (tanh in layer 1 and sigmoid in layer 2)
    and returns activations (containing A1 and A2)
    '''
    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2 = 1 / (1 + np.exp(-Z2))

    assert(A2.shape == (1, X.shape[1]))

    activations = {
            "A1": A1,
            "A2": A2}

    return activations


test_activations = forward_prop(X, test_params)

# %%


def back_prop(X, parameters, Y, activations):
    '''
    Takes
        X as in forward_prop()
        parameters as in forward_prop()
        Y: (1, m) column vector of ground truth labels
        activations: dict containing A1 and A2 matrices from forward_prop()

    Computes derivatives for each of the parameters (W1, b1, W2, b2) and
    returns their gradients (in a variable caleld gradients)
    '''
    m = Y.shape[1]

#    dZ2 = np.subtract(activations["A2"], Y)
#    dW2 = np.multiply(1/m, np.dot(dZ2, activations["A1"].T))
#    db2 = np.multiply(1/m, np.sum(dZ2, axis=1, keepdims=True))
#
#    dZ1 = np.multiply(np.dot(dW2.T, dZ2), (1 - np.power(activations["A1"], 2)))
#    dW1 = np.multiply(1/m, np.dot(dZ1, X.T))
#    db1 = np.multiply(1/m, np.sum(dZ1, axis=1, keepdims=True))

    dZ2 = activations["A2"] - Y
    dW2 = 1/m * np.dot(dZ2, activations["A1"].T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(parameters["W2"].T, dZ2) * (1 - np.power(activations["A1"], 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}

    return gradients


test_grads = back_prop(X, test_params, Y, test_activations)


# %%

def update_parameters(parameters, gradients, alpha=0.01):
    '''
    Takes
        parameters as in forward_prop()
        gradients (i.e., parameter derivatives) computed in back_prop()
        alpha: the learning rate

    Updates each of the parameters. Returns updated parameters
    '''
    parameters["W1"] = parameters["W1"] - alpha * gradients["dW1"]
    parameters["b1"] = parameters["b1"] - alpha * gradients["db1"]
    parameters["W2"] = parameters["W2"] - alpha * gradients["dW2"]
    parameters["b2"] = parameters["b2"] - alpha * gradients["db2"]

    return parameters

test_params = update_parameters(test_params, test_grads)


# %%

def predict(A2):
    '''
    Takes A2, computes predictions.
    '''
    predictions = (A2 >= 0.5).astype(float)

    return predictions

# %%

def nn_model(X, Y, n_1, alpha=0.01, iterations=1000):
    '''
    Wrapper function.
    First evaluates parameter_shapes(), initialize_parameters().
    Then loops gradient descent consisting of forward_prop(), back_prop() and
    update_parameters()

    Takes
        X: feature matrix with dims (n_0, m). In the data I have loaded in this
            n_0 = 2.
        Y: ground truth labels for training examples (1, m)
        n_1: The number of hidden layer units
        alpha: learning rate
        iterations: number of gradient descent iterations

    Returns learned parameters for a 1 layer neural net with tanh in layer 1
    and sigmoid in layer 2.
    '''

    # Initialize
    n_0, n_1, n_2 = parameter_shapes(X, Y, n_1=4)
    parameters = initialize_parameters(n_0, n_1, n_2)

    # Gradient Descent
    for i in range(iterations):
        activations = forward_prop(X, parameters)
        gradients = back_prop(X, parameters, Y, activations)
        parameters = update_parameters(parameters, gradients, alpha=alpha)
#        print(activations["A2"][0][0])

        if i % 1000 == 0:
            error = 1 - np.mean(predict(activations['A2']) == Y)
            print("Iteration " + str(i) + ": " + str(error))

    return parameters, activations


# %%

my_params, my_activations = nn_model(X, Y, n_1=4, alpha=.05, iterations=20000)


#%%

