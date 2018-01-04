"""
In this exercise, I am trying to implement a 2-layer neural network in order
to carry out the same image classification as in assignment 2 (cats vs. not
cats).

The script has the following outline:

* Load in data
* Convert image to vector
* Initialize parameters based on dimensions of the neural net
* Forward propagation using ReLU in layer 1 and sigmoid in layer 2
* Backward prop
* Update parameters
* Wrap gradient descent in a function and loop
"""

import matplotlib.pyplot as plt
import os
from scipy import ndimage

# Helper Functions
from Assignment_3.helpers_forward import *
from Assignment_3.helpers_backward import *

os.chdir('C:/Users/Alex/Documents/Python/DataSets/DeepLearningCoursera')

# Load in data
train_set_x_orig, train_set_y, test_set_x_orig, \
test_set_y, classes = load_dataset()

# Define globals:
X_TRAIN, X_TEST = im_to_vec(train_set_x_orig), im_to_vec(test_set_x_orig)
Y_TRAIN, Y_TEST = train_set_y, test_set_y
NN_MODEL_COST = []


def show_image(data=train_set_x_orig, index=None, Y=Y_TRAIN):
    """
    Plots an image and prints whether it is a cat image or not.
    :param data: Array of raw RBG image data to plot
    :param index: Which image to plot
    :param Y: Ground truth label
    :return: None
    """
    if index == None:
        index = np.random.randint(0, len(train_set_x_orig))
    plt.imshow(data[index])

    print("Image " + str(index) + ". It's a " + classes[
        Y[0, index]].decode("utf-8") + " picture.")


def nn_train(X, Y, model_dims=[7], alpha=0.01, iterations=1000, compute_cost=False):
    """
    Trains a neural network to recognize an image.

    :param X: Matrix of training data -- ndarray with dimensions (n_x, m)

    :param Y: Vector of true labels for training data -- ndarray with
      dimensions (1, m)

    :param model_dims: array or list, where each element corresponds to the
     number of nodes in a hidden unit. The length of model_dims is the number
     of HIDDEN units in the model (ignoring the input layer and the output
     layer). Examples:

     * [3, 4] -- 2 hidden units; layer 1 has 3 nodes and layer 2 has 4 nodes

     * [12] -- A single hidden layer with 12 nodes

    :param alpha: The learning rate. At each iteration of gradient descent, the
      parameters W and b are updated according to the formula:
      W -= alpha * dW
      b -= alpha * db

    :param iterations: The number of iterations of gradient descent.

    :param compute_cost: Compute cross-entropy cost? If True, computes the cost and
      plots the cost at each iteration.

    :return:
      * Updated parameters W and b for layers 1,..., L. These can be fed
        into nn_predict() in order to make predictions and assess error rates.
      * nn_model_cost. List of Cost at each iteration. If cost=False, this
        will be an empty list.

    """
    assert type(model_dims) == list, "Provide model_dims as a list"

    # Initialize parameters
    np.random.seed(2)
    dims = [len(X)] + model_dims + [1]
    parameters = initialize_parameters(dims)
    L = len(parameters)
    nn_model_cost = []

    # Gradient Descent
    for i in range(iterations):
        A, caches_z, caches_Aprev_W_b = forward_model(X, parameters)
        grads = backward_model(A[L], Y, caches_z, caches_Aprev_W_b)
        parameters = update_parameters(parameters, grads, alpha=alpha)
        if compute_cost and i % 10 == 0:
            nn_model_cost.append(cost(A[L], Y))

    if compute_cost:
        plt.plot(np.squeeze(nn_model_cost))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title('Learning rate: ' + str(alpha))
        plt.show()

    return parameters, nn_model_cost


def nn_predict(X, Y, parameters):
    pass



parameters, cost = nn_train(X_TRAIN, Y_TRAIN, [7],
                            alpha=0.0075, iterations=1000, compute_cost=True)


# dAL = -(np.divide(Y, A[L]) - np.divide(1 - Y, 1 - A[L]))
#
# # Note: caches_Aprev_W_b[L]["W"] is W[L]; caches_Aprev_W_b[L]['A_prev'] is A[L-1]
# grads = {}
#
# grads[L] = backward_linear_activation(dAL,
#                                       caches_Aprev_W_b[L]['A_prev'],
#                                       caches_Aprev_W_b[L]['W'],
#                                       caches_z[L],
#                                       activation='sigmoid')
#
# grads[1] = backward_linear_activation(grads[1 + 1]['dA_prev'],
#                                       caches_Aprev_W_b[1]['A_prev'],
#                                       caches_Aprev_W_b[1]['W'],
#                                       caches_z[1],
#                                       activation='relu')


# A1, Z1, cache1 = forward_linear_activation(X, parameters['W1'], parameters['b1'])
#
# Z1.shape
#
# A2, Z2, cache2 = forward_linear_activation(A1, parameters['W2'], parameters['b2'])
#
# A2.shape
# Z2.shape
#
