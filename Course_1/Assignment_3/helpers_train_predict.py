"""
This file contains the functions nn_train and nn_predict. These wrap around the
helper functions written in helpers (basic functions), helpers_backward
(functions for back prop), and helpers_forward (functions for forward prop)
"""


from Assignment_3.helpers_forward import *
from Assignment_3.helpers_backward import *


def nn_train(X, Y, model_dims=[7], alpha=0.01, iterations=1000, compute_cost=False):
    """
    Trains a neural network to recognize an image.

    :param X: Matrix of training data -- ndarray with dimensions (n_x, m)

    :param Y: Vector of true labels for training data -- ndarray with
      dimensions (1, m)

    :param model_dims: array or list, where each element corresponds to the
     number of nodes in a hidden unit. The length of model_dims is the number
     of HIDDEN units in the model. The output layer size is always 1. The input
	 layer will automatically adopt the size of the input vector. Examples:

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
            nn_model_cost.append(cross_entropy_cost(A[L], Y))

    if compute_cost:
        plt.plot(np.squeeze(nn_model_cost))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title('Learning rate: ' + str(alpha))
        plt.show()

    return parameters, nn_model_cost


def nn_predict(parameters, X=None, image=None, Y=None):
    """
    :param parameters: W and b for all layers 1,..., L. It is assumed that they
      have been tuned by nn_train()

    :param X: matrix of data - ndarray with dims (n_x, m)

    :param image: A 64*64 image with RGB channels. One of X or image must be
      supplied

    :param Y: vector of true labels - ndarray with dimensions(1,m), if available

    :return:
      * yhat - a vector of predicted labels
      * if Y is provided, will calculate the error rate.
    """
    assert (image is not None or X is not None), 'You must supply either an image or X'
    assert (image is None or X is None), 'You must supply either an image or X'
    L = len(parameters)

    if image is not None: X = im_to_vec(image)


    A, caches_z, caches_Aprev_W_b = forward_model(X, parameters)
    yhat = (A[L] > 0.5).astype(int)

    # error rate:
    if Y is not None: accuracy = np.mean(yhat == Y)
    else: accuracy = None

    return yhat, accuracy
