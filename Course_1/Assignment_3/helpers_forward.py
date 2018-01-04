from Assignment_3.helpers import *

def forward_linear(A_prev, W, b):
    """
    :param A_prev: Matrix of activation values from the previous layer with
      dimensions n^[l-1] * m. In the first layer, A = X.

    :param W: Matrix of weights with dims n^[l] * n^[l-1]

    :param b: Vector of bias parameters with dims n^[l] * 1

    :return: Z, a matrix containing the linear combinations of W*A + b
    """
    Z = np.dot(W, A_prev) + b
    cache_Aprev_W_b = {"A_prev": A_prev, "W": W, "b": b}

    # return Z, (A_prev, W, b)
    assert Z.shape == (W.shape[0], A_prev.shape[1])
    return Z, cache_Aprev_W_b

def forward_linear_activation(A_prev, W, b, activation='relu'):
    """
    :param A_prev: Matrix of activation values from the previous layer with
      dimensions n^[l-1] * m. In the first layer, A = X.

    :param W: Matrix of weights with dims n^[l] * n^[l-1]

    :param b: Vector of bias parameters with dims n^[l] * 1

    :return: Al, a matrix of activation outputs for layer `l`. First Z, the
    linear combination of W*A + b, is computed. A is equal to Z transformed
    # by either ReLU or sigmoid. Defaults to relu unless sigmoid is specified.
    """
    Z, cache_Aprev_W_b = forward_linear(A_prev, W, b)

    if activation.lower() == 'sigmoid':
        Al, cache_z = sigmoid(Z)  # Sigmoid function
    else:
        Al, cache_z = relu(Z)

    return Al, cache_z, cache_Aprev_W_b


def forward_model(X, parameters):
    """
     Computes forward propagation.

    :param X: The input vector with length n_x * 1

    :param parameters: The dictionary of parameters
      from initialize_parameters().

    :return:
      * Returns A, a dictionary containing the matrices A[0], ... A[L].
        `yhat`, is A[L] (i.e., the model's prediction).
      * caches_z, a dictionary containing all of the matrices of Z[1], ... Z[L]
      * caches_Aprev_W_b, a dictionary containing all of A_prev, W, and b. Note
        that caches_Aprev_W_b[l] will produce A[l-1], W[l], and b[l].
    """
    L = len(parameters)

    # Dictionaries:
    # caches contains all of the W and b parameters. caches[1][1] == `W1`
    # A contains all of the A matrices. A[1] = `A1`
    caches_Aprev_W_b = {}
    caches_z = {}
    A = {}
    A[0] = X


    #Forward Propagation
    for l in range(1, L):
        A[l], caches_z[l], caches_Aprev_W_b[l] = \
            forward_linear_activation(A[l - 1],
                                      parameters[l]['W'],
                                      parameters[l]['b'])

    A[L], caches_z[L], caches_Aprev_W_b[L] = \
        forward_linear_activation(A[L - 1],
                                  parameters[L]['W'],
                                  parameters[L]['b'],
                                  activation='sigmoid')

    assert(A[L].shape == (1, X.shape[1]))
    return A, caches_z, caches_Aprev_W_b


def cross_entropy_cost(AL, Y):
    """
    Provided in the course
    Implement the cost function

    :param AL: Probability vector corresponding to label predictions -
      shape (1, number of examples)

    :param Y: vector of true labels (0 if non-cat, 1 if cat) -
      shape (1, number of examples)

    :returns: cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = -(1. / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)))

    # Make sure shape is what we expect (e.g., turns [[17]] into 17).
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost

