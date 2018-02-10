from Assignment_3.helpers import *

def backward_linear(dZ, A_prev, W):
    """
    Computes the gradients for the linear portion of back-prop (Given A[l-1] and
    dZ[l], computes dW[l], db[l], and dA[l-1].

    :param dZ: dZ[l]

    :param A_prev: A[l-1]

    :param W: W[l]

    :return:
      * dA_prev, which is dA[l-1]
      * dW, which is dW[l]
      * db, which is db[l]
    """
    m = np.shape(A_prev)[1]

    dA_prev = np.dot(W.T, dZ)
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)

    grad = {"dA_prev": dA_prev, "dW": dW, "db": db}

    return grad


def backward_linear_activation(dA, A_prev, W, Z, activation='relu'):
    """
    Wraps around backward_linear(). Given dA, it computes the gradient for the
    activation (i.e., dZ[l]) and feeds it into backward_linear. Output is the
    same as backward_linear()

    :param dA: dA[l]

    :param A_prev: A[l-1]

    :param W: W[l]

    :param Z: Z[l]

    :param activation: Specify type of activation function (relu or sigmoid).
      Defaults to relu.

    :return:
      * dA_prev, which is dA[l-1]
      * dW, which is dW[l]
      * db, which is db[l]
    """
    if activation.lower() == 'sigmoid':
        dZ = sigmoid_backward(dA, Z)
    else:
        dZ = relu_backward(dA, Z)

    return backward_linear(dZ, A_prev, W)


def backward_model(AL, Y, caches_z, caches_Aprev_W_b,):
    """
    Given: 
	* AL, the predictions for the model (output of forward_model)
	* Y, the true class labels
	backward_model computes the gradient of the cost function, dA[L].
    
	dA[L] is fed into backward_linear_activation to compute the remaining
    gradients, where layer [L] uses a sigmoid activation function, and [L-1] to
    1 use ReLU.

    :param AL: A[L], with dimensions 1 * m
	
    :param Y: Y a vector containing the true labels (m-dimensional)
    
	:param caches_z: Dictionary of length L containing all of Z[1],..., Z[L]
      computed in forward propagation
  
	:param caches_Aprev_W_b: Dictionary containing all of A_prev, W, and b
      for layers 1,..., L. Note that caches_Aprev_W_b[1] will return A[0], W[1],
      and b[1]

    :return: grads, a dictionary of the gradients for each parameter at every
    layer. e.g., dA[1],... dA[L]; W[1],..., W[L]; b[1],..., b[L]
    """
    grads = {}
    L = len(caches_z)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)    # Ensure that Y is an ndarray with same dims as AL

	# Backprop for layer L
    # dAL is the derivative of cost wrt to AL (dJ/dAL)    
	dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    	
    # Note: caches_Aprev_W_b[L]["W"] is W[L]; caches_Aprev_W_b[L]['A_prev'] is A[L-1]
    grads[L] = backward_linear_activation(dAL,
                                          caches_Aprev_W_b[L]['A_prev'],
                                          caches_Aprev_W_b[L]['W'],
                                          caches_z[L],
                                          activation='sigmoid')

    # Backprop for remaining layers
    for l in reversed(range(1, L)):
        grads[l] = backward_linear_activation(grads[l+1]['dA_prev'],
                                              caches_Aprev_W_b[l]['A_prev'],
                                              caches_Aprev_W_b[l]['W'],
                                              caches_z[l],
                                              activation='relu')

    return grads


def update_parameters(parameters, grads, alpha=0.01):
    """
    :param parameters: Dictionary containing W[1],..., W[L] and b[1],..., b[L]

    :param grads: Dictionary of gradients from backward_model(). Contains
      gradients dW[1],..., dW[L] and db[1],..., db[L]

    :param alpha:

    :return:
    """
    L = len(parameters)

    for l in range(1, L):
        parameters[l]['W'] -= alpha * grads[l]['dW']
        parameters[l]['b'] -= alpha * grads[l]['db']

    return parameters

