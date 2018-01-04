"""
In this exercise, I am trying to implement a neural network in order
to carry out the same image classification as in assignment 1 (cats vs. not
cats).

Although the assignment given in the course is more of a "fill in the blanks"
type exercise, I have tried to implement the functions to create functions to
train a neural net and predict observations 'from scratch.'

There are a number of helper functions:

* helpers.py -- Basic helper functions including load_dataset(), show_image(),
  relu(), sigmoid(), relu_backward(), and sigmoid_backward()

* helpers_forward -- Helper functions for forward propagation including
  forward_model() and cross_entropy_cost()

* helpers_backward -- Helper functions for backward prop, including
  backward_model() and update_parameters()

* helpers_train_predict -- Wrap functions from helpers_forward and
  helpers_backward into nn_predict() and nn_train(). nn_train() is meant to be
  easily used out of the box; parameters such as layers/nodes, learning rate,
  and iterations can be adjusted. However, learning rate is constant throughout.
"""

import os
# from scipy import ndimage

# Helper Functions
from Assignment_3.helpers_train_predict import *

os.chdir('C:/Users/Alex/Documents/Python/DataSets/DeepLearningCoursera')

# Load in data
train_set_x_orig, train_set_y, test_set_x_orig, \
test_set_y, classes = load_dataset()

# Define globals:
X_TRAIN, X_TEST = im_to_vec(train_set_x_orig), im_to_vec(test_set_x_orig)
Y_TRAIN, Y_TEST = train_set_y, test_set_y
NN_MODEL_COST = []

#####

# Open up a random picture
show_image(train_set_x_orig, Y_TRAIN)

#####

# Models:

# 2 Layer NN (1 Hidden layer)
parameters, cost = nn_train(X_TRAIN, Y_TRAIN, model_dims=[7],
                            alpha=0.005, iterations=2000, compute_cost=True)

yhat, accuracy = nn_predict(parameters, X=X_TRAIN, Y=Y_TRAIN)
print(accuracy)    # .97, 0.90

yhat, accuracy = nn_predict(parameters, X=X_TEST, Y=Y_TEST)
print(accuracy)    # .74, 0.66


# 4 Layer NN (3 Hidden layers)
parameters, cost = nn_train(X_TRAIN, Y_TRAIN, model_dims=[20, 7, 5],
                            alpha=0.005, iterations=4000, compute_cost=True)

yhat, accuracy = nn_predict(parameters, X=X_TRAIN, Y=Y_TRAIN)
print(accuracy)    # 1.

yhat, accuracy = nn_predict(parameters, X=X_TEST, Y=Y_TEST)
print(accuracy)    # .82