try:
   import cPickle as pickle
except:
   import pickle
import os
import numpy as np
import sys
import timeit
import theano
import theano.tensor as T
import pandas as pd
import utils.data as utld
import utils.nn as nn
import matplotlib.pyplot as plt

"""Optimize LeNet using gradient descent.

"""

def halt():
    sys.exit(0)

img_dataset_x = pickle.load(
    open('../data/img_dataset_train_x.pkl')
).astype(theano.config.floatX)

img_dataset_y = pickle.load(open('../data/img_dataset_train_y.pkl'))

ds = utld.dataset(img_dataset_x, img_dataset_y)

# Hyperparameters
learning_rate = 0.01
reg1_rate = 0.00 # regularization term 1 rate
reg2_rate = 0.00 # regularization term 2 rate

n_epochs = 20000

print('... building model')

x = T.dtensor4('x')
y = T.ivector('y')

rng = np.random.RandomState(4321)


# filters for LeNet:
filter_shapes = ((4, 4, 3, 3), (5, 4, 3, 3))

# filters for DeeperLeNet:
#filter_shapes = (
#                    (10, 10, 3, 3),
#                    (10, 8, 3, 3),
#                    (8, 6, 3, 3),
#                    (6, 6, 3, 3),
#                    (6, 4, 3, 3)
#                )

classifier = nn.LeNet(
    rng=rng,
    input=x,
    filter_shapes=filter_shapes,
    n_hidden=300
)
    
test_model = theano.function(
    inputs=[x, y],
    outputs=classifier.errors(y)
)

validate_model = theano.function(
    inputs=[x, y],
    outputs=classifier.errors(y)
)

cost = (
    classifier.negative_log_likelihood(y)
    + classifier.L1 * reg1_rate
    + classifier.L2_sqr * reg2_rate
)
    
# gradients
grads = T.grad(cost, classifier.params)

updates = [
    (param, param - learning_rate * grad)
    for param, grad in zip(classifier.params, grads)
]

train_model = theano.function(
    inputs=[x, y],
    outputs=[cost, classifier.errors(y)],
    updates=updates
)

start_time = timeit.default_timer()

# k-fold cross validation
for i in range(ds.n_cv_folds):
    # reset W and b
    classifier.reset()
    print('... training model for cross validation iteration %i' % i)
    # split kaggle train set into test, validation and train sets.
    test_set_xy, valid_set_xy, train_set_xy = ds.cv_split()

    test_set_x, test_set_y = test_set_xy
    valid_set_x, valid_set_y = valid_set_xy
    train_set_x, train_set_y = train_set_xy
    
    # Train Model:
    epoch = 0
    while (epoch < n_epochs):
        epoch = epoch + 1
        train_cost, train_error = train_model(train_set_x, train_set_y)
        if epoch % 100 == 0:
            print(
               '        Epoch = %i %%' % epoch
            )
            print(
               '        Train Cost = %f' % train_cost
            )
            print(
               '        Train Error = %.2f %%' % (train_error * 100.)
            )
            valid_error = validate_model(valid_set_x, valid_set_y)
            print(
               '       Validation Error = %.2f %%' % (valid_error * 100.)
            )
    # compute validation
    valid_error = validate_model(valid_set_x, valid_set_y)

    print(
        '       Training Error = %.2f %%' % (train_error * 100.)
    )

    print(
        '       Validation Error = %.2f %%' % (valid_error * 100.)
    )
    
test_error = test_model(test_set_x, test_set_y)

print(
    '       Test Error = %.2f %%' % (test_error * 100.)
)

end_time = timeit.default_timer()

print(
    'Code run for %d epochs, with %.1f minutes.' % (
        epoch, ((end_time - start_time) / 60))
)