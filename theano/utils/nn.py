try:
   import cPickle as pickle
except:
   import pickle
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

"""
Include desired neural network classes.

"""

class LogisticRegression(object):
    """Multi-class logistic regression."""

    def __init__(self, input, n_in, n_out):

        self.n_in = n_in
        self.n_out = n_out

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
        )

        # regularization term 1
        self.reg1 = (
            abs(self.W).sum()
            + abs(self.W).sum()
        )

        # regularization term 2
        self.reg2 = (
            (self.W ** 2).sum()
            + (self.W ** 2).sum()
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input

    def reset(self):
        """Reset W and b values.
        """

        self.W.set_value(np.zeros((self.n_in, self.n_out), 
                                    dtype=theano.config.floatX), 
                                    borrow=True)
        self.b.set_value(np.zeros((self.n_out,), 
                                    dtype=theano.config.floatX), 
                                    borrow=True)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in a batch."""

        # Check if y has same dimension of y_pred.
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # Check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    """Hidden layer for Multilayer perceptron."""

    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                  activation=T.tanh):

        self.input = input

        if W is None:
            W_vals = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == T.nnet.sigmoid:
                W_vals = W_vals * 4.

            W = theano.shared(value=W_vals, name='W', borrow=True)

        if b is None:
            b_vals = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_vals, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]

        self.n_in = n_in
        self.n_out = n_out

        self.rng = rng
        self.activation = activation


    def reset(self):
        W_vals = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (self.n_in + self.n_out)),
                    high=np.sqrt(6. / (self.n_in + self.n_out)),
                    size=(self.n_in, self.n_out)
                ),
                dtype=theano.config.floatX
            )

        if self.activation == T.nnet.sigmoid:
            W_vals = W_vals * 4.

        b_vals = np.zeros((self.n_out,), dtype=theano.config.floatX)

        self.W.set_value(W_vals, borrow=True)
        self.b.set_value(b_vals, borrow=True)


class ConvLayer(object):
    """Convolutional layer."""

    def __init__(self, rng, input, filter_shape, image_shape=None,
                            down_pooling=False,
                            pool_size=(2, 2),
                            activation=T.nnet.relu):

        if not image_shape is None:
            assert filter_shape[1] == image_shape[1]
        
        self.input = input

        self.fan_in = np.prod(filter_shape[1:])
        self.fan_out = filter_shape[0] * np.prod(filter_shape[2:])

        self.W_bound = np.sqrt(6. / (self.fan_in + self.fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-self.W_bound, high=self.W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_vals = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_vals, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            input_shape=image_shape,
            filter_shape=filter_shape
        )

        if down_pooling:
            conv_out = pool.pool_2d(
                input=conv_out,
                ds=pool_size,
                ignore_border=True
            )

        lin_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]

        self.filter_shape = filter_shape

        self.rng = rng

    def reset(self):
        """Reset W and b.

        """

        W_vals = np.asarray(
                self.rng.uniform(low=-self.W_bound, high=self.W_bound, size=self.filter_shape),
                dtype=theano.config.floatX
            )

        b_vals = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)

        self.W.set_value(W_vals, borrow=True)
        self.b.set_value(b_vals, borrow=True)


class LeNet(object):
    """Convolutional neural network from LeNet family."""

    def __init__(self, rng, input, filter_shapes, n_hidden, image_shape=None):

        conv_input = input

        self.layer3 = ConvLayer(
            rng=rng,
            input=conv_input,
            filter_shape=filter_shapes[0],
            image_shape=image_shape,
            down_pooling=True,
            activation=T.nnet.relu
        )

        self.layer2 = ConvLayer(
            rng=rng,
            input=self.layer3.output,
            filter_shape=filter_shapes[1],
            image_shape=image_shape,
            down_pooling=True,
            activation=T.nnet.relu
        )

        layer1_input = self.layer2.output.flatten(2)

        self.layer1 = HiddenLayer(
            rng=rng,
            input=layer1_input,
            n_in=105,
            n_out=n_hidden,
            activation=T.nnet.relu
        )

        self.layer0 = LogisticRegression(
            input=self.layer1.output,
            n_in=n_hidden,
            n_out=2
        )

        self.L1 = (
            abs(self.layer0.W).sum()
            + abs(self.layer1.W).sum()
            + abs(self.layer2.W).sum()
            + abs(self.layer3.W).sum()
        )

        self.L2_sqr = (
            (self.layer0.W ** 2).sum()
            + (self.layer1.W ** 2).sum()
            + (self.layer2.W ** 2).sum()
            + (self.layer3.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.layer0.negative_log_likelihood
        )

        self.errors = self.layer0.errors

        self.params = (
            self.layer0.params
            + self.layer1.params 
            + self.layer2.params 
            + self.layer3.params 
        )

        self.y_pred = self.layer0.y_pred

        self.input = input

    def reset(self):
        self.layer0.reset()
        self.layer1.reset()
        self.layer2.reset()
        self.layer3.reset()


class DeeperLeNet(object):
    """AlexNet-like convolutional neural network."""

    def __init__(self, rng, input, filter_shapes, n_hidden, image_shape=None):

        conv_input = input

        self.layer6 = ConvLayer(
            rng=rng,
            input=conv_input,
            filter_shape=filter_shapes[4],
            image_shape=image_shape,
            down_pooling=True,
            activation=T.nnet.relu
        )

        self.layer5 = ConvLayer(
            rng=rng,
            input=self.layer6.output,
            filter_shape=filter_shapes[3],
            image_shape=image_shape,
            down_pooling=False,
            activation=T.nnet.relu
        )

        self.layer4 = ConvLayer(
            rng=rng,
            input=self.layer5.output,
            filter_shape=filter_shapes[2],
            image_shape=image_shape,
            down_pooling=False,
            activation=T.nnet.relu
        )

        self.layer3 = ConvLayer(
            rng=rng,
            input=self.layer4.output,
            filter_shape=filter_shapes[1],
            image_shape=image_shape,
            down_pooling=False,
            activation=T.nnet.relu
        )

        self.layer2 = ConvLayer(
            rng=rng,
            input=self.layer3.output,
            filter_shape=filter_shapes[0],
            image_shape=image_shape,
            down_pooling=True,
            activation=T.nnet.relu
        )

        layer1_input = self.layer2.output.flatten(2)

        self.layer1 = HiddenLayer(
            rng=rng,
            input=layer1_input,
            n_in=110,
            n_out=n_hidden,
            activation=T.nnet.relu
        )

        self.layer0 = LogisticRegression(
            input=self.layer1.output,
            n_in=n_hidden,
            n_out=2
        )

        self.L1 = (
            abs(self.layer0.W).sum()
            + abs(self.layer1.W).sum()
            + abs(self.layer2.W).sum()
            + abs(self.layer3.W).sum()
            + abs(self.layer4.W).sum()
            + abs(self.layer5.W).sum()
            + abs(self.layer6.W).sum()
        )

        self.L2_sqr = (
            (self.layer0.W ** 2).sum()
            + (self.layer1.W ** 2).sum()
            + (self.layer2.W ** 2).sum()
            + (self.layer3.W ** 2).sum()
            + (self.layer4.W ** 2).sum()
            + (self.layer5.W ** 2).sum()
            + (self.layer6.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.layer0.negative_log_likelihood
        )

        self.errors = self.layer0.errors

        self.params = (
            self.layer0.params
            + self.layer1.params 
            + self.layer2.params 
            + self.layer3.params
            + self.layer4.params
            + self.layer5.params
            + self.layer6.params
        )

        self.y_pred = self.layer0.y_pred

        self.input = input

    def reset(self):
        self.layer0.reset()
        self.layer1.reset()
        self.layer2.reset()
        self.layer3.reset()
        self.layer4.reset()
        self.layer5.reset()
        self.layer6.reset()



def predict(dataset_x, classifier):
    """Predict labels."""

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred
    )

    predicted_values = predict_model(dataset_x)

    return predicted_values