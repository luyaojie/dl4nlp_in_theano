import logging

import theano.tensor as T

from __init__ import default_initializer
from dropout import dropout_from_layer
from utils import shared_rand_matrix, shared_zero_matrix

__author__ = 'roger'
logger = logging.getLogger(__name__)


class LogisticClassifier(object):
    def __init__(self, num_in, initializer=default_initializer, dropout=0, verbose=True):
        self.num_in = num_in
        self.num_out = 1
        self.dropout = dropout

        self.W = shared_rand_matrix(shape=(num_in, ), name="logistic_W", initializer=initializer)
        self.b = shared_zero_matrix(shape=(1,), name='logistic_b')
        self.params = [self.W, self.b]

        self.l1_norm = T.sum(T.abs_(self.W))
        self.l2_norm = T.sum(self.W ** 2)

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension : %d' % self.num_in)
            logger.debug('Output Label Num: %d' % self.num_out)
            logger.debug('Dropout Rate    : %f' % self.dropout)

    def forward(self, input):
        dropout_input = dropout_from_layer(input, self.dropout)
        return T.nnet.sigmoid(T.dot(dropout_input, self.W) + self.b[0])

    def forward_batch(self, input):
        return self.forward(input)

    def loss(self, input, truth):
        """
        negative log likelihood loss function
        :param input:
        :param truth: n_examples * label (0 or 1)
        :return:
        """
        predict = self.forward(input)
        return - T.mean(truth * T.log(predict) + (1 - truth) * T.log(1 - predict))


class SoftmaxClassifier(object):
    def __init__(self, num_in, num_out, initializer=default_initializer, dropout=0, verbose=True):
        self.num_in = num_in
        self.num_out = num_out
        self.dropout = dropout

        self.W = shared_rand_matrix(shape=(num_in, num_out), name="softmax_W", initializer=initializer)
        self.b = shared_zero_matrix((num_out, ), 'softmax_b')
        self.params = [self.W, self.b]
        self.l1_norm = T.sum(T.abs_(self.W))
        self.l2_norm = T.sum(self.W ** 2)

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension : %d' % self.num_in)
            logger.debug('Output Label Num: %d' % self.num_out)
            logger.debug('Dropout Rate    : %f' % self.dropout)

    def forward(self, input):
        dropout_input = dropout_from_layer(input, self.dropout)
        return T.nnet.softmax(T.dot(dropout_input, self.W) + self.b)

    def forward_batch(self, input):
        return self.forward(input)

    def loss(self, input, truth):
        """
        negative log likelihood loss function
        :param input
        :param truth: n_examples * label
        :return:
        """
        return -T.mean(T.log(self.forward(input))[T.arange(truth.shape[0]), truth])
