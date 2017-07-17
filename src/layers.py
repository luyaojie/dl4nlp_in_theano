import logging

import numpy as np
import theano
import theano.tensor as T

from activations import Activation
from classifier import SoftmaxClassifier
from dropout import dropout_from_layer, set_dropout_on
from optimizer import AdaGradOptimizer
from src import default_initializer
from utils import shared_rand_matrix, shared_zero_matrix, align_batch_size

__author__ = 'roger'
logger = logging.getLogger(__name__)


class HiddenLayer(object):
    def __init__(self, in_dim, hidden_dim, activation, prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.act = Activation(activation)
        self.dropout = dropout
        self.W = shared_rand_matrix((self.hidden_dim, self.in_dim), prefix + 'W', initializer)
        self.b = shared_zero_matrix((self.hidden_dim,), prefix + 'b')
        self.params = [self.W, self.b]
        self.norm_params = [self.W]
        self.l1_norm = T.sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = T.sum([T.sum(param ** 2) for param in self.norm_params])

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def forward(self, x):
        """
        :param x: (dim, )
        """
        dropout_x = dropout_from_layer(x, self.dropout)
        return self.act.activate(T.dot(self.W, dropout_x) + self.b)

    def forward_batch(self, x):
        """
        :param x: (batch, dim)
        """
        dropout_x = dropout_from_layer(x, self.dropout)
        # (batch, in) (in, hidden) + (None, hidden) -> (batch, hidden)
        return self.act.activate(T.dot(dropout_x, self.W.T) + self.b)


class MultiHiddenLayer(object):
    def __init__(self, in_dim, hidden_dims, activation='relu', prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = hidden_dims[-1]
        layer_num = len(self.hidden_dims)
        self.activations = [activation] * layer_num if type(activation) is not [list, tuple] else activation
        self.initializers = [initializer] * layer_num if type(initializer) is not [list, tuple] else initializer
        self.dropouts = [dropout] * layer_num if type(dropout) is not [list, tuple] else dropout
        self.layers = [HiddenLayer(d_in, d_h, activation=act, prefix=prefix + "layer%d_" % i,
                                   initializer=init, dropout=drop, verbose=verbose)
                       for d_in, d_h, act, i, init, drop
                       in zip([in_dim] + hidden_dims, hidden_dims, self.activations, range(layer_num),
                              self.initializers, self.dropouts)]
        self.params = []
        self.norm_params = []
        for layer in self.layers:
            self.params += layer.params
            self.norm_params += layer.norm_params
        self.l1_norm = T.sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = T.sum([T.sum(param ** 2) for param in self.norm_params])
        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Layer Num:  %d' % layer_num)

    def forward(self, x):
        hidden = x
        for layer in self.layers:
            hidden = layer.forward(hidden)
        return hidden

    def forward_batch(self, x):
        hidden = x
        for layer in self.layers:
            hidden = layer.forward_batch(hidden)
        return hidden


class HighwayLayer(HiddenLayer):
    def __init__(self, in_dim, activation, hidden_dim=None, transform_gate="sigmoid", prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        # By construction the dimensions of in_dim and out_dim have to match, and hence W_T and W_H are square matrices.
        if hidden_dim is not None:
            assert in_dim == hidden_dim
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        super(HighwayLayer, self).__init__(in_dim, in_dim, activation, prefix, initializer, dropout, verbose)
        self.transform_gate = Activation(transform_gate)
        self.W_H, self.W_H.name = self.W, prefix + "W_H"
        self.b_H, self.b_H.name = self.b, prefix + "b_H"
        self.W_T = shared_rand_matrix((self.hidden_dim, self.in_dim), prefix + 'W_T', initializer)
        self.b_T = shared_zero_matrix((self.hidden_dim,), prefix + 'b_T')
        self.params = [self.W_H, self.W_T, self.b_H, self.b_T]
        self.norm_params = [self.W_H, self.W_T]
        self.l1_norm = T.sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = T.sum([T.sum(param ** 2) for param in self.norm_params])

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Transform Gate:   %s' % self.transform_gate.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def forward(self, x):
        """
        :param x: (in, )
        """
        x_dropout = dropout_from_layer(x, self.dropout)
        # (in, in) (in, ) + (in, ) -> (in)
        t = self.transform_gate.activate(T.dot(self.W_T, x_dropout) + self.b_T)
        # (in, in) (in, ) + (in, ) -> (in)
        z_t = self.act.activate(T.dot(self.W_H, x_dropout) + self.b_H)
        # (in, ) * (in, ) + (in, ) * (in, ) -> (in, )
        return t * z_t + (1 - t) * x_dropout

    def forward_batch(self, x):
        """
        :param x: (batch, in)
        """
        x_dropout = dropout_from_layer(x, self.dropout)
        # (batch, in) (in, in) + (in, ) -> (batch, in)
        t = self.transform_gate.activate(T.dot(x_dropout, self.W_T.T) + self.b_T)
        # (batch, in) (in, in) + (in, ) -> (batch, in)
        z_t = self.act.activate(T.dot(x_dropout, self.W_H.T) + self.b_H)
        # (batch, in) * (batch, in) + (batch, in) * (batch, in) -> (batch, in)
        return t * z_t + (1 - t) * x_dropout


class EmbeddingClassifier(object):
    def __init__(self, lookup_table, in_dim, hidden_dims, num_label, activation,
                 batch_size=64, initializer=default_initializer, dropout=0, verbose=True):
        self.batch_size = batch_size
        word_index = T.imatrix()  # (batch, max_len)
        gold_truth = T.ivector()  # (batch, 1)
        encoder = MultiHiddenLayer(in_dim=in_dim, hidden_dims=hidden_dims, activation=activation,
                                   initializer=initializer, dropout=dropout, verbose=verbose)
        mask = (word_index > 0) * T.constant(1, dtype=theano.config.floatX)
        word_embedding = lookup_table.W[word_index]
        hidden = T.sum(word_embedding * mask[:, :, None], axis=1) / T.sum(mask, axis=1)[:, None]
        rnn_output = encoder.forward_batch(hidden)
        classifier = SoftmaxClassifier(num_in=encoder.out_dim, num_out=num_label, initializer=initializer)
        classifier_output = classifier.forward(rnn_output)
        loss = classifier.loss(rnn_output, gold_truth)
        params = lookup_table.params + classifier.params + encoder.params
        sgd_optimizer = AdaGradOptimizer(lr=0.95, norm_lim=16)
        except_norm_list = [param.name for param in lookup_table.params]
        updates = sgd_optimizer.get_update(loss, params, except_norm_list)

        self.train_x = shared_zero_matrix((batch_size, 1), dtype=np.int32)
        self.train_y = shared_zero_matrix(1, dtype=np.int32)
        self.dev_x = shared_zero_matrix((batch_size, 1), dtype=np.int32)
        self.test_x = shared_zero_matrix((batch_size, 1), dtype=np.int32)

        index = T.ivector()
        self.train_batch = theano.function(inputs=[index],
                                           outputs=[classifier_output, loss],
                                           updates=updates,
                                           givens={word_index: self.train_x[index],
                                                   gold_truth: self.train_y[index]}
                                           )
        self.get_norm = theano.function(inputs=[],
                                        outputs=[lookup_table.l2_norm, classifier.l2_norm])
        self.pred_train_batch = theano.function(inputs=[index],
                                                outputs=classifier_output,
                                                givens={word_index: self.train_x[index]}
                                                )
        self.pred_dev_batch = theano.function(inputs=[index],
                                              outputs=classifier_output,
                                              givens={word_index: self.dev_x[index]}
                                              )
        self.pred_test_batch = theano.function(inputs=[index],
                                               outputs=classifier_output,
                                               givens={word_index: self.test_x[index]}
                                               )

    def set_gpu_data(self, train, dev, test):
        self.train_x.set_value(train[0])
        self.train_y.set_value(train[1])
        self.dev_x.set_value(dev[0])
        self.test_x.set_value(test[0])

    def predict(self, x, predict_indexs, predict_function):
        num_batch = len(predict_indexs) / self.batch_size
        predict = list()
        for i in xrange(num_batch):
            indexs = predict_indexs[i * self.batch_size: (i + 1) * self.batch_size]
            predict.append(predict_function(indexs))
        return np.argmax(np.concatenate(predict), axis=1)[:len(x)]

    def train(self, train, dev, test):
        train_x, train_y = train
        dev_x, dev_y = dev
        test_x, test_y = test
        self.set_gpu_data(train, dev, test)
        train_index = align_batch_size(range(len(train_x)), self.batch_size)
        dev_index = align_batch_size(range(len(dev_x)), self.batch_size)
        test_index = align_batch_size(range(len(test_x)), self.batch_size)
        num_batch = len(train_index) / self.batch_size
        batch_list = range(num_batch)
        from sklearn.metrics import accuracy_score
        logger.info("epoch_num train_loss train_acc dev_acc test_acc")
        for j in xrange(100):
            loss_list = list()
            batch_list = np.random.permutation(batch_list)
            set_dropout_on(True)
            for i in batch_list:
                indexs = train_index[i * self.batch_size: (i + 1) * self.batch_size]
                output, loss = self.train_batch(indexs)
                loss_list.append(loss)
            set_dropout_on(False)
            logger.info("epoch %d" % j,)
            logger.info(np.mean(loss_list))
            train_pred = self.predict(train_x, train_index, self.pred_train_batch)
            logger.info(accuracy_score(train_y, train_pred),)
            dev_pred = self.predict(dev_x, dev_index, self.pred_dev_batch)
            logger.info(accuracy_score(dev_y, dev_pred),)
            test_pred = self.predict(test_x, test_index, self.pred_test_batch)
            logger.info(accuracy_score(test_y, test_pred),)
