import logging
from abc import ABCMeta, abstractmethod

import theano
import theano.tensor as T

from Initializer import OrthogonalInitializer
from __init__ import default_initializer
from activations import Activation
from dropout import dropout_from_layer
from pooling import get_pooling, get_pooling_batch
from utils import shared_rand_matrix, shared_zero_matrix, ndarray_slice, shared_matrix

__author__ = 'roger'
logger = logging.getLogger(__name__)


class AbstractRecurrentEncoder(object):

    __metaclass__ = ABCMeta

    def __init__(self, in_dim, hidden_dim, pooling, activation='tanh', dropout=0):
        self.in_dim = in_dim
        self.out_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.dropout = dropout
        self.act = Activation(activation)

    @abstractmethod
    def forward_scan(self, x): pass

    @abstractmethod
    def forward_scan_batch(self, x, mask): pass

    def forward_sequence(self, x):
        dropout_x = dropout_from_layer(x, self.dropout)
        return self.forward_scan(dropout_x)

    def forward_sequence_batch(self, x, mask):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        """
        dropout_x = dropout_from_layer(x, self.dropout)
        return self.forward_scan_batch(dropout_x, mask)

    def forward(self, x):
        """
        :param x: (len, dim)
        """
        # Use Pooling to reduce into a fixed-length representation
        return get_pooling(self.forward_sequence(x), self.pooling)

    def forward_batch(self, x, mask):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        """
        # Use Pooling to reduce into a fixed-length representation
        # (max_len, batch, dim) -> (batch, max_len, dim) -> (batch, dim)
        hidden = self.forward_sequence_batch(x, mask)
        return get_pooling_batch(hidden, mask, self.pooling)


class RecurrentEncoder(AbstractRecurrentEncoder):
    def __init__(self, in_dim, hidden_dim, pooling, activation='tanh', prefix="", initializer=default_initializer,
                 dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        super(RecurrentEncoder, self).__init__(in_dim, hidden_dim, pooling, activation, dropout)

        self.in_dim = in_dim
        self.out_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.dropout = dropout
        self.act = Activation(activation)
        # Composition Function Weight
        # Feed-Forward Matrix (hidden, in)
        self.W = shared_rand_matrix((self.hidden_dim, self.in_dim), prefix + 'W_forward', initializer)
        # Bias Term (hidden)
        self.b = shared_zero_matrix((self.hidden_dim,), prefix + 'b_forward')
        # Recurrent Matrix (hidden, hidden)
        self.U = shared_rand_matrix((self.hidden_dim, self.hidden_dim), prefix + 'U_forward', initializer)

        self.params = [self.W, self.U, self.b]
        self.norm_params = [self.W, self.U]

        # L1, L2 Norm
        self.l1_norm = T.sum(T.abs_(self.W)) + T.sum(T.abs_(self.U))
        self.l2_norm = T.sum(self.W ** 2) + T.sum(self.U ** 2)

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def _step(self, x_t, h_t_1, w, u, b):
        """
        step function of forward
        :param x_t:   (in, )
        :param h_t_1: (hidden, )
        :param w:     (hidden, in)
        :param u:     (hidden, hidden)
        :param b:     (hidden, )
        :return:      (hidden)
        """
        # (hidden, in) (in, ) + (hidden, hidden) (hidden, ) + (hidden, ) -> hidden
        h_t = self.act.activate(T.dot(w, x_t) + T.dot(u, h_t_1) + b)
        return h_t

    def _step_batch(self, x_t, mask, h_t_1, w, u, b):
        """
        step function of forward in batch version
        :param x_t:   (batch, in)
        :param mask:  (batch, )
        :param h_t_1: (batch, hidden)
        :param w:     (hidden, in)
        :param u:     (hidden, hidden)
        :param b:     (hidden)
        :return:      (batch, hidden)
        """
        # (batch, in) (in, hidden) -> (batch, hidden)
        h_t = self.act.activate(T.dot(x_t, w.T) + T.dot(h_t_1, u.T) + b)
        # (batch, hidden) * (batch, None) + (batch, hidden) * (batch, None) -> (batch, hidden)
        return h_t * mask[:, None] + h_t_1 * (1 - mask[:, None])

    def forward_scan(self, x):
        h0 = T.zeros((self.hidden_dim, ))
        hs, _ = theano.scan(fn=self._step,
                            sequences=x,
                            outputs_info=[h0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        return hs

    def forward_scan_batch(self, x, mask):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        """
        h0 = T.zeros((x.shape[0], self.hidden_dim))
        hs, _ = theano.scan(fn=self._step_batch,
                            sequences=[T.transpose(x, (1, 0, 2)),  # (batch, max_len, dim) -> (max_len, batch, dim)
                                       T.transpose(mask, (1, 0))],     # (batch, max_len) -> (max_len, batch)
                            outputs_info=[h0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        # (max_len, batch, dim) -> (batch, max_len, dim)
        return T.transpose(hs, (1, 0, 2))


class MultiLayerRecurrentEncoder(object):
    def __init__(self, in_dim, hidden_dims, pooling, activation='tanh', prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = hidden_dims[-1]
        layer_num = len(self.hidden_dims)
        self.activations = [activation] * layer_num if type(activation) is not [list, tuple] else activation
        self.poolings = [pooling] * layer_num if type(pooling) is not [list, tuple] else pooling
        self.initializers = [initializer] * layer_num if type(initializer) is not [list, tuple] else initializer
        self.dropouts = [dropout] * layer_num if type(dropout) is not [list, tuple] else dropout
        self.layers = [RecurrentEncoder(d_in, d_h, pooling=pool, activation=act, prefix=prefix + "layer%d_" % i,
                                        initializer=init, dropout=drop, verbose=verbose)
                       for d_in, d_h, pool, act, i, init, drop
                       in zip([in_dim] + hidden_dims, hidden_dims, self.poolings, self.activations, range(layer_num),
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

    def forward_sequence(self, x):
        hidden = x
        for layer in self.layers:
            hidden = layer.forward_sequence(hidden)
        return hidden

    def forward_sequence_batch(self, x, mask):
        hidden = x
        for layer in self.layers:
            hidden = layer.forward_sequence_batch(hidden, mask)
        return hidden

    def forward(self, x):
        """
        :param x: (len, dim)
        """
        # Use Pooling to reduce into a fixed-length representation
        return get_pooling(self.forward_sequence(x), self.poolings[-1])

    def forward_batch(self, x, mask):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        """
        # Use Pooling to reduce into a fixed-length representation
        # (max_len, batch, dim) -> (batch, max_len, dim) -> (batch, dim)
        hidden = self.forward_sequence_batch(x, mask)
        return get_pooling_batch(hidden, mask, self.poolings[-1])


class LSTMEncoder(AbstractRecurrentEncoder):
    def __init__(self, in_dim, hidden_dim, pooling, activation='tanh', gates=("sigmoid", "sigmoid", "sigmoid"),
                 prefix="", initializer=OrthogonalInitializer(), dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        super(LSTMEncoder, self).__init__(in_dim, hidden_dim, pooling, activation, dropout)
        self.in_gate, self.forget_gate, self.out_gate = Activation(gates[0]), Activation(gates[1]), Activation(gates[2])

        # W [in, forget, output, recurrent] (4 * hidden, in)
        self.W = shared_rand_matrix((self.hidden_dim * 4, self.in_dim), prefix + 'W', initializer)
        # U [in, forget, output, recurrent] (4 * hidden, hidden)
        self.U = shared_rand_matrix((self.hidden_dim * 4, self.hidden_dim), prefix + 'U', initializer)
        # b [in, forget, output, recurrent] (4 * hidden,)
        self.b = shared_zero_matrix((self.hidden_dim * 4,), prefix + 'b')

        self.params = [self.W, self.U, self.b]
        self.l1_norm = T.sum(T.abs_(self.W)) + T.sum(T.abs_(self.U))
        self.l2_norm = T.sum(self.W ** 2) + T.sum(self.U ** 2)

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Input Gate:       %s' % self.in_gate.method)
            logger.debug('Forget Gate:      %s' % self.forget_gate.method)
            logger.debug('Output Gate:      %s' % self.out_gate.method)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def _step(self, x_t, h_t_1, c_t_1, w, u, b):
        pre_calc = T.dot(w, x_t) + T.dot(u, h_t_1) + b
        i_t = self.in_gate.activate(ndarray_slice(pre_calc, 0, self.hidden_dim))
        f_t = self.forget_gate.activate(ndarray_slice(pre_calc, 1, self.hidden_dim))
        o_t = self.out_gate.activate(ndarray_slice(pre_calc, 2, self.hidden_dim))
        g_t = self.act.activate(ndarray_slice(pre_calc, 3, self.hidden_dim))
        c_t = f_t * c_t_1 + i_t * g_t
        h_t = o_t * self.act.activate(c_t)
        return h_t, c_t

    def _step_batch(self, x_t, m_t, h_t_1, c_t_1, w, u, b):
        # (batch, in) (in, hidden * 4) + (hidden, in) (in, hidden * 4) + (hidden * 4)
        #   -> (batch, hidden * 4)
        pre_calc = T.dot(x_t, w.T) + T.dot(h_t_1, u.T) + b
        # (batch, hidden * 4) -> (batch, hidden) (batch, hidden) (batch, hidden) (batch, hidden)
        i_t = self.in_gate.activate(ndarray_slice(pre_calc, 0, self.hidden_dim))
        f_t = self.forget_gate.activate(ndarray_slice(pre_calc, 1, self.hidden_dim))
        o_t = self.out_gate.activate(ndarray_slice(pre_calc, 2, self.hidden_dim))
        g_t = self.act.activate(ndarray_slice(pre_calc, 3, self.hidden_dim))
        # (batch, hidden) * (batch, hidden) + (batch, hidden) * (batch, hidden)
        # -> (batch, hidden)
        c_t = f_t * c_t_1 + i_t * g_t
        # (batch, hidden) * (batch, hidden) -> (batch, hidden)
        h_t = o_t * self.act.activate(c_t)
        c_t = m_t[:, None] * c_t + (1. - m_t)[:, None] * c_t_1
        h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_t_1
        # (batch, hidden) (batch, hidden)
        return h_t, c_t

    def forward_scan(self, x):
        h0 = shared_zero_matrix((self.hidden_dim,), 'h0_forward')
        c0 = shared_zero_matrix((self.hidden_dim,), 'c0_forward')
        hs, _ = theano.scan(fn=self._step,
                            sequences=x,
                            outputs_info=[h0, c0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        return hs[0]

    def forward_scan_batch(self, x, mask):
        h0 = T.zeros((x.shape[0], self.hidden_dim))
        c0 = T.zeros((x.shape[0], self.hidden_dim))
        hs, _ = theano.scan(fn=self._step_batch,
                            sequences=[T.transpose(x, (1, 0, 2)),
                                       T.transpose(mask, (1, 0))],
                            outputs_info=[h0, c0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        return T.transpose(hs[0], (1, 0, 2))


class GRUEncoder(AbstractRecurrentEncoder):
    def __init__(self, in_dim, hidden_dim, pooling, activation='tanh', gates=("sigmoid", "sigmoid"),
                 prefix="", initializer=default_initializer, dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        super(GRUEncoder, self).__init__(in_dim, hidden_dim, pooling, activation, dropout)
        self.reset_gate, self.update_gate = Activation(gates[0]), Activation(gates[1])

        # W [reset, update, recurrent] (3 * hidden, in)
        self.W = shared_rand_matrix((self.hidden_dim * 3, self.in_dim), prefix + 'W', initializer)
        # U [reset, update, recurrent] (3 * hidden, hidden)
        self.U = shared_rand_matrix((self.hidden_dim * 3, self.hidden_dim), prefix + 'U', initializer)
        # b [reset, update, recurrent] (3 * hidden,)
        # self.b = shared_zero_matrix((self.hidden_dim * 3,), prefix + 'b')

        self.params = [self.W, self.U]  # , self.b]
        self.l1_norm = T.sum(T.abs_(self.W)) + T.sum(T.abs_(self.U))
        self.l2_norm = T.sum(self.W ** 2) + T.sum(self.U ** 2)

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Reset Gate:       %s' % self.reset_gate.method)
            logger.debug('Update Gate:      %s' % self.update_gate.method)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def _step(self, x_t, h_t_1, w, u):
        # (hidden * 2, in)
        reset_update_w = w[:self.hidden_dim * 2, :]
        # (hidden * 2, hidden)
        reset_update_u = u[:self.hidden_dim * 2, :]
        # (hidden, in)
        recurrent_w = w[self.hidden_dim * 2:, :]
        # (hidden, hidden)
        recurrent_u = u[self.hidden_dim * 2:, :]
        # (in,) dot (in, hidden * 2) + (hidden,) dot (hidden, hidden * 2) -> (hidden * 2)
        pre_calc = T.dot(x_t, reset_update_w.T) + T.dot(h_t_1, reset_update_u.T)
        # (hidden * 2) -> (hidden) (hidden)
        reset_t = self.reset_gate.activate(ndarray_slice(pre_calc, 0, self.hidden_dim))
        update_t = self.update_gate.activate(ndarray_slice(pre_calc, 1, self.hidden_dim))
        # (in,) dot (in, hidden) + [(hidden,) * (hidden,)] dot (hidden, hidden)-> (hidden, )
        g_t = T.dot(x_t, recurrent_w.T) + T.dot(h_t_1 * reset_t, recurrent_u.T)
        # (hidden,) * (hidden,) + (hidden,) * (hidden,) -> (hidden,)
        h_t = update_t * h_t_1 + (1 - update_t) * g_t
        return h_t

    def _step_batch(self, x_t, m_t, h_t_1, w, u):
        # (hidden * 2, in)
        reset_update_w = w[:self.hidden_dim * 2, :]
        # (hidden * 2, hidden)
        reset_update_u = u[:self.hidden_dim * 2, :]
        # (hidden, in)
        recurrent_w = w[self.hidden_dim * 2:, :]
        # (hidden, hidden)
        recurrent_u = u[self.hidden_dim * 2:, :]
        # (batch, in,) dot (in, hidden * 2) + (batch, hidden,) dot (hidden, hidden * 2) -> (hidden * 2)
        pre_calc = T.dot(x_t, reset_update_w.T) + T.dot(h_t_1, reset_update_u.T)
        # (batch, hidden * 2) -> (batch, hidden) (batch, hidden)
        reset_t = self.reset_gate.activate(ndarray_slice(pre_calc, 0, self.hidden_dim))
        update_t = self.update_gate.activate(ndarray_slice(pre_calc, 1, self.hidden_dim))
        # (batch, in,) dot (in, hidden) + [(batch, hidden,) * (batch, hidden,)] dot (hidden, hidden)-> (hidden, )
        g_t = T.dot(x_t, recurrent_w.T) + T.dot(h_t_1 * reset_t, recurrent_u.T)
        # (batch, hidden,) * (batch, hidden,) + (batch, hidden,) * (batch, hidden,) -> (batch, hidden,)
        h_t = update_t * h_t_1 + (1 - update_t) * g_t
        # (batch, :) * (batch, hidden,) + (batch, :) * (batch, hidden,)
        h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_t_1
        # (batch, hidden)
        return h_t

    def forward_scan(self, x):
        h0 = shared_zero_matrix((self.hidden_dim,), 'h0_forward')
        hs, _ = theano.scan(fn=self._step,
                            sequences=x,
                            outputs_info=[h0],
                            non_sequences=[self.W, self.U],
                            )
        return hs

    def forward_scan_batch(self, x, mask):
        h0 = T.zeros((x.shape[0], self.hidden_dim))
        hs, _ = theano.scan(fn=self._step_batch,
                            sequences=[T.transpose(x, (1, 0, 2)),
                                       T.transpose(mask, (1, 0))],
                            outputs_info=[h0, ],
                            non_sequences=[self.W, self.U],
                            )
        return T.transpose(hs, (1, 0, 2))


class BiGRUEncoder(GRUEncoder):
    def __init__(self, in_dim, hidden_dim, pooling, activation='tanh', gates=("sigmoid", "sigmoid"),
                 prefix="", initializer=default_initializer, bidirection_shared=False, dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        super(BiGRUEncoder, self).__init__(in_dim, hidden_dim, pooling, activation, gates, prefix,
                                           initializer, dropout, verbose)
        self.out_dim = hidden_dim * 2
        # Composition Function Weight -- Gates
        if bidirection_shared:
            # W [reset, update, recurrent] (3 * hidden, in)
            self.W_forward, self.W_forward.name = self.W, prefix + "W_shared"
            self.W_backward = self.W_forward
            # U [reset, update, recurrent] (3 * hidden, in)
            self.U_forward, self.U_forward.name = self.U, prefix + "U_shared"
            self.U_backward = self.U_forward
            # b [reset, update, recurrent] (3 * hidden, in)
            # self.b_forward, self.b_forward.name = self.b, prefix + "b_shared"
            # self.b_backward = self.b_forward
            self.params = [self.W_forward, self.U_forward]  # , self.b_forward]
            self.norm_params = [self.W_forward, self.U_forward]
        else:
            # W [reset, update, recurrent] (3 * hidden, in)
            self.W_forward, self.W_forward.name = self.W, prefix + "W_forward"
            self.W_backward = shared_rand_matrix((self.hidden_dim * 3, self.in_dim),
                                                 prefix + 'W_backward', initializer)
            # U [reset, update, recurrent] (3 * hidden, in)
            self.U_forward, self.U_forward.name = self.U, prefix + "U_forward"
            self.U_backward = shared_rand_matrix((self.hidden_dim * 3, self.hidden_dim),
                                                 prefix + 'U_backward', initializer)
            # b [reset, update, recurrent] (3 * hidden, in)
            # self.b_forward, self.b_forward.name = self.b, prefix + "b_forward"
            # self.b_backward = shared_zero_matrix((self.hidden_dim * 3,), prefix + 'b_backward')
            self.params = [self.W_forward, self.U_forward,  # self.b_forward,
                           self.W_backward, self.U_backward]  # , self.b_backward]
            self.norm_params = [self.W_forward, self.U_forward, self.W_backward, self.U_backward]

        self.l1_norm = T.sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = T.sum([T.sum(param ** 2) for param in self.norm_params])

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            if bidirection_shared:
                logger.debug('%s' % "Forward/Backward Shared Parameter")
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Reset Gate:       %s' % self.reset_gate.method)
            logger.debug('Update Gate:      %s' % self.update_gate.method)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def backward_scan(self, x):
        h0_backward = shared_zero_matrix(self.hidden_dim, 'h0_backward')
        h_backwards, _ = theano.scan(fn=self._step,
                                     sequences=x,
                                     outputs_info=[h0_backward],
                                     non_sequences=[self.W_backward, self.U_backward],
                                     go_backwards=True,
                                     )
        return h_backwards[::-1]

    def backward_scan_batch(self, x, mask):
        h0_backward = T.zeros((x.shape[0], self.hidden_dim))
        h_backwards, _ = theano.scan(fn=self._step_batch,
                                     sequences=[T.transpose(x, (1, 0, 2)),
                                                T.transpose(mask, (1, 0))],
                                     outputs_info=[h0_backward],
                                     non_sequences=[self.W_backward, self.U_backward],
                                     go_backwards=True,
                                     )
        return T.transpose(h_backwards, (1, 0, 2))[:, ::-1]

    def forward_sequence(self, x):
        dropout_x = dropout_from_layer(x, self.dropout)
        return T.concatenate([self.forward_scan(dropout_x),
                              self.backward_scan(dropout_x),
                              ], axis=1)

    def forward_sequence_batch(self, x, mask):
        dropout_x = dropout_from_layer(x, self.dropout)
        return T.concatenate([self.forward_scan_batch(dropout_x, mask),
                              self.backward_scan_batch(dropout_x, mask),
                              ], axis=2)

    def forward(self, x):
        """
        :param x: (len, dim)
        """
        # Use Pooling to reduce into a fixed-length representation
        bi_hidden = self.forward_sequence(x)
        if self.pooling == 'last' or self.pooling == 'final':
            return T.concatenate([bi_hidden[-1, :self.hidden_dim],
                                  bi_hidden[0, self.hidden_dim:]
                                  ])
        else:
            return get_pooling(bi_hidden, self.pooling)

    def forward_batch(self, x, mask):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        """
        # Use Pooling to reduce into a fixed-length representation
        # (batch, max_len, dim) -> (batch, dim)
        bi_hidden = self.forward_sequence_batch(x, mask)
        if self.pooling == 'last' or self.pooling == 'final':
            return T.concatenate([bi_hidden[:, -1, :self.hidden_dim],
                                  bi_hidden[:, 0, self.hidden_dim:],
                                  ], axis=1)
        else:
            return get_pooling_batch(bi_hidden, mask, self.pooling)


class SGUEncoder(object):
    pass


class DSGUEncoder(object):
    pass


class BiRecurrentEncoder(RecurrentEncoder):
    def __init__(self, in_dim, hidden_dim, pooling, activation='tanh', prefix="",
                 initializer=default_initializer, dropout=0, bidirection_shared=False, verbose=True):
        super(BiRecurrentEncoder, self).__init__(in_dim, hidden_dim, pooling, activation, prefix,
                                                 initializer, dropout, verbose)
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        self.out_dim = hidden_dim * 2
        # Forward Direction - Backward Direction
        if bidirection_shared:
            # Feed-Forward Matrix (hidden, in)
            self.W_forward = self.W
            self.W_forward.name = prefix + "W_shared"
            self.W_backward = self.W_forward
            # Bias Term (hidden,)
            self.b_forward = self.b
            self.b_forward.name = prefix + "b_shared"
            self.b_backward = self.b_forward
            # Recurrent Matrix (hidden, hidden)
            self.U_forward = self.U
            self.U_forward.name = prefix + "U_shared"
            self.U_backward = self.U_forward

            self.params = [self.W_forward, self.U_forward, self.b_forward]
            self.norm_params = [self.W_forward, self.U_forward]
        else:
            # Feed-Forward Matrix (hidden, in)
            self.W_forward = self.W
            self.W_forward.name = prefix + "W_forward"
            self.W_backward = shared_rand_matrix((self.hidden_dim, self.in_dim), prefix + 'W_backward', initializer)
            # Bias Term (hidden,)
            self.b_forward = self.b
            self.b_forward.name = prefix + "b_forward"
            self.b_backward = shared_zero_matrix((self.hidden_dim,), prefix + 'b_backward')
            # Recurrent Matrix (hidden, hidden)
            self.U_forward = self.U
            self.U_forward.name = prefix + "U_forward"
            self.U_backward = shared_rand_matrix((self.hidden_dim, self.hidden_dim), prefix + 'U_backward', initializer)

            self.params = [self.W_forward, self.W_backward, self.U_forward, self.U_backward,
                           self.b_forward, self.b_backward]
            self.norm_params = [self.W_forward, self.W_backward, self.U_forward, self.U_backward]
        # L1, L2 Norm
        self.l1_norm = T.sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = T.sum([T.sum(param ** 2) for param in self.norm_params])

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def backward_scan(self, x):
        h0_backward = T.zeros((self.hidden_dim, ))
        h_backwards, _ = theano.scan(fn=self._step,
                                     sequences=x,
                                     outputs_info=[h0_backward],
                                     non_sequences=[self.W_backward, self.U_backward, self.b_backward],
                                     go_backwards=True,
                                     )
        return h_backwards[::-1]

    def backward_scan_batch(self, x, mask):
        h0_backward = T.zeros((x.shape[0], self.hidden_dim))
        h_backwards, _ = theano.scan(fn=self._step_batch,
                                     sequences=[T.transpose(x, (1, 0, 2)),
                                                T.transpose(mask, (1, 0))],
                                     outputs_info=[h0_backward],
                                     non_sequences=[self.W_backward, self.U_backward, self.b_backward],
                                     go_backwards=True,
                                     )
        return T.transpose(h_backwards, (1, 0, 2))[:, ::-1]

    def forward_sequence(self, x):
        dropout_x = dropout_from_layer(x, self.dropout)
        return T.concatenate([self.forward_scan(dropout_x),
                              self.backward_scan(dropout_x),
                              ], axis=1)

    def forward_sequence_batch(self, x, mask):
        dropout_x = dropout_from_layer(x, self.dropout)
        return T.concatenate([self.forward_scan_batch(dropout_x, mask),
                              self.backward_scan_batch(dropout_x, mask),
                              ], axis=2)

    def forward(self, x):
        """
        :param x: (len, dim)
        """
        # Use Pooling to reduce into a fixed-length representation
        bi_hidden = self.forward_sequence(x)
        if self.pooling == 'last' or self.pooling == 'final':
            return T.concatenate([bi_hidden[-1, :self.hidden_dim],
                                  bi_hidden[0, self.hidden_dim:]
                                  ])
        else:
            return get_pooling(bi_hidden, self.pooling)

    def forward_batch(self, x, mask):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        """
        # Use Pooling to reduce into a fixed-length representation
        # (batch, max_len, dim) -> (batch, dim)
        bi_hidden = self.forward_sequence_batch(x, mask)
        if self.pooling == 'last' or self.pooling == 'final':
            return T.concatenate([bi_hidden[:, -1, :self.hidden_dim],
                                  bi_hidden[:, 0, self.hidden_dim:],
                                  ], axis=1)
        else:
            return get_pooling_batch(bi_hidden, mask, self.pooling)


class BiLSTMEncoder(LSTMEncoder):
    def __init__(self, in_dim, hidden_dim, pooling, activation='tanh', gates=("sigmoid", "sigmoid", "sigmoid"),
                 prefix="", initializer=default_initializer, bidirection_shared=False, dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        super(BiLSTMEncoder, self).__init__(in_dim, hidden_dim, pooling, activation, gates, prefix,
                                            initializer, dropout, verbose)
        self.out_dim = hidden_dim * 2
        # Composition Function Weight -- Gates
        if bidirection_shared:
            # W [in, forget, output, recurrent]
            self.W_forward, self.W_forward.name = self.W, prefix + "W_shared"
            self.W_backward = self.W_forward
            # U [in, forget, output, recurrent]
            self.U_forward, self.U_forward.name = self.U, prefix + "U_shared"
            self.U_backward = self.U_forward
            # b [in, forget, output, recurrent]
            self.b_forward, self.b_forward.name = self.b, prefix + "b_shared"
            self.b_backward = self.b_forward

            self.params = [self.W_forward, self.U_forward, self.b_forward]
            self.norm_params = [self.W_forward, self.U_forward]
        else:
            # W [in, forget, output, recurrent]
            self.W_forward, self.W_forward.name = self.W, prefix + "W_forward"
            self.W_backward = shared_rand_matrix((self.hidden_dim * 4, self.in_dim),
                                                 prefix + 'W_backward', initializer)
            # U [in, forget, output, recurrent]

            self.U_forward, self.U_forward.name = self.U, prefix + "U_forward"
            self.U_backward = shared_rand_matrix((self.hidden_dim * 4, self.hidden_dim),
                                                 prefix + 'U_backward', initializer)
            # b [in, forget, output, recurrent]
            self.b_forward, self.b_forward.name = self.b, prefix + "b_forward"
            self.b_backward = shared_zero_matrix((self.hidden_dim * 4,), prefix + 'b_backward')
            self.params = [self.W_forward, self.U_forward, self.b_forward,
                           self.W_backward, self.U_backward, self.b_backward]
            self.norm_params = [self.W_forward, self.U_forward, self.W_backward, self.U_backward]

        self.l1_norm = T.sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = T.sum([T.sum(param ** 2) for param in self.norm_params])

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            if bidirection_shared:
                logger.debug('%s' % "Forward/Backward Shared Parameter")
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Input Gate:       %s' % self.in_gate.method)
            logger.debug('Forget Gate:      %s' % self.forget_gate.method)
            logger.debug('Output Gate:      %s' % self.out_gate.method)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def backward_scan(self, x):
        h0_backward = shared_zero_matrix(self.hidden_dim, 'h0_backward')
        c0_backward = shared_zero_matrix(self.hidden_dim, 'c0_backward')
        h_backwards, _ = theano.scan(fn=self._step,
                                     sequences=x,
                                     outputs_info=[h0_backward, c0_backward],
                                     non_sequences=[self.W_backward, self.U_backward, self.b_backward],
                                     go_backwards=True,
                                     )
        return h_backwards[0][::-1]

    def backward_scan_batch(self, x, mask):
        h0_backward = T.zeros((x.shape[0], self.hidden_dim))
        c0_backward = T.zeros((x.shape[0], self.hidden_dim))
        h_backwards, _ = theano.scan(fn=self._step_batch,
                                     sequences=[T.transpose(x, (1, 0, 2)),
                                                T.transpose(mask, (1, 0))],
                                     outputs_info=[h0_backward, c0_backward],
                                     non_sequences=[self.W_backward, self.U_backward, self.b_backward],
                                     go_backwards=True,
                                     )
        return T.transpose(h_backwards[0], (1, 0, 2))[:, ::-1]

    def forward_sequence(self, x):
        dropout_x = dropout_from_layer(x, self.dropout)
        return T.concatenate([self.forward_scan(dropout_x),
                              self.backward_scan(dropout_x),
                              ], axis=1)

    def forward_sequence_batch(self, x, mask):
        dropout_x = dropout_from_layer(x, self.dropout)
        return T.concatenate([self.forward_scan_batch(dropout_x, mask),
                              self.backward_scan_batch(dropout_x, mask),
                              ], axis=2)

    def forward(self, x):
        """
        :param x: (len, dim)
        """
        # Use Pooling to reduce into a fixed-length representation
        bi_hidden = self.forward_sequence(x)
        if self.pooling == 'last' or self.pooling == 'final':
            return T.concatenate([bi_hidden[-1, :self.hidden_dim],
                                  bi_hidden[0, self.hidden_dim:]
                                  ])
        else:
            return get_pooling(bi_hidden, self.pooling)

    def forward_batch(self, x, mask):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        """
        # Use Pooling to reduce into a fixed-length representation
        # (batch, max_len, dim) -> (batch, dim)
        bi_hidden = self.forward_sequence_batch(x, mask)
        if self.pooling == 'last' or self.pooling == 'final':
            return T.concatenate([bi_hidden[:, -1, :self.hidden_dim],
                                  bi_hidden[:, 0, self.hidden_dim:],
                                  ], axis=1)
        else:
            return get_pooling_batch(bi_hidden, mask, self.pooling)


class RecurrentNormEncoder(object):
    def __init__(self, in_dim, hidden_dim, pooling, activation, prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        self.in_dim = in_dim
        self.out_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.dropout = dropout
        self.act = Activation(activation)
        # Composition Function Weight
        # Feed-Forward Matrix (hidden, in)
        self.W = shared_rand_matrix((8, 8), prefix + 'W_forward', initializer)
        # Bias Term (hidden)
        self.b = shared_zero_matrix((8, 8), prefix + 'b_forward')
        # Recurrent Matrix (hidden, hidden)
        self.U = shared_rand_matrix((8, 8), prefix + 'U_forward', initializer)

        self.params = [self.W, self.U, self.b]
        self.norm_params = [self.W, self.U]

        # L1, L2 Norm
        self.l1_norm = T.sum(T.abs_(self.W)) + T.sum(T.abs_(self.U))
        self.l2_norm = T.sum(self.W ** 2 + self.U ** 2)

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def _step(self, x_t, h_t_1, w, u, b):
        """
        step function of forward
        :param x_t:   (in, )
        :param h_t_1: (hidden, )
        :param w:     (hidden, in)
        :param u:     (hidden, hidden)
        :param b:     (hidden, )
        :return:      (hidden)
        """
        # (hidden, in) (in, ) + (hidden, hidden) (hidden, ) + (hidden, ) -> hidden
        h_t = self.act.activate(T.dot(w, x_t) + T.dot(u, h_t_1) + b)
        return h_t

    def _step_batch(self, x_t, mask, h_t_1, w, u, b):
        """
        step function of forward in batch version
        :param x_t:   (batch, in)
        :param mask:  (batch, )
        :param h_t_1: (batch, hidden)
        :param w:     (hidden, in)
        :param u:     (hidden, hidden)
        :param b:     (hidden)
        :return:      (batch, hidden)
        """
        # (batch, in) (in, hidden) -> (batch, hidden)
        h_t_1 = T.reshape(h_t_1, (h_t_1.shape[0], 8, 8))
        x_t = T.reshape(x_t, (x_t.shape[0], 8, 8))
        x_t = x_t / x_t.norm(2, axis=1)[:, None, :]
        h_t = self.act.activate(T.dot(x_t, w.T) + T.dot(h_t_1, u.T) + b)
        h_t = h_t / h_t.norm(2, axis=1)[:, None, :]
        h_t_1 = T.reshape(h_t_1, (h_t_1.shape[0], 64))
        h_t = T.reshape(h_t, (h_t.shape[0], 64))
        # (batch, hidden) * (batch, None) + (batch, hidden) * (batch, None) -> (batch, hidden)
        return h_t * mask[:, None] + h_t_1 * (1 - mask[:, None])

    def forward_sequence(self, x):
        h0 = shared_zero_matrix((self.hidden_dim,), 'h0')
        hs, _ = theano.scan(fn=self._step,
                            sequences=x,
                            outputs_info=[h0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        return hs

    def forward_sequence_batch(self, x, mask, batch_size):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        :param batch_size:
        """
        h0 = shared_zero_matrix((batch_size, self.hidden_dim), 'h0')
        hs, _ = theano.scan(fn=self._step_batch,
                            sequences=[T.transpose(x, (1, 0, 2)),  # (batch, max_len, dim) -> (max_len, batch, dim)
                                       T.transpose(mask, (1, 0))],     # (batch, max_len) -> (max_len, batch)
                            outputs_info=[h0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        # (max_len, batch, dim) -> (batch, max_len, dim)
        return T.transpose(hs, (1, 0, 2))

    def forward(self, x):
        """
        :param x: (len, dim)
        """
        # Use Pooling to reduce into a fixed-length representation
        return get_pooling(self.forward_sequence(x), self.pooling)

    def forward_batch(self, x, mask, batch_size):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        :param batch_size:
        """
        # Use Pooling to reduce into a fixed-length representation
        # (max_len, batch, dim) -> (batch, max_len, dim) -> (batch, dim)
        hidden = self.forward_sequence_batch(x, mask, batch_size)
        return get_pooling_batch(hidden, mask, self.pooling)


def test():
    import numpy as np
    np.random.seed(0)
    test_times = 50
    for i in xrange(test_times):
        batch = np.random.random_integers(2, 50)
        print "Batch:", batch
        length = 6
        in_dim = 5
        hidden = 4
        a = np.random.uniform(size=(batch, length, in_dim))
        a = a.astype(theano.config.floatX)
        x = T.matrix()
        x_batch = T.tensor3()
        mask = np.zeros(shape=(batch, length), dtype=theano.config.floatX)
        mask[0, :] = 1
        for j in xrange(1, batch):
            mask[j, :np.random.randint(1, length)] += 1
        print np.sum(mask, axis=1)
        mask_shared = shared_matrix(mask)
        a = a * mask[:, :, None]
        pooling_method_list = ['max', 'mean', 'sum', 'min', 'final']
        pooling_method = np.random.choice(pooling_method_list)
        print pooling_method
        convlayer = BiGRUEncoder(in_dim, hidden, pooling=pooling_method, dropout=0)

        x_c = convlayer.forward(x)
        test_f = theano.function([x], x_c)
        x_c_batch = convlayer.forward_batch(x_batch, mask_shared)
        test_f_batch = theano.function([x_batch], x_c_batch)

        result_list = list()
        for v, m in zip(a, mask):
            c = v[:int(np.sum(m))]
            result_list.append(test_f(c))
        result_list = np.array(result_list, dtype=theano.config.floatX)
        b = test_f_batch(a)
        error = np.sum(result_list - b)
        print error
        if error > 0.00001:
            print "Error"
            exit(1)
    print "Test Pass !!!"


if __name__ == "__main__":
    test()
