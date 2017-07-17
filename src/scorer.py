import abc
import logging

import theano
import theano.tensor as T

from __init__ import default_initializer
from activations import Activation
from utils import shared_rand_matrix, shared_zero_matrix, shared_ones_matrix

logger = logging.getLogger(__name__)
__author__ = 'roger'


class EntityScorer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self): pass

    @abc.abstractmethod
    def score(self, entity1, entity2, relation): pass

    @abc.abstractmethod
    def score_batch(self, entity1, entity2, relation): pass

    @abc.abstractmethod
    def score_one_relation(self, entity1, entity2, relation): pass


class SingleLayerModel(EntityScorer):
    def __init__(self, entity_dim, relation_num, hidden=50, activation='tanh',
                 initializer=default_initializer, prefix='', verbose=True):
        super(SingleLayerModel, self).__init__()
        self.hidden = hidden
        self.entity_dim = entity_dim
        self.relation_num = relation_num
        # (relation_num, k, entity_dim)
        self.W_1 = shared_rand_matrix((relation_num, self.hidden, self.entity_dim),
                                      prefix + 'SingleLayer_W1', initializer)
        # (relation_num, k, entity_dim)
        self.W_2 = shared_rand_matrix((relation_num, self.hidden, self.entity_dim),
                                      prefix + 'SingleLayer_W2', initializer)
        # (relation_num, k, )
        self.u = shared_ones_matrix((relation_num, self.hidden,), prefix + 'SingleLayer_u')
        self.act = Activation(activation)
        self.params = [self.W_1, self.W_2, self.u]
        self.norm_params = [self.W_1, self.W_2, self.u]
        self.l1_norm = T.sum(T.abs_(self.W_1)) + T.sum(T.abs_(self.W_2)) + T.sum(T.abs_(self.u))
        self.l2_norm = T.sum(self.W_1 ** 2) + T.sum(self.W_2 ** 2) + T.sum(self.u ** 2)

        if verbose:
            logger.debug('Architecture of Single Layer Model built finished, summarized as below:')
            logger.debug('Entity Dimension: %d' % self.entity_dim)
            logger.debug('Hidden Dimension: %d' % self.hidden)
            logger.debug('Relation Number:  %d' % self.relation_num)
            logger.debug('Initializer:      %s' % initializer)
            logger.debug('Activation:       %s' % activation)

    def score(self, e1, e2, r_index):
        """
        :param e1: (entity_dim, )
        :param e2: (entity_dim, )
        :param r_index: scalar
        :return: 
        """
        # (hidden, entity_dim) dot (entity_dim) + (hidden, entity_dim) dot (entity_dim) -> (hidden, )
        hidden = T.dot(self.W_1[r_index], e1) + T.dot(self.W_2[r_index], e2)
        # (hidden, ) -> (hidden, )
        act_hidden = self.act.activate(hidden)
        # (hidden, ) dot (hidden, ) -> 1
        return T.dot(self.u[r_index], act_hidden)

    def score_batch(self, e1, e2, r_index):
        """
        :param e1: (batch, entity_dim, )
        :param e2: (batch, entity_dim, )
        :param r_index: (batch, )
        :return: 
        """
        # (batch, hidden, entity_dim) dot (batch, entity_dim) + (batch, hidden, entity_dim) dot (batch, entity_dim)
        hidden = T.batched_dot(self.W_1[r_index], e1)
        hidden += T.batched_dot(self.W_2[r_index], e2)
        # (batch, hidden) -> (batch, hidden)
        act_hidden = self.act.activate(hidden)
        # (batch, hidden) dot (batch, hidden) -> (batch, )
        return T.sum(act_hidden * self.u[r_index], axis=1)

    def score_one_relation(self, e1, e2, r_index):
        """
        :param e1: (batch, entity_dim, )
        :param e2: (batch, entity_dim, )
        :param r_index: scalar
        :return: 
        """
        # (batch, entity_dim) dot (entity_dim, hidden) + (batch, entity_dim) dot (entity_dim, hidden) -> (batch, hidden)
        hidden = T.dot(e1, self.W_1[r_index].transpose()) + T.dot(e2, self.W_2[r_index].transpose())
        # (batch, hidden) -> (batch, hidden)
        act_hidden = self.act.activate(hidden)
        # (batch, hidden) dot (hidden, ) -> (batch, )
        return T.dot(act_hidden, self.u[r_index])


class BilinearModel(EntityScorer):
    def __init__(self, entity_dim, relation_num, activation='iden',
                 initializer=default_initializer, prefix='', verbose=True):
        super(BilinearModel, self).__init__()
        self.entity_dim = entity_dim
        self.relation_num = relation_num
        # (relation_num, entity_dim, entity_dim)
        self.W = shared_rand_matrix((relation_num, self.entity_dim, self.entity_dim),
                                    prefix + 'Bilinear_W', initializer)
        self.act = Activation(activation)
        self.params = [self.W]
        self.norm_params = [self.W]
        self.l1_norm = T.sum(T.abs_(self.W))
        self.l2_norm = T.sum(self.W ** 2)

        if verbose:
            logger.debug('Architecture of Bilinear Model built finished, summarized as below:')
            logger.debug('Entity Dimension: %d' % self.entity_dim)
            logger.debug('Relation Number:  %d' % self.relation_num)
            logger.debug('Initializer:      %s' % initializer)
            logger.debug('Activation:       %s' % activation)

    def score(self, e1, e2, r_index):
        """
        :param e1: (entity_dim, )
        :param e2: (entity_dim, )
        :param r_index: scalar
        :return: 
        """
        # (entity_dim, ) dot (entity_dim, entity_dim) dot (entity_dim, ) -> scalar
        hidden = T.dot(e1, T.dot(self.W[r_index], e2))
        return self.act.activate(hidden)

    def score_batch(self, e1, e2, r_index):
        """
        :param e1: (batch, entity_dim, )
        :param e2: (batch, entity_dim, )
        :param r_index: (batch, )
        :return: 
        """
        # (batch, entity_dim, ) dot (batch, entity_dim, entity_dim) -> (batch, entity_dim)
        hidden = T.batched_dot(e1, self.W[r_index])
        # (batch, entity_dim) dot (batch, entity_dim, ) -> (batch, )
        hidden = T.sum(hidden * e2, axis=1)
        return self.act.activate(hidden)

    def score_one_relation(self, e1, e2, r_index):
        """
        :param e1: (batch, entity_dim, )
        :param e2: (batch, entity_dim, )
        :param r_index: scalar
        :return: 
        """
        # (batch, entity_dim, ) dot (entity_dim, entity_dim) -> (batch, entity_dim)
        hidden = T.dot(e1, self.W[r_index])
        # (batch, entity_dim) dot (batch, entity_dim, ) -> (batch, )
        hidden = T.sum(hidden * e2, axis=1)
        return self.act.activate(hidden)


class NeuralTensorModel(EntityScorer):
    def __init__(self, entity_dim, relation_num, activation='tanh', hidden=5, keep_normal=False,
                 initializer=default_initializer, prefix='', verbose=True):
        super(NeuralTensorModel, self).__init__()
        self.entity_dim = entity_dim
        self.relation_num = relation_num
        self.hidden = hidden
        self.slice_seq = T.arange(hidden)
        self.keep_normal = keep_normal
        # (relation_num, entity_dim, entity_dim, hidden)
        self.W = shared_rand_matrix((relation_num, self.entity_dim, self.entity_dim, self.hidden),
                                    prefix + 'NTN_W', initializer)
        # (relation_num, hidden)
        self.U = shared_ones_matrix((relation_num, self.hidden), name=prefix + 'NTN_U')
        if keep_normal:
            # (relation_num, entity_dim, hidden)
            self.V = shared_rand_matrix((relation_num, self.entity_dim * 2, self.hidden), prefix + 'NTN_V', initializer)
            # (relation_num, hidden)
            self.b = shared_zero_matrix((relation_num, self.hidden), name=prefix + 'NTN_B')
            self.params = [self.W, self.V, self.U, self.b]
            self.norm_params = [self.W, self.V, self.U, self.b]
        else:
            self.params = [self.W]
            self.norm_params = [self.W]
        self.act = Activation(activation)
        self.l1_norm = T.sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = T.sum([T.sum(param ** 2) for param in self.norm_params])

        if verbose:
            logger.debug('Architecture of Tensor Model built finished, summarized as below:')
            logger.debug('Entity Dimension: %d' % self.entity_dim)
            logger.debug('Hidden Dimension: %d' % self.hidden)
            logger.debug('Relation Number:  %d' % self.relation_num)
            logger.debug('Initializer:      %s' % initializer)
            logger.debug('Activation:       %s' % activation)

    @staticmethod
    def step(_slice, e1, e2, w):
        """
        :param _slice: scalar
        :param e1: (entity_dim, )
        :param e2: (entity_dim, )
        :param w : (entity_dim, entity_dim, hidden)
        :return: 
        """
        # (entity_dim, ) dot (entity_dim, entity_dim) dot (entiy_dim) -> scalar
        return T.dot(e1, T.dot(w[_slice], e2))

    @staticmethod
    def step_relation(_slice, e1, e2, w):
        """
        :param _slice: scalar
        :param e1: (batch, entity_dim)
        :param e2: (batch, entity_dim)
        :param w : (entity_dim, entity_dim, hidden)
        :return: 
        """
        # (batch, entity_dim, ) dot (entity_dim, entity_dim) -> (batch, entity_dim)
        hidden = T.dot(e1, w[:, :, _slice])
        # (batch, entity_dim) dot (batch, entity_dim, ) -> (batch, )
        hidden = T.sum(hidden * e2, axis=1)
        return hidden

    @staticmethod
    def step_batch(_slice, e1, e2, w):
        """
        :param _slice: scalar
        :param e1: (batch, entity_dim)
        :param e2: (batch, entity_dim)
        :param w : (batch, entity_dim, entity_dim, hidden)
        :return: 
        """
        # (batch, entity_dim, ) dot (batch, entity_dim, entity_dim) -> (batch, entity_dim)
        hidden = T.batched_dot(e1, w[:, :, :, _slice])
        # (batch, entity_dim) dot (batch, entity_dim, ) -> (batch, )
        hidden = T.sum(hidden * e2, axis=1)
        return hidden

    def score(self, e1, e2, r_index):
        """
        :param e1: (entity_dim, )
        :param e2: (entity_dim, )
        :param r_index: scalar
        :return: 
        """
        # (entity_dim, ) dot (entity_dim, entity_dim, hidden) dot (entity_dim, ) -> (hidden, )
        hidden1_sep, _ = theano.scan(fn=self.step,
                                     sequences=[self.slice_seq],
                                     non_sequences=[e1, e2, self.W[r_index]],
                                     name='single_scan')
        hidden1 = T.concatenate([hidden1_sep])
        if self.keep_normal:
            # (2 * entity_dim, ) dot (2 * entity_dim, hidden) -> (hidden, )
            hidden2 = T.dot(T.concatenate([e1, e2]), self.V[r_index])
            # (hidden, ) + (hidden, ) + (hidden, ) -> (hidden, )
            hidden = hidden1 + hidden2 + self.b[r_index]
        else:
            hidden = hidden1
        # (hidden, ) -> (hidden, )
        act_hidden = self.act.activate(hidden)
        # (hidden, ) dot (hidden, ) -> scalar
        return T.dot(act_hidden, self.U[r_index])

    def score_batch(self, e1, e2, r_index):
        """
        :param e1: (batch, entity_dim, )
        :param e2: (batch, entity_dim, )
        :param r_index: (batch, )
        :return: 
        """
        # (batch, entity_dim) dot (batch, entity_dim, entity_dim, hidden) dot (batch, entity_dim) -> hidden * (batch, )
        hidden1_sep, _ = theano.scan(fn=self.step_batch,
                                     sequences=[self.slice_seq],
                                     non_sequences=[e1, e2, self.W[r_index]],
                                     name='batch_scan')
        # hidden * (batch, ) -> (batch, hidden)
        hidden1 = T.concatenate([hidden1_sep], axis=1).transpose()
        if self.keep_normal:
            # (batch, 2 * entity_dim) dot (batch, 2 * entity_dim, hidden) -> (batch, hidden, )
            hidden2 = T.batched_dot(T.concatenate([e1, e2], axis=1), self.V[r_index])
            # (batch, hidden) + (batch, hidden) + (batch, hidden) -> (batch, hidden)
            hidden = hidden1 + hidden2 + self.b[r_index]
        else:
            hidden = hidden1
        # (batch, hidden) -> (batch, hidden)
        act_hidden = self.act.activate(hidden)
        # (batch, hidden) dot (batch, hidden) -> (batch, )
        return T.sum(act_hidden * self.U[r_index], axis=1)

    def score_one_relation(self, e1, e2, r_index):
        """
        :param e1: (batch, entity_dim, )
        :param e2: (batch, entity_dim, )
        :param r_index: scalar
        :return: 
        """
        # (batch, entity_dim) dot (entity_dim, entity_dim, hidden) dot (batch, entity_dim) -> hidden * (batch, )
        hidden1_sep, _ = theano.scan(fn=self.step_relation,
                                     sequences=self.slice_seq,
                                     non_sequences=[e1, e2, self.W[r_index]],
                                     name='relation_scan')
        # hidden * (batch, ) -> (batch, hidden)
        hidden1 = T.concatenate([hidden1_sep], axis=1).transpose()
        if self.keep_normal:
            # (batch, 2 * entity_dim) dot (2 * entity_dim, hidden) -> (batch, hidden)
            hidden2 = T.dot(T.concatenate([e1, e2], axis=1), self.V[r_index])
            # (batch, hidden) + (batch, hidden) + (hidden) -> (batch, hidden)
            hidden = hidden1 + hidden2 + self.b[r_index][None, :]
        else:
            hidden = hidden1
        # (batch, hidden) -> (batch, hidden)
        act_hidden = self.act.activate(hidden)
        # (batch, hidden) dot (batch, hidden) -> (batch, )
        return T.sum(act_hidden * self.U[r_index], axis=1)


class TransEModel(EntityScorer):
    def __init__(self, entity_dim, relation_num, activation='iden',
                 initializer=default_initializer, prefix='', verbose=True):
        super(TransEModel, self).__init__()
        self.entity_dim = entity_dim
        self.relation_num = relation_num
        # (relation_num, entity_dim, entity_dim)
        self.W = shared_rand_matrix((relation_num, self.entity_dim),
                                    prefix + 'TransE_R', initializer)
        self.act = Activation(activation)
        self.params = [self.W]
        self.norm_params = [self.W]
        self.l1_norm = T.sum(T.abs_(self.W))
        self.l2_norm = T.sum(self.W ** 2)

        if verbose:
            logger.debug('Architecture of TransE Model built finished, summarized as below:')
            logger.debug('Entity Dimension: %d' % self.entity_dim)
            logger.debug('Relation Number:  %d' % self.relation_num)
            logger.debug('Initializer:      %s' % initializer)
            logger.debug('Activation:       %s' % activation)

    def score(self, e1, e2, r_index):
        """
        :param e1: (entity_dim, )
        :param e2: (entity_dim, )
        :param r_index: scalar
        :return: 
        """
        # (entity_dim, ) + (entity_dim, ) - (entity_dim, ) -> (entity_dim, )
        hidden = e1 + self.W[r_index] - e2
        # (entity_dim, ) -> scalar
        d = T.sum(hidden ** 2)
        return self.act.activate(d)

    def score_batch(self, e1, e2, r_index):
        """
        :param e1: (batch, entity_dim, )
        :param e2: (batch, entity_dim, )
        :param r_index: (batch, )
        :return: 
        """
        # (batch, entity_dim, ) + (batch, entity_dim, ) - (batch, entity_dim, ) -> (batch, entity_dim, )
        hidden = e1 + self.W[r_index] - e2
        d = T.sum(hidden ** 2, axis=1)
        return self.act.activate(d)

    def score_one_relation(self, e1, e2, r_index):
        """
        :param e1: (batch, entity_dim, )
        :param e2: (batch, entity_dim, )
        :param r_index: scalar
        :return: 
        """
        # (batch, entity_dim, ) + (batch, entity_dim, ) - (batch, entity_dim, ) -> (batch, entity_dim, )
        hidden = e1 + self.W[r_index][None, :] - e2
        d = T.sum(hidden ** 2, axis=1)
        return self.act.activate(d)
