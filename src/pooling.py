import logging

import theano.tensor as T

from __init__ import BIG_INT

__author__ = 'roger'
logger = logging.getLogger(__name__)


def get_pooling(hs, pooling_method):
    if pooling_method == 'max':
        return T.max(hs, axis=0)
    elif pooling_method == 'min':
        return T.min(hs, axis=0)
    elif pooling_method in ['averaging', 'mean' , 'average']:
        return T.mean(hs, axis=0)
    elif pooling_method == 'sum':
        return T.sum(hs, axis=0)
    elif pooling_method in ['final', 'last']:
        return hs[-1]
    else:
        raise NotImplementedError('Not implemented pooling method: {}'.format(pooling_method))


def get_pooling_batch(hs, mask, pooling_method):
    """
    :param hs:   (batch, len, dim)
    :param mask: (batch, len)
    :param pooling_method:
    :return:
    """
    if pooling_method == 'max':
        add_v = ((1 - mask) * -BIG_INT)[:, :, None]
        return T.max(hs + add_v, axis=1)
    elif pooling_method == 'min':
        add_v = ((1 - mask) * BIG_INT)[:, :, None]
        return T.min(hs + add_v, axis=1)
    elif pooling_method in ['averaging', 'mean' , 'average']:
        return T.sum(hs * mask[:, :, None], axis=1) / T.sum(mask, axis=1)[:, None]
    elif pooling_method == 'sum':
        return T.sum(hs * mask[:, :, None], axis=1)
    elif pooling_method in ['final', 'last']:
        return hs[:, -1, :]
    else:
        raise NotImplementedError('Not implemented pooling method: {}'.format(pooling_method))


class PoolingLayer(object):
    def __init__(self, pooling, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        self.pooling = pooling
        # Composition Function Weight
        self.params = []
        self.norm_params = []

        # L1, L2 Norm
        self.l1_norm = 0
        self.l2_norm = 0

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Pooling methods:  %s' % self.pooling)

    def forward(self, x):
        return get_pooling(x, self.pooling)

    def forward_batch(self, x, mask):
        return get_pooling_batch(x, mask, self.pooling)


class MaxPoolingLayer(PoolingLayer):
    def __init__(self, verbose=True):
        super(MaxPoolingLayer, self).__init__(pooling='max', verbose=verbose)


class MeanPoolingLayer(PoolingLayer):
    def __init__(self, verbose=True):
        super(MeanPoolingLayer, self).__init__(pooling='mean', verbose=verbose)


class MinPoolingLayer(PoolingLayer):
    def __init__(self, verbose=True):
        super(MinPoolingLayer, self).__init__(pooling='min', verbose=verbose)


class CBOWLayer(PoolingLayer):
    def __init__(self, in_dim, pooling='mean', verbose=True):
        super(CBOWLayer, self).__init__(pooling=pooling, verbose=verbose)
        self.in_dim = in_dim
        self.out_dim = in_dim
