# -*- coding: utf-8 -*-
# Mostly from Blocks and Lasagne and Keras
# https://github.com/mila-udem/blocks/blob/master/blocks/initialization.py
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
from abc import ABCMeta, abstractmethod

import numpy as np
from six import add_metaclass

__author__ = 'roger'


@add_metaclass(ABCMeta)
class Initializer(object):
    @abstractmethod
    def generate(self, shape):
        """
        Generate an initial set of parameters from a given distribution.
        :param shape:
        :return:
        """


class NormalInitializer(Initializer):
    """
    mean : float
        Mean ("centre") of the distribution.
    scale : float
        Standard deviation (spread or "width") of the distribution.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __str__(self):
        return "Normal (mean=%f, std=%f)" % (self.mean, self.std)

    def generate(self, shape):
        """

        :param shape:
        :return:
        """
        m = np.random.normal(loc=self.mean, scale=self.std, size=shape)
        return as_floatx(m)


class UniformInitializer(Initializer):
    def __init__(self, scale):
        self.scale = scale

    def __str__(self):
        return "Uniform (scale=%f)" % self.scale

    def generate(self, shape):
        """
        :param shape:
        :return:
        """
        m = np.random.uniform(low=-self.scale, high=self.scale, size=shape)
        return as_floatx(m)


class GlorotUniformInitializer(Initializer):
    def __str__(self):
        return "GlorotUniform sqrt(6.0 / (in + out))"

    def generate(self, shape):
        """
        :param shape:
        :return:
        """
        if type(shape) == int or len(shape) == 1:
            scale = 0.001
        else:
            fan_in = shape[0]
            fan_out = shape[1]
            scale = np.sqrt(6. / (fan_in + fan_out))
        m = np.random.uniform(low=-scale, high=scale, size=shape)
        return as_floatx(m)


class IdentityInitializer(Initializer):
    def __init__(self, mul):
        self.mul = mul

    def __str__(self):
        return "Identity with %f" % self.mul

    def generate(self, shape):
        if len(shape) != 2:
            raise ValueError
        rows, cols = shape
        return as_floatx(self.mult * np.eye(rows, cols))


class OrthogonalInitializer(Initializer):
    # From lasagne
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)
        self.gain = gain

    def __str__(self):
        return "Orthogonal (%f)" % self.gain

    def generate(self, shape):
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are "
                               "supported.")

        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return as_floatx(self.gain * q)


# Variable Operation
def as_floatx(variable):
    import theano
    import theano.tensor as T
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)
    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return T.cast(variable, theano.config.floatX)