import theano.tensor as T

from utils import shared_scalar

__author__ = 'roger'


def iden(x):
    return x


class Activation(object):
    def __init__(self, method):
        att = method.split("_")
        if len(att) > 1:
            self.alpha = shared_scalar(float(att[1]))
        else:
            self.alpha = shared_scalar(0)
        self.method = method
        method_name = att[0].lower()
        self.method_name = method_name
        if method_name == "sigmoid":
            self.func = T.nnet.sigmoid
        elif method_name == "tanh":
            self.func = T.tanh
        elif method_name == "relu":
            self.func = T.nnet.relu
        elif method_name == "elu":
            self.func = T.nnet.elu
        elif method_name == 'iden':
            self.func = iden
        else:
            raise ValueError('Invalid Activation function %s !' % method)

    def activate(self, x):
        if self.method[:3] == "elu":
            return T.nnet.elu(x, alpha=self.alpha)
        elif self.method[:4] == "relu":
            return T.nnet.relu(x, alpha=self.alpha)
        else:
            return self.func(x)
