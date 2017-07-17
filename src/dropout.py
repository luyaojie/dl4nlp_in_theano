from theano.tensor.shared_randomstreams import RandomStreams
import theano
import numpy as np
from utils import as_floatx


dropout_on = theano.shared(as_floatx(1.0), borrow=True, )


def dropout_from_layer(x, p):
    """
    :param x: input
    :param p: the probablity of dropping a unit
    """
    srng = RandomStreams(np.random.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1.0 - p, size=x.shape)
    off_gain = 1.0 - p
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    return x * as_floatx(mask) * dropout_on + (1 - dropout_on) * x * off_gain


def set_dropout_on(training):
    flag = 1.0 if training else 0.0
    dropout_on.set_value(flag)
