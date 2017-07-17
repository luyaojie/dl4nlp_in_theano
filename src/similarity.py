import theano.tensor as T

__author__ = 'roger'


def cosine_similarity_batch(x, y):
    """
    :param x: (batch, dim)
    :param y: (batch, dim)
    :return:
    """
    norm_x = T.sqrt(T.sum(T.sqr(x), axis=1))         # (batch, )
    norm_y = T.sqrt(T.sum(T.sqr(y), axis=1))         # (batch, )
    return T.sum(x * y, axis=1) / (norm_x * norm_y)  # (batch, )


def cosine_similarity(x, y):
    """
    :param x: (batch, dim)
    :param y: (batch, dim)
    :return:
    """
    norm_x = T.sqrt(T.sum(T.sqr(x)))         # (batch, )
    norm_y = T.sqrt(T.sum(T.sqr(y)))         # (batch, )
    return T.sum(x * y) / (norm_x * norm_y)  # (batch, )
