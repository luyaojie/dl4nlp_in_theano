import logging

import theano.tensor as T
from theano.ifelse import ifelse

from Initializer import GlorotUniformInitializer
from activations import Activation
from dropout import dropout_from_layer
from pooling import get_pooling, get_pooling_batch
from utils import shared_rand_matrix, shared_zero_matrix

logger = logging.getLogger(__name__)


def temporal_padding_2d(x, padding=(1, 1)):
    """Pad the middle dimension of a 2D matrix
    with "padding" zeros left and right.

    Apologies for the inane API, but Theano makes this
    really hard.
    Code from https://github.com/fchollet/keras/blob/master/keras/backend/theano_backend.py
    x: (length, dim)
    """
    assert len(padding) == 2
    input_shape = x.shape
    output_shape = (input_shape[0] + padding[0] + padding[1],
                    input_shape[1])
    output = T.zeros(output_shape)
    result = T.set_subtensor(output[padding[0]:x.shape[0] + padding[0], :], x)
    return result


def temporal_padding_mask(mask, kernel_size, padding_size):
    """Pad the middle dimension of a 2D matrix
    with "padding" zeros left and right.

    Apologies for the inane API, but Theano makes this
    really hard.
    Code from https://github.com/fchollet/keras/blob/master/keras/backend/theano_backend.py
    x: (batch, length)
    """
    mask_shape = mask.shape
    mask_sum = T.sum(mask, axis=1)
    output_length = mask_sum - kernel_size + 2 * padding_size + 1
    max_output_length = mask_shape[1] - kernel_size + 2 * padding_size + 1
    real_output_length = T.maximum(output_length, 1)
    range_base = T.arange(max_output_length)
    range_matrix = T.outer(T.ones((mask_shape[0],)), range_base)
    mask = (range_matrix < real_output_length[:, None]) * T.constant(1.0)
    return mask


def temporal_padding_3d(x, padding=(1, 1)):
    """Pad the middle dimension of a 3D tensor
    with "padding" zeros left and right.

    Apologies for the inane API, but Theano makes this
    really hard.
    Code from https://github.com/fchollet/keras/blob/master/keras/backend/theano_backend.py
    """
    assert len(padding) == 2
    input_shape = x.shape
    output_shape = (input_shape[0],
                    input_shape[1] + padding[0] + padding[1],
                    input_shape[2])
    output = T.zeros(output_shape)
    result = T.set_subtensor(output[:, padding[0]:x.shape[1] + padding[0], :], x)
    return result


class ConvolutionLayer(object):
    def __init__(self, in_dim, hidden_dim, kernel_size=3, padding='same', pooling='max', dilation_rate=1.0,
                 activation='relu', prefix="", initializer=GlorotUniformInitializer(), dropout=0.0, verbose=True):
        """
        Init Function for ConvolutionLayer
        :param in_dim:
        :param hidden_dim:
        :param kernel_size:
        :param padding: 'same', 'valid'
        :param pooling: 'max', 'mean', 'min'
        :param dilation_rate:
        :param activation:
        :param prefix:
        :param initializer:
        :param dropout:
        :param verbose:
        """
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))

        self.in_dim = in_dim
        self.out_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.pooling = pooling
        self.dropout = dropout
        self.act = Activation(activation)
        self.padding_size = int(self.dilation_rate * (self.kernel_size - 1))
        # Composition Function Weight
        # Kernel Matrix (kernel_size, hidden, in)
        self.W = shared_rand_matrix((self.kernel_size, self.hidden_dim, self.in_dim), prefix + 'W', initializer)
        # Bias Term (hidden)
        self.b = shared_zero_matrix((self.hidden_dim,), prefix + 'b')

        self.params = [self.W, self.b]
        self.norm_params = [self.W]

        # L1, L2 Norm
        self.l1_norm = T.sum(T.abs_(self.W))
        self.l2_norm = T.sum(self.W ** 2)

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Filter Num  (Hidden): %d' % self.hidden_dim)
            logger.debug('Kernel Size (Windows): %d' % self.kernel_size)
            logger.debug('Padding method :  %s' % self.padding)
            logger.debug('Dilation Rate  :  %s' % self.dilation_rate)
            logger.debug('Padding Size   :  %s' % self.padding_size)
            logger.debug('Pooling method :  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def forward_conv(self, x):
        """
        #TODO
        :param x: (length, dim)
        :return:  (length+2*(kernel-1, hidden_dim)
        """
        # T.nn.conv2d (batch size, input channels, input rows, input columns)
        # dl4nlp      (batch size, 1,              length,     in_dim)
        x = x.dimshuffle(['x', 'x', 0, 1])
        # T.nn.conv2d (output channels, input channels, filter rows, filter columns)
        # dl4nlp      (hidden_dim,      1,              kernel_size, in_dim)
        filter_w = self.W.dimshuffle([1, 'x', 0, 2])
        # T.nn.conv2d (batch size, output channels, output rows,     output columns)
        # dl4nlp      (batch size, hidden_dim,      length+kernel-1, 1)
        conv_result = T.nnet.conv2d(x, filter_w,
                                    border_mode='valid',)
        # (batch size, hidden_dim, length+kernel-1, 1) -> (length+kernel-1, hidden_dim)
        conv_result = T.transpose(conv_result[0, :, :, 0], (1, 0))
        return conv_result

    def forward_conv_batch(self, x):
        """
        :param x: (batch, length, dim)
        :return:  (batch, length - kernel + 2*padding_size + 1, hidden_dim)
        """
        # T.nn.conv2d (batch size, input channels, input rows, input columns)
        # dl4nlp      (batch size, 1,              length,     in_dim)
        x = x.dimshuffle([0, 'x', 1, 2])
        # T.nn.conv2d (output channels, input channels, filter rows, filter columns)
        # dl4nlp      (hidden_dim,      1,              kernel_size, in_dim)
        filter_w = self.W.dimshuffle([1, 'x', 0, 2])
        # T.nn.conv2d (batch size, output channels, output rows,     output columns)
        # dl4nlp      (batch size, hidden_dim,      length+kernel-1, 1)
        conv_result = T.nnet.conv2d(x, filter_w,
                                    border_mode='valid',)
        # from theano.printing import Print
        # conv_result = Print()(conv_result)
        # (batch size, hidden_dim, length - kernel + 2*padding_size + 1, 1)
        #   -> (batch, length - kernel + 2*padding_size + 1, hidden_dim)
        conv_result = T.transpose(conv_result[:, :, :, 0], (0, 2, 1))
        return conv_result

    def forward(self, x):
        """
        :param x: (length, dim)
        :return: (hidden_dim, )
        """
        if self.padding_size > 0:
            # (padding_size + length + padding_size, dim)
            x = temporal_padding_2d(x, (self.padding_size, self.padding_size))
        safe_x = temporal_padding_2d(x, (0, self.kernel_size - x.shape[0]))
        # If Kernel Size is greater than sentence length, padding at the end of sentence
        x = ifelse(T.gt(self.kernel_size - x.shape[0], 0),
                   safe_x,
                   x)
        conv_result = self.forward_conv(x)
        pooling_result = get_pooling(conv_result, self.pooling)
        dropout_out = dropout_from_layer(pooling_result, self.dropout)
        return self.act.activate(dropout_out + self.b)

    def forward_batch(self, x, mask):
        """
        :param x: (batch, length, dim)
        :param mask: (batch, length, )
        :return: (batch, length, hidden_dim)
        """
        # conv_after_length = length - kernel + 2 * padding_size + 1
        new_x = x
        if self.padding_size > 0:
            # (padding_size + length + padding_size, dim)
            new_x = temporal_padding_3d(x, (self.padding_size, self.padding_size))
            # (batch, conv_after_length)
            mask = temporal_padding_mask(mask, kernel_size=self.kernel_size, padding_size=self.padding_size)
        elif self.padding_size == 0:
            # (batch, conv_after_length)
            mask = temporal_padding_mask(mask, kernel_size=self.kernel_size, padding_size=0)
        else:
            raise RuntimeError("Dilation Rate >= 0")
        # safe_x = temporal_padding_3d(x, (0, self.kernel_size - x.shape[1]))
        # safe_mask = T.ones((x.shape[0], ), dtype=theano.config.floatX).dimshuffle([0, 'x'])
        # !!! convert safe_mask from col to matrix
        # safe_mask = T.unbroadcast(safe_mask, 1)
        # x, mask = ifelse(T.gt(self.kernel_size - x.shape[1], 0),
        #                  (safe_x, safe_mask),
        #                  (new_x, mask))
        # (batch, conv_after_length, hidden_dim)
        conv_result = self.forward_conv_batch(new_x)
        # new_x = Print(new_x)
        # mask = Print()(mask)
        pooling_result = get_pooling_batch(conv_result, mask, self.pooling)
        dropout_out = dropout_from_layer(pooling_result, self.dropout)
        return self.act.activate(dropout_out + self.b)


class MultiFilterConvolutionLayer():
    def __init__(self, in_dim, hidden_dim, kernel_sizes=[3, 4, 5], padding='same', pooling='max', dilation_rate=1.0,
                 activation='relu', prefix="", initializer=GlorotUniformInitializer(), dropout=0.0, verbose=True):
        """
        Init Function for ConvolutionLayer
        :param in_dim:
        :param hidden_dim:
        :param kernel_sizes:
        :param padding: 'same', 'valid'
        :param pooling: 'max', 'mean', 'min'
        :param dilation_rate:
        :param activation:
        :param prefix:
        :param initializer:
        :param dropout:
        :param verbose:
        """
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        self.conv_layers = list()
        self.in_dim = in_dim
        self.out_dim = hidden_dim * len(kernel_sizes)
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.pooling = pooling
        self.dropout = dropout
        self.act = Activation(activation)

        self.params = list()
        self.norm_params = list()

        # L1, L2 Norm
        self.l1_norm = 0
        self.l2_norm = 0

        for filter_hs in kernel_sizes:
            self.conv_layers.append(ConvolutionLayer(in_dim=self.in_dim, hidden_dim=hidden_dim, kernel_size=filter_hs,
                                                     padding=self.padding, pooling=self.pooling,
                                                     dilation_rate=self.dilation_rate, activation=activation,
                                                     prefix=prefix+"filter%s_" % filter_hs, initializer=initializer,
                                                     dropout=dropout, verbose=verbose))
            self.params += self.conv_layers[-1].params
            self.norm_params += self.conv_layers[-1].norm_params
            self.l1_norm += self.conv_layers[-1].l1_norm
            self.l2_norm += self.conv_layers[-1].l2_norm

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Filter Num  (Hidden): %d' % self.hidden_dim)
            logger.debug('Kernel Size (Windows): %s' % self.kernel_sizes)
            logger.debug('Padding method :  %s' % self.padding)
            logger.debug('Dilation Rate  :  %s' % self.dilation_rate)
            logger.debug('Pooling method :  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def forward(self, x):
        layer_out = list()
        for layer in self.conv_layers:
            layer_out.append(layer.forward(x))
        return T.concatenate(layer_out)

    def forward_batch(self, x, mask):
        layer_out = list()
        for layer in self.conv_layers:
            layer_out.append(layer.forward_batch(x, mask))
        return T.concatenate(layer_out, 1)


class MultiChannelConvolutionLayer(ConvolutionLayer):
    pass


def test():
    import numpy as np
    np.random.seed(0)
    import theano
    test_times = 50
    for i in xrange(test_times):
        batch = np.random.random_integers(2, 50)
        print "Batch:", batch
        length = 25
        in_dim = 5
        hidden = 4
        kernel_size = np.random.choice([3, 4, 5])
        a = np.random.uniform(size=(batch, length, in_dim))
        a = a.astype(theano.config.floatX)
        x = T.matrix()
        x_batch = T.tensor3()
        mask = np.zeros(shape=(batch, length))
        mask[0, :] = 1
        for j in xrange(1, batch):
            mask[j, :np.random.randint(1, length)] += 1
        print np.sum(mask, axis=1)
        mask = mask.astype(theano.config.floatX)
        a = a * mask[:, :, None]
        pooling_method_list = ['max', 'mean', 'sum', 'min']
        pooling_method = np.random.choice(pooling_method_list)
        dilation_rate_list = [0, 0.25, 0.5, 0.75, 1]
        convlayer = ConvolutionLayer(in_dim, hidden, kernel_size=kernel_size, pooling=pooling_method,
                                     dilation_rate=np.random.choice(dilation_rate_list),
                                     dropout=0)

        x_c = convlayer.forward(x)
        test_f = theano.function([x], x_c)
        x_c_batch = convlayer.forward_batch(x_batch, mask)
        test_f_batch = theano.function([x_batch], x_c_batch)

        result_list = list()
        for v, m in zip(a, mask):
            c = v[:int(np.sum(m))]
            result_list.append(test_f(c))
        result_list = np.array(result_list, dtype=theano.config.floatX)
        b = test_f_batch(a)
        error = np.sum(result_list - b)
        print error
        if error > 0.001:
            print "Error"
            exit(1)


if __name__ == "__main__":
    test()
