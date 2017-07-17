# -*- coding: utf-8 -*-
import logging

import theano
import theano.tensor as T
from theano.ifelse import ifelse

from __init__ import default_initializer
from activations import Activation
from utils import shared_rand_matrix, shared_zero_matrix, shared_scalar, shared_zero_scalar

__author__ = 'roger'

logger = logging.getLogger(__name__)
epsilon = 1e-7


class RecursiveEncoder(object):
    def __init__(self, in_dim, hidden_dim, initializer=default_initializer, normalize=True, dropout=0,
                 reconstructe=True, activation="tanh", verbose=True):
        """
        :param in_dim:          输入维度
        :param hidden_dim:      隐层维度
        :param initializer:     随机初始化器
        :param normalize:       是否归一化
        :param dropout:         dropout率
        :param activation:      激活函数
        :param verbose:         是否输出Debug日志内容
        :return:
        """
        self.in_dim = in_dim
        self.out_dim = hidden_dim
        self.hidden_dim = hidden_dim
        assert self.in_dim == self.hidden_dim

        self.initializer = initializer
        self.normalize = normalize
        self.dropout = dropout
        self.verbose = verbose
        self.act = Activation(activation)
        # Composition Function Weight
        # (dim, 2 * dim)
        self.W = shared_rand_matrix((self.hidden_dim, 2 * self.in_dim), 'W', initializer=initializer)
        # (dim, )
        self.b = shared_zero_matrix((self.hidden_dim,), 'b')
        # Reconstruction Function Weight
        # (2 * dim, dim)
        self.Wr = shared_rand_matrix((2 * self.in_dim, self.hidden_dim), 'Wr', initializer=initializer)
        # (2 * dim, )
        self.br = shared_zero_matrix((self.in_dim * 2,), 'br')
        self.params = [self.W, self.b, self.Wr, self.br]
        self.norm_params = [self.W, self.Wr]

        self.l1_norm = sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = sum([T.sum(param ** 2) for param in self.norm_params])

        if verbose:
            logger.debug('Architecture of RAE built finished, summarized as below: ')
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Normalize:        %s' % self.normalize)
            logger.debug('Activation:       %s' % self.act)
            logger.debug('Dropout Rate:     %s' % self.dropout)

    def compose(self, left_v, right_v):
        v = T.concatenate([left_v, right_v])
        z = self.act.activate(self.b + T.dot(self.W, v))
        if self.normalize:
            z = z / z.norm(2)
        r = self.act.activate(self.br + T.dot(self.Wr, z))
        w_left_r, w_right_r = r[:self.hidden_dim], r[self.hidden_dim:]
        if self.normalize:
            w_left_r = w_left_r / w_left_r.norm(2)
            w_right_r = w_right_r / w_right_r.norm(2)
        loss_rec = T.sum((w_left_r - left_v) ** 2) + T.sum((w_right_r - right_v) ** 2)
        return z, loss_rec

    def encode(self, seq, vecs, loss_rec):
        # vecs[t[0]] and vecs[t[0]] ==> vecs[t[2]]
        w_left, w_right = vecs[seq[0]], vecs[seq[1]]
        z, loss_rec = self.compose(w_left, w_right)
        return T.set_subtensor(vecs[seq[2]], z), loss_rec

    def forward(self, x, seq):
        """
        :param x:   (length, dim)
        :param seq: (length - 1, 3)
        :return:
        """
        # (length, dim) -> (2 * length - 1, dim)
        vector = T.concatenate([x, T.zeros_like(x)[:-1, :]], axis=0)
        # vector = theano.printing.Print()(vector)
        # scan length-1 times
        hs, _ = theano.scan(fn=self.encode,
                            sequences=seq,
                            outputs_info=[vector, shared_scalar(0)],
                            name="compose_phrase")
        comp_vec_init = hs[0][-1][-1]
        comp_rec_init = T.sum(hs[1])
        if self.normalize:
            hidden = x[0] / x[0].norm(2)
        else:
            hidden = x[0]
        comp_vec = ifelse(x.shape[0] > 1, comp_vec_init, hidden)
        comp_rec = ifelse(x.shape[0] > 1, comp_rec_init, shared_zero_scalar())
        return comp_vec, comp_rec

    def compose_batch(self, left, right, W, b, Wr, br):
        """
        合成函数代表一个Batch中的其中一个合成过程
        :param left:  (batch, dim)
        :param right: (batch, dim)
        :param W:     (dim, dim)
        :param b:     (dim, )
        :param Wr:    (dim, dim)
        :param br:    (dim,)
        :return:
        """
        v = T.concatenate([left, right], axis=1)  # [(batch, dim) (batch, dim)] -> (batch, 2 * dim)
        z = self.act.activate(b + T.dot(v, W.T))  # (batch, 2 * dim) dot (dim, 2 * dim)T -> (batch, dim)
        if self.normalize:
            z = z / (z.norm(2, axis=1)[:, None] + epsilon)  # (batch, dim) -> (batch, dim) normalize by row
        r = self.act.activate(br + T.dot(z, Wr.T))  # (batch, dim) dot (2 * dim, dim)T -> (batch, 2 * dim)
        # (batch, 2 * dim) -> [(batch, dim) (batch. dim)]
        left_r, right_r = r[:, :self.hidden_dim], r[:, self.hidden_dim:]
        if self.normalize:
            # (batch, dim) -> (batch, dim) normalize by row
            left_r /= (left_r.norm(2, axis=1)[:, None] + epsilon)
            # (batch, dim) -> (batch, dim) normalize by row
            right_r /= (right_r.norm(2, axis=1)[:, None] + epsilon)
        # (batch, )
        loss_rec = T.sum((left_r - left) ** 2, axis=1) + T.sum((right_r - right) ** 2, axis=1)
        # (batch, dim) (batch)
        return z, loss_rec

    def encode_batch(self, _seq, _mask, _input, _pre, loss_rec, W, b, Wr, br, range_index):
        """
        batch合成短语表示过程中 单词循环执行的函数
        :param _seq:   (batch, 3)
        :param _mask:  (batch, )
        :param _input: (batch, word * 2 - 1, dim)
        :param _pre:   (batch, dim)
        :param loss_rec: (batch, )
        :param W:      (dim, dim)
        :param b:      (dim, )
        :param Wr:     (dim, dim)
        :param br:     (dim,)
        :return:       (batch, dim)
        """
        left = _seq[:, 0]
        right = _seq[:, 1]
        # (batch, dim)
        # left_vec = _input[T.arange(self.batch), left]
        left_vec = _input[range_index, left]
        # (batch, dim)
        right_vec = _input[range_index, right]
        # (batch, dim) (batch, dim) -> (batch, 2 * dim), (batch, )
        left_right, loss_rec = self.compose_batch(left_vec, right_vec, W, b, Wr, br)
        # (batch, 2 * dim)
        # 若掩码已为0 则代表已经超出原短语长度 此为多余计算 直接去上一轮结果作为该轮结果
        left_right = _mask[:, None] * left_right + (1. - _mask[:, None]) * _pre
        # (batch, )
        # 若掩码已为0 则代表已经超出原短语长度 此为多余计算 用0掩码消去
        loss_rec *= _mask
        # (batch, word * 2 - 1, dim), (batch, dim), (batch, )
        return T.set_subtensor(_input[range_index, _seq[:, 2]], left_right), left_right, loss_rec

    def forward_batch(self, x, mask, seqs):
        """
        :param x:    (batch, length, dim)
        :param mask: (batch, length)
        :param seqs: (batch, length - 1, 3)
        :return:
        """
        zeros_rec = T.zeros((x.shape[0],))
        # (batch, length, dim) -> (batch, 2 * length - 1, dim)
        vector = T.concatenate([x, T.zeros_like(x)[:, :-1, :]], axis=1)
        # scan仅能循环扫描张量的第一维 故转置输入的张量
        # (batch, length - 1, 3) -> (length - 1, batch, 3)
        seqs = T.transpose(seqs, axes=(1, 0, 2))
        # (batch, length - 1) -> (length - 1, batch)
        mask = T.transpose(mask, axes=(1, 0))
        range_index = T.arange(x.shape[0])
        result, _ = theano.scan(fn=self.encode_batch,  # 编码函数，对batch数量的短语进行合成
                                sequences=[seqs, mask[1:]],  # 扫描合成路径和掩码
                                # 因合成次数为短语长度-1 所以对于长度为1的短语，掩码第一次循环即为0
                                # 故取vector的第0维（第一个词）作为初始值，直接返回
                                outputs_info=[vector, vector[:, 0, :], zeros_rec],
                                non_sequences=[self.W, self.b, self.Wr, self.br, range_index],
                                name="compose_scan")
        phrases, pres, loss_recs = result
        # (word - 1, batch, dim) -> (batch, dim)
        # 最后一次合成扫描返回的结果为最终表示
        phrases = pres[-1]
        sum_loss_recs = T.sum(loss_recs, axis=0)
        # (batch, dim)
        # 归一化
        if self.normalize:
            phrases = phrases / phrases.norm(2, axis=1)[:, None]
        return phrases, sum_loss_recs


def random_seq(length):
    if length == 1:
        return [[0, 0, 0]]
    import random
    seq = list()
    words = range(length)
    for i in xrange(length - 1):
        left_index = random.randint(0, len(words) - 1)
        if left_index == len(words) - 1:
            left_index, right_index = left_index - 1, left_index
        else:
            right_index = left_index + 1
        r = words.pop(right_index)
        l = words.pop(left_index)
        words.insert(left_index, length + i)
        seq.append((l, r, length + i))
    return seq


def test():
    import numpy as np
    np.random.seed(0)
    import theano
    test_times = 50
    for i in xrange(test_times):
        # batch = np.random.random_integers(2, 50)
        batch = 50
        print "Batch:", batch
        length = 25
        in_dim = 5
        hidden = in_dim
        a = np.random.uniform(size=(batch, length, in_dim))
        a = a.astype(theano.config.floatX)

        x = T.matrix()
        x_batch = T.tensor3()
        s = T.imatrix()
        ss = T.itensor3()
        mm = T.matrix()

        mask = np.zeros(shape=(batch, length))
        mask[0, :] = 1
        seq = np.zeros(shape=(batch, length - 1, 3))
        seq[0, :, :] = random_seq(length)
        for j in xrange(1, batch):
            random_len = np.random.randint(1, length)
            mask[j, :random_len] += 1
            seq[j, :random_len - 1] = random_seq(random_len)
            seq[j, random_len - 1:] += 2 * random_len - 2
        seq = seq.astype(np.int32)
        print np.sum(mask, axis=1)
        mask = mask.astype(theano.config.floatX)
        a = a * mask[:, :, None]
        pooling_method_list = ['max', 'mean', 'sum', 'min']
        pooling_method = np.random.choice(pooling_method_list)
        convlayer = RecursiveEncoder(in_dim, hidden, dropout=0, normalize=np.random.choice([True, False]))

        x_c = convlayer.forward(x, s)
        test_f = theano.function([x, s], x_c)
        x_c_batch = convlayer.forward_batch(x_batch, mm, ss)
        test_f_batch = theano.function([x_batch, mm, ss, ], x_c_batch)

        result_list = list()
        loss_list = list()
        for v, m, _s in zip(a, mask, seq):
            c = v[:int(np.sum(m))]
            if int(np.sum(m)) == 1:
                _s = _s[:1]
            else:
                _s = _s[:int(np.sum(m)) - 1]
            hidden, loss = test_f(c, _s)
            result_list.append(hidden)
            loss_list.append(loss)
        result_list = np.array(result_list, dtype=theano.config.floatX)
        loss_list = np.array(loss_list, dtype=theano.config.floatX)
        hidden, loss = test_f_batch(a, mask, seq)
        error1 = np.sum(result_list - hidden)
        error2 = np.sum(loss_list - loss)
        print error1, error2
        if error1 > 0.001 or error2 > 0.001:
            print "Error"
            exit(1)


if __name__ == "__main__":
    test()
