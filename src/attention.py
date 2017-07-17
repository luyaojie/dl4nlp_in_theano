import abc

import theano.tensor as T

from __init__ import default_initializer
from activations import Activation
from utils import shared_rand_matrix

__author__ = 'roger'


class WordBasedAttention(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, word_dim, seq_dim, initializer=default_initializer):
        self.word_dim = word_dim
        self.seq_dim = seq_dim
        self.initializer = initializer
        self.params = []
        self.norm_params = []
        self.l1_norm = 0
        self.l2_norm = 0

    @abc.abstractmethod
    def score(self, word, sequence, ): pass

    @abc.abstractmethod
    def score_batch(self, word, sequence, ): pass

    def attention(self, word, sequence, ):
        """
        :param word: (dim, )
        :param sequence: (length, dim, )
        :return: weight: (length, )
        """
        score = self.score(word, sequence)
        exp_score = T.exp(score)
        sum_exp_score = T.sum(exp_score)
        weight = exp_score / sum_exp_score
        return weight

    def attention_batch(self, word, sequence, mask):
        """
        :param word: (batch, dim, )
        :param sequence: (batch, length, dim, )        
        :param mask: (batch, length, )
        :return: weight: (batch, length)
        """
        score = self.score_batch(word, sequence)
        exp_score = T.exp(score) * mask
        sum_exp_score = T.sum(exp_score, axis=1)
        weight = exp_score / sum_exp_score[:, None]
        return weight


class DotWordBaedAttetnion(WordBasedAttention):
    """
    Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, and Christopher D. Manning
    In Proceedings of EMNLP 2015
    http://aclweb.org/anthology/D/D15/D15-1166.pdf
    """
    def __init__(self, word_dim, seq_dim):
        super(DotWordBaedAttetnion, self).__init__(word_dim=word_dim, seq_dim=seq_dim)
        assert word_dim == seq_dim

    def score(self, word, sequence, ):
        """
        :param word: (dim, )
        :param sequence: (length, dim)
        :return: score: (length, )
        """
        score = T.dot(sequence, word)
        return score

    def score_batch(self, word, sequence, ):
        """
        :param word: (batch, dim)
        :param sequence: (batch, length, dim)
        :return: weight: (batch, length)
        """
        # (batch, length, dim) dot (batch, dim) -> (batch, length)
        score = T.batched_dot(sequence, word)
        return score


class BilinearWordBasedAttention(WordBasedAttention):
    """
    Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, and Christopher D. Manning
    In Proceedings of EMNLP 2015
    http://aclweb.org/anthology/D/D15/D15-1166.pdf
    """
    def __init__(self, word_dim, seq_dim, initializer=default_initializer):
        super(BilinearWordBasedAttention, self).__init__(word_dim=word_dim, seq_dim=seq_dim, initializer=initializer)
        # (word_dim, seq_dim)
        self.W = shared_rand_matrix((self.seq_dim, self.word_dim, ), 'Attention_W', initializer)
        self.params = [self.W]
        self.norm_params = [self.W]

    def score(self, word, sequence, ):
        """
        :param word: (word_dim, )
        :param sequence: (length, seq_dim, )
        :return score: (length, )
        """
        # (length, seq_dim, ) dot (seq_dim, word_dim) dot (word_dim, ) -> (length, )
        score = T.dot(T.dot(sequence, self.W), word)
        return score

    def score_batch(self, word, sequence, ):
        """
        :param word: (batch, word_dim)
        :param sequence: (batch, length, seq_dim)
        :return: score: (batch, length, )
        """
        # (batch, length, seq_dim) dot (seq_dim, word_dim) -> (batch, length, word_dim)
        # (batch, length, word_dim) dot (batch, word_dim) -> (batch, length)
        score = T.batched_dot(T.dot(sequence, self.W), word)
        return score


class ConcatWordBasedAttention(WordBasedAttention):
    """
    Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, and Christopher D. Manning
    In Proceedings of EMNLP 2015
    http://aclweb.org/anthology/D/D15/D15-1166.pdf
    """
    def __init__(self, word_dim, seq_dim, initializer=default_initializer):
        super(ConcatWordBasedAttention, self).__init__(word_dim=word_dim, seq_dim=seq_dim,
                                                       initializer=default_initializer)
        # (word_dim + seq_dim)
        self.W = shared_rand_matrix((self.word_dim + self.seq_dim, ), 'Attention_W', initializer)
        self.params = [self.W]
        self.norm_params = [self.W]

    def score(self, word, sequence, ):
        """
        :param word: (word_dim, )
        :param sequence: (length, seq_dim)
        :return: score: (length, )
        """
        w1, w2 = self.W[:self.word_dim], self.W[self.word_dim:]
        # (word_dim, ) dot (word_dim, ) -> scalar
        hidden1 = T.dot(word, w1)
        # (length, seq_dim) dot (seq_dim, ) -> (length, )
        hidden2 = T.dot(sequence, w2)
        score = hidden1 + hidden2
        return score

    def score_batch(self, word, sequence, ):
        """
        :param word: (batch, word_dim)
        :param sequence: (batch, length, seq_dim)
        :return: score: (batch, length, )
        """
        w1, w2 = self.W[:self.word_dim], self.W[self.word_dim:]
        # (batch, word_dim) dot (word_dim, ) -> (batch, )
        hidden1 = T.dot(word, w1)
        # (batch, length, seq_dim) dot (seq_dim, ) -> (batch, length, )
        hidden2 = T.dot(sequence, w2)
        score = hidden1[:, None] + hidden2
        return score


class NNWordBasedAttention(WordBasedAttention):
    """
    Neural Machine Translation By Jointly Learning To Align and Translate
    Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio
    In Proceedings of ICLR 2015
    http://arxiv.org/abs/1409.0473v3
    """
    def __init__(self, word_dim, seq_dim, hidden_dim, activation='tanh', initializer=default_initializer):
        super(NNWordBasedAttention, self).__init__(word_dim=word_dim, seq_dim=seq_dim, initializer=default_initializer)
        # (dim, dim)
        self.hidden_dim = hidden_dim
        self.W = shared_rand_matrix((self.word_dim, self.hidden_dim), 'Attention_W', initializer)
        self.U = shared_rand_matrix((self.seq_dim, self.hidden_dim), 'Attention_U', initializer)
        self.v = shared_rand_matrix((self.hidden_dim, ), 'Attention_v', initializer)
        self.act = Activation(activation)
        self.params = [self.W]
        self.norm_params = [self.W]

    def score(self, word, sequence, ):
        """
        :param word: (word_dim, )
        :param sequence: (length, seq_dim)
        :return: score: (length, )
        """
        # (word_dim, ) dot (word_dim, hidden_dim) -> (hidden_dim, )
        hidden1 = T.dot(word, self.W)
        # (length, seq_dim) dot (seq_dim, hidden_dim) -> (length, hidden_dim)
        hidden2 = T.dot(sequence, self.U)
        # (hidden_dim, ) + (length, hidden_dim) -> (length, hidden_dim)
        hidden = hidden1[None, :] + hidden2
        # (length, hidden_dim) -> (length, hidden_dim)
        act_hidden = self.act.activate(hidden)
        # (length, hidden_dim) dot (hidden_dim, ) -> (length, )
        score = T.dot(act_hidden, self.v)
        return score

    def score_batch(self, word, sequence, ):
        """
        :param word: (batch, word_dim)
        :param sequence: (batch, length, seq_dim)
        :return: score: (batch, length, )
        """
        # (batch, word_dim) dot (word_dim, hidden_dim) -> (batch, hidden_dim)
        hidden1 = T.dot(word, self.W)
        # (batch, length, seq_dim) dot (seq_dim, hidden_dim) -> (batch, length, hidden_dim)
        hidden2 = T.dot(sequence, self.U)
        # (batch, length, hidden_dim) + (batch, hidden_dim) -> (batch, length, hidden_dim)
        hidden = hidden1[:, None, :] + hidden2
        # (batch, length, hidden_dim) -> (batch, length, hidden_dim)
        act_hidden = self.act.activate(hidden)
        # (batch, length, hidden_dim) dot (hidden_dim, ) -> (batch, length, )
        score = T.dot(act_hidden, self.v)
        return score
