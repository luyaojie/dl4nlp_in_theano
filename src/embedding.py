# coding=utf-8
"""
Most Code in WordEmbedding (load/save word2vec format) refer to Gensim
"""
import logging

import numpy as np
import theano
import theano.tensor as T

from Initializer import UniformInitializer
from __init__ import default_initializer, OOV_KEY
from utils import shared_rand_matrix, shared_matrix

__author__ = 'roger'
REAL = np.float32
logger = logging.getLogger(__name__)
import sys
if sys.version_info[0] >= 3:
    unicode = str


def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode.
    :param text:
    :param encoding: 
    :param errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


class Embedding(object):
    def __init__(self, w=None, size=10000, dim=50, initializer=default_initializer, prefix=""):
        if w is None:
            # Random generate Matrix
            self.size = size
            self.dim = dim
            self.W = shared_rand_matrix(shape=(self.size, self.dim), initializer=initializer,
                                        name=prefix + 'Embedding')
            logger.info("Initialize %d Word with %s" % (self.size, initializer))
        else:
            self.size = w.shape[0]
            self.dim = w.shape[1]
            self.W = shared_matrix(np.array(w, dtype=theano.config.floatX), name=prefix + 'Embedding')
        self.params = [self.W]
        self.norm_params = [self.W[1:]]
        self.l1_norm = T.sum(T.abs_(self.W[1:]))  # sum([T.sum(T.abs_(param)) for param in self.params])
        self.l2_norm = T.sum(self.W[1:] ** 2)  # sum([T.sum(param ** 2) for param in self.params])

    def __getitem__(self, item):
        return self.W[item]

    def get_dim(self):
        return self.dim

    def get_value(self):
        return self.W.get_value()


class WordEmbedding(Embedding):
    def __init__(self, word_idx, dim=50, filename=None, initializer=None, normalize=False,
                 add_unknown_word=True, prefix="Word_", verbose=True, binary=True, encoding="utf8",
                 unicode_errors='strict'):
        self.n_words = max(word_idx.values()) + 1
        self.word_idx = word_idx
        self.idx_word = {idx: word for word, idx in self.word_idx.iteritems()}
        self.initializer = initializer
        if filename is None:
            self.dim = dim
            super(WordEmbedding, self).__init__(size=self.n_words, prefix=prefix, dim=dim)
        else:
            w, dim = self.read_wordvec(filename, self.word_idx, binary=binary, normalize=normalize,
                                       add_unknown_word=add_unknown_word, encoding=encoding,
                                       unicode_errors=unicode_errors)
            self.dim = dim
            super(WordEmbedding, self).__init__(w=w, prefix=prefix, dim=self.dim, )
        if verbose:
            logger.debug('Word Embedding built finished, summarized as below: ')
            logger.debug('Word Size: %d' % self.n_words)
            logger.debug('Word dimension: %d' % self.dim)

    def read_wordvec(self, filename, word_idx, binary=True, add_unknown_word=True, encoding="utf8", normalize=False,
                     unicode_errors='strict'):
        word_matrix, dim, vocab = self.load_word2vec_format(filename, word_idx, binary=binary, encoding=encoding,
                                                            unicode_errors=unicode_errors, normalize=normalize)
        if self.initializer is None:
            self.initializer = UniformInitializer(scale=np.std(word_matrix))
        unknown = 0
        for word in word_idx:
            if word not in vocab and add_unknown_word:
                # add unknown words
                unknown += 1
                word_matrix[word_idx[word]] = self.initializer.generate(dim)
        logger.info("Initialize %d Word with %s" % (unknown, self.initializer))
        return word_matrix, dim

    def add_extra_words(self, words, filename=None):
        """
        Update Word Embedding for extra new words
        :param words:    List of words
        :param filename: Pre-trained Word Embedding
        :return:
        """
        extra_index = self.size
        update_word_idx = dict()
        for w in words:
            if w not in self.word_idx:
                self.word_idx[w] = extra_index
                update_word_idx[w] = extra_index - self.size
                extra_index += 1
        update_num = extra_index - self.size

        # Read or generate Extra Embeddings
        if filename:
            extra_embedding, dim = self.read_wordvec(filename, update_word_idx)
            assert dim == self.dim
        else:
            extra_embedding = np.zeros(shape=(update_num, self.dim), dtype=theano.config.floatX)
            for i in xrange(update_num):
                extra_embedding[i] = self.initializer.generate(self.dim)
        new_embedding = np.concatenate([np.array(self.W.get_value(), dtype=theano.config.floatX), extra_embedding])
        # Update Params
        self.W.set_value(new_embedding)
        self.n_words = max(self.word_idx.values()) + 1
        self.size = self.n_words
        logger.debug("Add %d words to Embedding" % update_num)
        return self.word_idx

    @staticmethod
    def save_word2vec_format(word_idx, word_matrix, filename, binary=False):
        # word_matrix[0] for mask in embedding.py
        assert len(word_idx) + 1 == word_matrix.shape[0]
        with open(filename, 'wb') as fout:
            fout.write(any2utf8("%s %s\n" % (len(word_idx), word_matrix.shape[1])))
            for word in word_idx.keys():
                _embedding = word_matrix[word_idx[word]].astype(REAL)
                if binary:
                    fout.write(any2utf8(word) + b" " + _embedding.tostring())
                else:
                    fout.write(any2utf8("%s %s\n" % (word, ' '.join("%f" % val for val in _embedding))))

    @staticmethod
    def load_word2vec_format(filename, word_idx=None, binary=False, normalize=False,
                             encoding='utf8', unicode_errors='strict'):
        """
        refer to gensim
        load Word Embeddings
        If you trained the C model using non-utf8 encoding for words, specify that
        encoding in `encoding`.
        :param filename :
        :param word_idx :
        :param binary   : a boolean indicating whether the data is in binary word2vec format.
        :param normalize:
        :param encoding :
        :param unicode_errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
        """
        vocab = dict()
        logger.info("loading word embedding from %s", filename)
        with open(filename, 'rb') as fin:
            header = to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
            # 预留第0维 用于任务的mask等
            if word_idx is None:
                word_matrix = np.zeros(shape=(vocab_size + 1, vector_size), dtype=theano.config.floatX)
            else:
                word_matrix = np.zeros(shape=(len(word_idx) + 1, vector_size), dtype=theano.config.floatX)
            word_matrix[0] = np.ones(shape=(vector_size, ), dtype=theano.config.floatX) / 2

            def add_word(_word, _weights):
                if word_idx:
                    if _word not in word_idx:
                        return
                    vocab[_word] = word_idx[_word]
                else:
                    vocab[_word] = len(vocab) + 1
                word_matrix[vocab[_word]] = _weights

            if binary:
                binary_len = np.dtype(REAL).itemsize * vector_size
                for line_no in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                            word.append(ch)
                    word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = np.fromstring(fin.read(binary_len), dtype=REAL)
                    add_word(word, weights)
            else:
                for line_no, line in enumerate(fin):
                    parts = to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                    word, weights = parts[0], list(map(REAL, parts[1:]))
                    add_word(word, weights)
        if word_idx is not None:
            assert (len(word_idx) + 1, vector_size) == word_matrix.shape
        if normalize:
            for word, idx in vocab.iteritems():
                word_matrix[idx] = word_matrix[idx] / np.sqrt(np.sum(np.square(word_matrix[idx])))
        logger.info("loaded %d words pre-trained from %s with %d" % (len(vocab), filename, vector_size))
        return word_matrix, vector_size, vocab

    @staticmethod
    def load_word2vec_word_map(filename, binary=False, oov=True, encoding='utf8', unicode_errors='strict'):
        """
        load Word Embeddings
        If you trained the C model using non-utf8 encoding for words, specify that
        encoding in `encoding`.
        :param filename :
        :param binary   : a boolean indicating whether the data is in binary word2vec format.
        :param oov      :
        :param encoding :
        :param unicode_errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
        """
        vocab = dict()
        logger.info("loading word embedding from %s", filename)
        with open(filename, 'rb') as fin:
            header = to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = map(int, header.split())  # throws for invalid file format

            def add_word(_word):
                vocab[_word] = len(vocab) + 1

            if binary:
                binary_len = np.dtype(REAL).itemsize * vector_size
                for line_no in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                            word.append(ch)
                    word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    fin.read(binary_len)
                    add_word(word)
            else:
                for line_no, line in enumerate(fin):
                    parts = to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                    # word, weights = parts[0], list(map(REAL, parts[1:]))
                    add_word(parts[0])
            if oov:
                add_word(OOV_KEY)
        logger.info("loaded %d word map pre-trained from %s" % (len(vocab), filename))
        return vocab
