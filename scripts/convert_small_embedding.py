# coding=utf-8
"""
Most Code in WordEmbedding (load/save word2vec format) refer to Gensim
A little difference from src.embedding (save_word2vec_format/load_word2vec_format)
"""
import sys

import numpy as np

REAL = np.float32


__author__ = 'roger'


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


def save_word2vec_format(word_idx, word_matrix, filename, binary=False):
    # word_matrix[0] for mask in embedding.py
    with open(filename, 'wb') as fout:
        fout.write(any2utf8("%s %s\n" % (len(word_idx), word_matrix.shape[1])))
        for word in word_idx.keys():
            _embedding = word_matrix[word_idx[word]].astype(REAL)
            if binary:
                fout.write(any2utf8(word) + b" " + _embedding.tostring())
            else:
                fout.write(any2utf8("%s %s\n" % (word, ' '.join("%f" % val for val in _embedding))))
    sys.stdout.write("saved %d words to %s with %d in %s style\n" % (len(word_idx), filename, word_matrix.shape[1],
                                                                     "binary" if binary else "text"))


def load_word2vec_format(filename, word_idx=None, binary=False,
                         encoding='utf8', unicode_errors='strict'):
    """
    refer to gensim
    load Word Embeddings
    If you trained the C model using non-utf8 encoding for words, specify that
    encoding in `encoding`.
    :param filename :
    :param word_idx :
    :param binary   : a boolean indicating whether the data is in binary word2vec format.
    :param encoding :
    :param unicode_errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
    """
    sys.stdout.write("loading word embedding from %s\n" % filename)
    vocab = dict()
    with open(filename, 'rb') as fin:
        header = to_unicode(fin.readline(), encoding=encoding)
        vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
        if word_idx is None:
            word_matrix = np.zeros(shape=(vocab_size + 1, vector_size), dtype=REAL)
        else:
            word_matrix = np.zeros(shape=(len(word_idx) + 1, vector_size), dtype=REAL)

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
    sys.stdout.write("loaded %d words pre-trained from %s with %d\n" % (len(vocab), filename, vector_size))
    if word_idx is not None:
        sys.stderr.write("missing %d words\n" % (len(word_idx) - len(vocab)))
    return word_matrix, vector_size, vocab


def main(word_map_file, embedding_file, new_embedding_file, binary=True,
         word_map_encoding='utf-8', embedding_encoding='utf-8'):
    word_map = dict()
    with open(word_map_file, 'r') as fin:
        for line in fin:
            att = line.decode(word_map_encoding).strip().split()
            if len(att) >= 1:
                word = att[0]
            else:
                continue
            if word in word_map:
                sys.stderr.write("duplicate word:", )
                sys.stderr.write(word)
                sys.stderr.write("\n")
            else:
                word_map[word] = len(word_map)
    word_embedding, vsize, vocab = load_word2vec_format(embedding_file, word_map,
                                                        binary=binary, encoding=embedding_encoding)
    save_word2vec_format(vocab, word_embedding, new_embedding_file, binary=binary)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage="Convert Big Vocab Word Embedding into Small Vocab Embedding")
    parser.add_argument('-w', '--word', type=str, required=True, help='Word Map File')
    parser.add_argument('-s', '--src', type=str, required=True, help='Source Embedding File')
    parser.add_argument('-t', '--tar', type=str, required=True, help='Target Embedding File')
    parser.add_argument('--binary', dest='binary', action='store_true',
                        help='Binary Style for Word Embedding File (Default)')
    parser.add_argument('--text', dest='binary', action='store_false',
                        help='Text Style for Word Embedding File')
    parser.add_argument('--word-encoding', dest='word_encoding', type=str,
                        default='utf-8', help='Word Map File Encoding, Default is utf-8.')
    parser.add_argument('--embedding-encoding', dest='embedding_encoding', type=str,
                        default='utf-8', help='Word Embedding Encoding, Default is utf-8.')
    parser.set_defaults(binary=True)
    args = parser.parse_args()
    main(word_map_file=args.word, embedding_file=args.src, new_embedding_file=args.tar,
         binary=args.binary, word_map_encoding=args.word_encoding, embedding_encoding=args.embedding_encoding)
