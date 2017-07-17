# coding=utf-8
import logging
import sys

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np

from __init__ import default_initializer, OOV_KEY
from dictionary import Dictionary

__author__ = 'roger'


sys.setrecursionlimit(10000)
logger = logging.getLogger(__name__)
_utils_stop_words = None


# Load and Save

def save_random_state(filename):
    with file(filename, 'wb') as fout:
        pickle.dump(np.random.get_state(), fout)
    logger.info("Save Random State to %s" % filename)


def load_random_state(filename):
    with file(filename, 'rb') as fin:
        np.random.set_state(pickle.load(fin))
    logger.info("Load Random State from %s" % filename)


def save_dev_test_loss(filename, dev_losses, test_losses):
    with file(filename, 'wb') as fout:
        pickle.dump(dev_losses, fout)
        pickle.dump(test_losses, fout)


def load_dev_test_loss(filename):
    with file(filename, 'rb') as fin:
        dev_losses = pickle.load(fin)
        test_losses = pickle.load(fin)
    return dev_losses, test_losses


def load_model(filename, compress=True):
    if compress:
        import gzip
        with gzip.GzipFile(filename, 'rb') as fin:
            model = pickle.load(fin)
    else:
        with file(filename, 'rb') as fin:
            model = pickle.load(fin)
    logger.info("Load Model from %s" % filename)
    return model


def save_model(filename, model, compress=True):
    if compress:
        import gzip
        with gzip.GzipFile(filename, 'wb', compresslevel=5) as out:
            pickle.dump(model, out)
    else:
        with file(filename, 'wb') as out:
            pickle.dump(model, out)
    logger.info("Save Model to %s" % filename)


# Theano Shared Variable
def shared_rand_matrix(shape, name=None, initializer=default_initializer):
    import theano
    matrix = initializer.generate(shape=shape)
    return theano.shared(value=np.asarray(matrix, dtype=theano.config.floatX), name=name)


def shared_matrix(w, name=None, dtype=None,):
    import theano
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(np.asarray(w, dtype=dtype), name=name)


def shared_zero_matrix(shape, name=None, dtype=None):
    import theano
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared((np.zeros(shape, dtype=dtype)), name=name)


def shared_ones_matrix(shape, name=None, dtype=None):
    import theano
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared((np.ones(shape, dtype=dtype)), name=name)


def shared_scalar(value=0, name=None, dtype=None):
    import theano
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(np.asscalar(np.array([value], dtype=dtype)), name=name)


def shared_zero_scalar(name=None, dtype=None):
    return shared_scalar(0, name=name, dtype=dtype)


def shared_one_scalar(name=None, dtype=None):
    return shared_scalar(1, name=name, dtype=dtype)


def constant_scalar(value=0, name=None, dtype=None):
    import theano.tensor
    if dtype is None:
        dtype = theano.config.floatX
    return theano.tensor.constant(np.asscalar(np.array([value], dtype=dtype)), name=name)


def constant_zero_scalar(name=None, dtype=None):
    return constant_scalar(0, name=name, dtype=dtype)


def constant_one_scalar(name=None, dtype=None):
    return constant_scalar(1, name=name, dtype=dtype)


# Variable Operation
def as_floatx(variable):
    import theano.tensor
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)
    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def ndarray_slice(x, n, dim):
    if x.ndim == 1:
        return x[n * dim:(n + 1) * dim]
    if x.ndim == 2:
        return x[:, n * dim:(n + 1) * dim]
    if x.ndim == 3:
        return x[:, :, n * dim:(n + 1) * dim]
    raise ValueError('Invalid slice dims!')


def array2str(array, space=' '):
    return space.join(["%.6f" % b for b in array])


# Train Operation
def progress_bar_str(curr, final):
    pre = int(float(curr) / final * 50)
    remain = 50 - pre
    progress_bar = "[%s>%s] %d/%d" % ('=' * pre, ' ' * remain, curr, final)
    return progress_bar


def align_batch_size(train_index, batch_size):
    """
    对训练数据根据Batch大小对齐，少则随机抽取实例加入在末尾
    :param train_index: 训练顺序
    :param batch_size: Batch大小
    :return:
    """
    if len(train_index) % batch_size == 0:
        return train_index
    else:
        raw_len = len(train_index)
        remain = batch_size - len(train_index) % batch_size
        for i in range(remain):
            ran_i = np.random.randint(0, raw_len - 1)
            train_index.append(train_index[ran_i])
        return train_index


def get_train_sequence(train_x, batch_size):
    """
    依据Batch大小产生训练顺序
    :param train_x:
    :param batch_size:
    :return:
    """
    train_index = range(len(train_x))
    train_index = align_batch_size(train_index, batch_size)
    np.random.shuffle(train_index)
    return train_index


def generate_cross_validation_index(train_data, cv_times=5, dev_ratio=0.1, random=True):
    """
    Generate Corss Validation Index
    :param train_data: np.array
    :param cv_times:
    :param dev_ratio:
    :param random:
    :return:
    """
    if random:
        data_index = np.random.permutation(train_data.shape[0])
    else:
        data_index = np.arange(train_data.shape[0])
    size_per_cv = train_data.shape[0] / cv_times
    for cv_i in xrange(cv_times):
        if cv_i == cv_times - 1:
            test_index = data_index[cv_i * size_per_cv:]
            train_index = data_index[:cv_i * size_per_cv]
        else:
            test_index = data_index[cv_i * size_per_cv: (cv_i + 1) * size_per_cv]
            train_index = np.concatenate([data_index[:cv_i * size_per_cv], data_index[(cv_i + 1) * size_per_cv:]])
        dev_range_right = np.round(train_index.shape[0] * dev_ratio).astype(np.int)
        random_train_index = np.random.permutation(train_index)
        dev_index, train_index = random_train_index[:dev_range_right], random_train_index[dev_range_right:]
        yield train_index, dev_index, test_index


def read_file(filename, word_dict, split_symbol=" |||| ", low_case=False, add_unknown_word=False, encoding="utf8"):
    x = list()
    y = list()
    import codecs
    instances = codecs.open(filename, 'r', encoding=encoding).readlines()
    instances = [(int(line.split(split_symbol)[0]), line.split(split_symbol)[1].strip()) for line in instances]
    for instance in instances:
        label, sen = instance
        token = list()
        for word in sen.split():
            if low_case:
                word.lower()
            if word not in word_dict:
                if add_unknown_word:
                    word_dict[word] = len(word_dict) + 1
                else:
                    word = OOV_KEY
            token.append(word_dict[word])
        y.append(label)
        x.append(token)
    len_list = [len(tokens) for tokens in x]
    max_len = np.max(len_list)
    for instance, length in zip(x, len_list):
        for j in xrange(max_len - length):
            instance.append(0)
    return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


def read_sst(train_file, dev_file, test_file, split_symbol=" |||| ", low_case=False):
    word_dict = dict()

    train_x, train_y = read_file(train_file, word_dict=word_dict, split_symbol=split_symbol, low_case=low_case)
    dev_x, dev_y = read_file(dev_file, word_dict=word_dict, split_symbol=split_symbol, low_case=low_case)
    test_x, test_y = read_file(test_file, word_dict=word_dict, split_symbol=split_symbol, low_case=low_case)
    return [train_x, train_y], [dev_x, dev_y], [test_x, test_y], word_dict


def pre_logger(log_file_name, file_handler_level=logging.DEBUG, screen_handler_level=logging.INFO):
    # Logging configuration
    # Set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    init_logger = logging.getLogger()
    init_logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("log/{}.log".format(log_file_name))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(file_handler_level)
    # Screen logger
    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(screen_handler_level)
    init_logger.addHandler(file_handler)
    init_logger.addHandler(screen_handler)
    return init_logger


def make_cv_index(data_size, cv_num, random=False):
    instance_id = range(data_size)
    if random:
        np.random.shuffle(instance_id)
    for cv in xrange(cv_num):
        train, dev = list(), list()
        for i in instance_id:
            if i % cv_num == cv:
                dev.append(i)
            else:
                train.append(i)
        yield train, dev


def is_stop_words(word, language='english'):
    global _utils_stop_words
    if _utils_stop_words is None or language not in _utils_stop_words:
        from nltk.corpus import stopwords
        if _utils_stop_words is None:
            _utils_stop_words = dict()
        _utils_stop_words[language] = set(stopwords.words(language))
    return word in _utils_stop_words[language]


def generate_sentence_token(sentence, max_len=0, remove_stop=False, low_case=False, language='english'):
    tokens = list()
    for token in sentence.split():
        if low_case:
            token = token.lower()
        if remove_stop and is_stop_words(token, language=language):
            continue
        tokens.append(token)
    if max_len > 0:
        return tokens[:max_len]
    else:
        return tokens


def load_sentence_pair_dict(data_file, max_len=0, remove_stop=False, encoding='utf8',
                            split_symbol='\t', low_case=False, language='english'):
    d = Dictionary()
    with open(data_file, 'r') as fin:
        for line in fin:
            label, q1, q2 = line.decode(encoding).strip().split(split_symbol)
            q1_token = generate_sentence_token(q1, max_len=max_len, remove_stop=remove_stop,
                                               low_case=low_case, language=language)
            q2_token = generate_sentence_token(q2, max_len=max_len, remove_stop=remove_stop,
                                               low_case=low_case, language=language)
            for token in q1_token + q2_token:
                d.add_word(token)
    return d


def load_sentence_dict(data_file, max_len=0, remove_stop=False, encoding='utf8',
                       split_symbol='\t', low_case=False, language='english'):
    d = Dictionary()
    with open(data_file, 'r') as fin:
        for line in fin:
            label, sentence = line.decode(encoding).strip().split(split_symbol)
            sentence_token = generate_sentence_token(sentence, max_len=max_len, remove_stop=remove_stop,
                                                     low_case=low_case, language=language)
            for token in sentence_token:
                d.add_word(token)
    return d


def load_sentence_pair(data_file, dictionary, max_len=0, remove_stop=True, encoding='utf8',
                       split_symbol='\t', low_case=True, language='english', add_oov=False):
    zero_count = 0
    real_max_len = 0
    oov_count = 0
    iv_count = 0
    _x1, _x2, _y = list(), list(), list()
    with open(data_file, 'r') as fin:
        for line in fin:
            label, q1, q2 = line.decode(encoding).strip().split(split_symbol)
            q1_token = generate_sentence_token(q1, max_len=max_len, remove_stop=remove_stop,
                                               low_case=low_case, language=language)
            q2_token = generate_sentence_token(q2, max_len=max_len, remove_stop=remove_stop,
                                               low_case=low_case, language=language)

            _y.append(int(label))

            def get_temp_token(_token_list):
                temp_token = list()
                _iv, _oov = 0, 0
                for token in _token_list:
                    if token in dictionary:
                        temp_token.append(dictionary[token])
                        _iv += 1
                    else:
                        if add_oov:
                            temp_token.append(dictionary.oov_index())
                        _oov += 1
                return temp_token, _iv, _oov

            q1_clean_token, q1_iv, q1_oov = get_temp_token(q1_token)
            q2_clean_token, q2_iv, q2_oov = get_temp_token(q2_token)
            _x1.append(q1_clean_token)
            _x2.append(q2_clean_token)

            q1_len, q2_len = len(q1_clean_token), len(q2_clean_token)
            iv_count = iv_count + q1_iv + q2_iv
            oov_count = oov_count + q1_oov + q2_oov
            if q1_len == 0 or q2_len == 0:
                zero_count += 1
            if max(q1_len, q2_len) > real_max_len:
                real_max_len = max(q1_len, q2_len)

    oov_ratio = float(oov_count) / (oov_count + iv_count)
    logger.info("OOV Ratio in %s: %s" % (data_file, oov_ratio))
    logger.info("NULL Sentence Pair in %s: %s" % (data_file, zero_count))
    for i in xrange(len(_x1)):
        _x1[i].extend([0] * (real_max_len - len(_x1[i])))
        _x2[i].extend([0] * (real_max_len - len(_x2[i])))
    _y = np.array(_y, dtype=np.int32)
    _x1 = np.array(_x1, dtype=np.int32)
    _x2 = np.array(_x2, dtype=np.int32)
    return _y, _x1, _x2


def load_sentence(data_file, dictionary, max_len=0, remove_stop=True, encoding='utf8',
                  split_symbol='\t', low_case=True, language='english', add_oov=False):
    zero_count = 0
    real_max_len = 0
    oov_count = 0
    iv_count = 0
    _x, _y = list(), list()
    with open(data_file, 'r') as fin:
        for line in fin:
            label, text = line.decode(encoding).strip().split(split_symbol)
            text_token = generate_sentence_token(text, max_len=max_len, remove_stop=remove_stop,
                                                 low_case=low_case, language=language)

            _y.append(int(label))

            def get_temp_token(_token_list):
                temp_token = list()
                _iv, _oov = 0, 0
                for token in _token_list:
                    if token in dictionary:
                        temp_token.append(dictionary[token])
                        _iv += 1
                    else:
                        if add_oov:
                            temp_token.append(dictionary.oov_index())
                        _oov += 1
                return temp_token, _iv, _oov

            text_clean_token, text_iv, text_oov = get_temp_token(text_token)
            _x.append(text_clean_token)

            text_len = len(text_clean_token)
            iv_count = iv_count + text_iv
            oov_count = oov_count + text_oov
            if text_len == 0:
                zero_count += 1
            if text_len > real_max_len:
                real_max_len = text_len

    oov_ratio = float(oov_count) / (oov_count + iv_count)
    logger.info("OOV Ratio in %s: %s" % (data_file, oov_ratio))
    logger.info("NULL Sentence Pair in %s: %s" % (data_file, zero_count))
    for i in xrange(len(_x)):
        _x[i].extend([0] * (real_max_len - len(_x[i])))
    _y = np.array(_y, dtype=np.int32)
    _x = np.array(_x, dtype=np.int32)
    return _y, _x


def duplicate_train_data(data_x1, data_x2, data_y):
    new_x1 = np.concatenate([data_x1, data_x2])
    new_x2 = np.concatenate([data_x2, data_x1])
    new_y = np.concatenate([data_y, data_y])
    return new_x1, new_x2, new_y
