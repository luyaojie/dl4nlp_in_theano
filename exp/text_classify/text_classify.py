import logging
import sys
import os
import numpy as np
import theano
import theano.tensor as T

sys.path.append('../../')
from src import OOV_KEY
from src.utils import align_batch_size, shared_zero_matrix, pre_logger, load_sentence_dict, load_sentence, \
    generate_cross_validation_index, save_model, load_model
from src.Initializer import UniformInitializer
from src.embedding import WordEmbedding
from src.optimizer import AdaDeltaOptimizer, AdaGradOptimizer, SGDOptimizer, SGDMomentumOptimizer, AdamOptimizer
from src.classifier import SoftmaxClassifier
from src.dropout import set_dropout_on
from src.recurrent import LSTMEncoder, BiLSTMEncoder, RecurrentEncoder, BiRecurrentEncoder, GRUEncoder, BiGRUEncoder
from src.convolution import MultiFilterConvolutionLayer
from src.metrics import prob_to_accuracy, prob_to_log_loss
from src.layers import MultiHiddenLayer
from src.pooling import CBOWLayer


logger = logging.getLogger(__name__)
DEV_RATIO = 0.1
MAX_LEN = 200
LOW_CASE = True
REMOVE_STOP = False
TOPK = 80000
MIN_COUNT = 3
DUPLICATE = True
MAX_ITER = 10
EMBEDDING_LR = 0.1
PRE_TEST_BATCH = 100
CROSS_VALIDATION_TIMES = 0
LABEL2INDEX = None
CONV_FILTER_SIZES = [3, 4, 5]


class TextClassifier(object):
    def __init__(self, key_index, label_num, pretrain_name=None, encoder='lstm', word_dim=300,
                 hidden='100_100', dropout=0.5, regularization_weight=0.0001,
                 optimizer_name='adagrad', lr=0.1, norm_lim=-1, label2index_filename=None):
        self.label2index, self.index2label = self.load_label_index(label2index_filename, label_num)

        self.indexs = T.imatrix()    # (batch, max_len)
        self.golden = T.ivector()    # (batch, )
        self.max_len = T.iscalar()   # max length

        self.s1_mask = self.indexs[:, :self.max_len] > 0
        self.s1_mask = self.s1_mask * T.constant(1.0, dtype=theano.config.floatX)

        if pretrain_name is None:
            self.embedding = WordEmbedding(key_index, dim=word_dim, initializer=UniformInitializer(scale=0.01))
        else:
            self.embedding = WordEmbedding(key_index, filename=pretrain_name, normalize=False, binary=True)
            assert self.embedding.dim == word_dim

        self.word_embeddings = self.embedding[self.indexs[:, :self.max_len]]

        if type(hidden) is str:
            hidden_dims = [int(hid) for hid in hidden.split('_')]
        else:
            hidden_dims = [hidden]

        if encoder == 'lstm':
            encoder_layer = LSTMEncoder(in_dim=word_dim, hidden_dim=hidden_dims[0], pooling='final', prefix="LSTM_",
                                        dropout=dropout)
        elif encoder == 'bilstm':
            encoder_layer = BiLSTMEncoder(in_dim=word_dim, hidden_dim=hidden_dims[0], pooling='final', prefix="BiLSTM_",
                                          bidirection_shared=True, dropout=dropout)
        elif encoder == 'recurrent':
            encoder_layer = RecurrentEncoder(in_dim=word_dim, hidden_dim=hidden_dims[0], pooling='final',
                                             prefix="Recurrent_", dropout=dropout)
        elif encoder == 'birecurrent':
            encoder_layer = BiRecurrentEncoder(in_dim=word_dim, hidden_dim=hidden_dims[0], pooling='final',
                                               prefix="BiRecurrent_", bidirection_shared=True, dropout=dropout)
        elif encoder == 'gru':
            encoder_layer = GRUEncoder(in_dim=word_dim, hidden_dim=hidden_dims[0], pooling='final',
                                       prefix="GRU_", dropout=dropout)
        elif encoder == 'bigru':
            encoder_layer = BiGRUEncoder(in_dim=word_dim, hidden_dim=hidden_dims[0], pooling='final',
                                         prefix="BiGRU_", bidirection_shared=True, dropout=dropout)
        elif encoder == 'cbow':
            encoder_layer = CBOWLayer(in_dim=word_dim, )
        elif encoder == 'cnn':
            encoder_layer = MultiFilterConvolutionLayer(in_dim=word_dim, hidden_dim=hidden_dims[0],
                                                        pooling='max', prefix="ConvLayer_",
                                                        kernel_sizes=CONV_FILTER_SIZES)
        else:
            raise NotImplementedError

        self.text_embedding = encoder_layer.forward_batch(self.word_embeddings, self.s1_mask)

        if len(hidden_dims) > 1:
            hidden_layer = MultiHiddenLayer(in_dim=encoder_layer.out_dim, hidden_dims=hidden_dims[1:], dropout=dropout,
                                            prefix='Full_Connected_Layer_')
            classifier_input = hidden_layer.forward_batch(self.text_embedding)
            classifier_input_dim = hidden_layer.out_dim
        else:
            classifier_input = self.text_embedding
            classifier_input_dim = encoder_layer.out_dim

        self.classifier = SoftmaxClassifier(classifier_input_dim, label_num, dropout=dropout)
        self.predict_loss = self.classifier.loss(classifier_input, self.golden)
        self.predict_prob = self.classifier.forward_batch(classifier_input)
        self.predict_label = T.argmax(self.predict_prob, axis=1)

        """Params in TextClassifier"""
        self.params = self.classifier.params + encoder_layer.params
        self.l2_norm = self.classifier.l2_norm + encoder_layer.l2_norm
        if len(hidden_dims) > 1:
            self.params += hidden_layer.params
            self.l2_norm += hidden_layer.l2_norm

        self.l2_loss = regularization_weight * self.l2_norm / 2
        self.loss = self.predict_loss + self.l2_loss

        """Opimizer and Loss"""
        if optimizer_name == 'adagrad':
            sgd_optimizer = AdaGradOptimizer(lr=lr, norm_lim=norm_lim)
        elif optimizer_name == 'adadelta':
            sgd_optimizer = AdaDeltaOptimizer(lr=lr, norm_lim=norm_lim)
        elif optimizer_name == 'sgd':
            sgd_optimizer = SGDOptimizer(lr=lr, norm_lim=norm_lim)
        elif optimizer_name == 'momentum':
            sgd_optimizer = SGDMomentumOptimizer(lr=lr, norm_lim=norm_lim)
        elif optimizer_name == 'adam':
            sgd_optimizer = AdamOptimizer(lr=lr, norm_lim=norm_lim)
        else:
            raise NotImplementedError

        self.train_indexs = T.ivector()
        self.train_data_x = shared_zero_matrix(shape=(5, 5), name="train_data_x", dtype=np.int32)
        self.train_data_y = shared_zero_matrix(shape=(5,), name="train_data_y", dtype=np.int32)

        self.model_params = self.params + self.embedding.params

        """Theano Function"""
        if EMBEDDING_LR > 0:
            embedding_updates = SGDOptimizer(lr=EMBEDDING_LR, norm_lim=-1).get_update(self.loss, self.embedding.params)
            updates = sgd_optimizer.get_update(self.loss, self.params, norm_exc_params=self.embedding.params)
            updates.update(embedding_updates)
        elif EMBEDDING_LR < 0:
            # Optimize Embedding using Global Optimizer
            self.params += self.embedding.params
            updates = sgd_optimizer.get_update(self.loss, self.params, norm_exc_params=self.embedding.params)
        else:
            # Fix Embedding
            updates = sgd_optimizer.get_update(self.loss, self.params, norm_exc_params=self.embedding.params)

        self.train_batch = theano.function(inputs=[self.train_indexs, self.max_len],
                                           outputs=[self.loss, self.predict_loss, self.l2_loss],
                                           updates=updates,
                                           givens=[(self.indexs, self.train_data_x[self.train_indexs]),
                                                   (self.golden, self.train_data_y[self.train_indexs])])

        self.loss_batch = theano.function(inputs=[self.indexs, self.golden, self.max_len],
                                          outputs=[self.loss, self.predict_loss, self.l2_loss],
                                          )

        self.pred_prob_batch = theano.function(inputs=[self.indexs, self.max_len],
                                               outputs=[self.predict_prob],)

        self.pred_label_batch = theano.function(inputs=[self.indexs, self.max_len],
                                                outputs=[self.predict_label],)

        self.get_l2_loss = theano.function(inputs=[], outputs=[self.l2_loss, self.l2_norm],)

    def save_model_param_to_file(self, filename):
        to_save = [param.get_value() for param in self.model_params]
        save_model(filename, model=to_save, compress=True)

    def load_model_param_from_file(self, filename):
        to_load = load_model(filename, compress=True)
        for load_param, param in zip(to_load, self.model_params):
            assert load_param.shape == param.get_value().shape
            param.set_value(load_param)

    @staticmethod
    def load_label_index(label_file, label_num):
        label2index = dict()
        index2label = dict()
        if label_file is not None:
            label2index = dict()
            index2label = dict()
            with open(label_file, 'r') as fin:
                for line in fin:
                    label, index = line.decode('utf8').strip().split('\t')
                    label2index[label] = int(index)
                    index2label[int(index)] = label
        else:
            for i in xrange(label_num):
                label2index[str(i)] = i
                index2label[i] = str(i)
        return label2index, index2label

    def set_train_data(self, data):
        x, y = data
        self.train_data_x.set_value(x)
        self.train_data_y.set_value(y)

    @staticmethod
    def predict(x, predict_function, batch_size=100):
        x_length = np.sum(x > 0, axis=1)
        predict_indexs = align_batch_size(range(len(x)), batch_size)
        num_batch = len(predict_indexs) / batch_size
        predict = list()
        for i in xrange(num_batch):
            indexs = predict_indexs[i * batch_size: (i + 1) * batch_size]
            max_len = np.max(x_length[indexs])
            p1 = predict_function(x[indexs], max_len)[0]
            predict.append(p1)
        return np.concatenate(predict)[:len(x)]

    def predict_text_prob(self, text):
        text_seq = text.split()
        indexs = [self.embedding.word_idx[token]
                  if token in self.embedding.word_idx
                  else self.embedding.word_idx[OOV_KEY]
                  for token in text_seq]
        return self.pred_prob_batch([indexs], len(text_seq))

    def prob_to_str(self, prob):
        str_list = list()
        for i, p in enumerate(prob):
            str_list.append("%s:%6.f" % (self.index2label[i], p))
        return '\t'.join(str_list)

    def predict_data_log_loss_acc(self, x, y, test_batch_size=1000):
        pred_prob = self.predict(x, self.pred_prob_batch, test_batch_size)
        acc = prob_to_accuracy(y, pred_prob)
        log_loss = prob_to_log_loss(y, pred_prob, labels=self.index2label.keys())
        return acc, log_loss

    def cross_validation_train(self, data, cv_times=5, max_iter=5, batch_size=128,
                               test_batch_size=1000, pre_test_batch=25, model_path=""):
        data_x, data_y = data
        cv_i = -1
        cv_epoch_log_loss = list()
        cv_epoch_acc = list()
        cv_batch_log_loss = list()
        cv_batch_acc = list()
        self.save_model_param_to_file("model/%s.cv_init.param.model" % model_path)
        for train_index, dev_index, test_index in generate_cross_validation_index(data_x, cv_times=cv_times,
                                                                                  dev_ratio=DEV_RATIO, random=True):
            self.load_model_param_from_file("model/%s.cv_init.param.model" % model_path)
            logger.debug("Train Size %s, Dev Size %s, Test Size %s" % (train_index.shape[0],
                                                                       dev_index.shape[0],
                                                                       test_index.shape[0]))
            cv_i += 1
            train_x, train_y = data_x[train_index], data_y[train_index]
            dev_x, dev_y = data_x[dev_index], data_y[dev_index]
            test_x, test_y = data_x[test_index], data_y[test_index]
            self.set_train_data([train_x, train_y])
            train_index = align_batch_size(range(len(train_y)), batch_size)
            train_x_length = np.sum((train_x > 0), axis=1)
            num_batch = len(train_index) / batch_size
            batch_list = range(num_batch)
            log_loss_history, acc_history = list(), list()
            batch_log_loss_history, batch_acc_history = list(), list()
            logger.info("start training")
            batch_count = 0
            for i in xrange(max_iter):
                iter_loss_list = list()
                iter_acc_list = list()
                batch_list = np.random.permutation(batch_list)
                for j in batch_list:
                    set_dropout_on(True)
                    batch_count += 1
                    indexs = train_index[j * batch_size: (j + 1) * batch_size]
                    max_len = np.max(train_x_length[indexs])
                    self.train_batch(indexs, max_len)
                    if batch_count % pre_test_batch == 0:
                        set_dropout_on(False)
                        dev_acc, dev_log_loss = self.predict_data_log_loss_acc(dev_x, dev_y, test_batch_size)
                        logger.info("cv %d batch %d,   dev log loss %s, acc %s"
                                    % (cv_i, batch_count, dev_log_loss, dev_acc))
                        test_acc, test_log_loss = self.predict_data_log_loss_acc(test_x, test_y, test_batch_size)
                        logger.info("cv %d batch %d,  test log loss %s, acc %s"
                                    % (cv_i, batch_count, test_log_loss, test_acc))
                        batch_log_loss_history.append([batch_count, dev_log_loss, test_log_loss, ])
                        batch_acc_history.append([batch_count, dev_acc, test_acc])
                set_dropout_on(False)
                train_acc, train_log_loss = self.predict_data_log_loss_acc(train_x, train_y, test_batch_size)
                iter_loss_list.append(train_log_loss)
                iter_acc_list.append(train_acc)
                iter_l2_loss, iter_l2_norm = self.get_l2_loss()
                logger.info("cv %d epoch %d, param l2 losss %s, l2 norm %s" % (cv_i, i, iter_l2_loss, iter_l2_norm))
                logger.info("cv %d epoch %d, train log loss %s, acc %s" % (cv_i, i, train_log_loss, train_acc))

                dev_acc, dev_log_loss = self.predict_data_log_loss_acc(dev_x, dev_y, test_batch_size)
                logger.info("cv %d epoch %d,   dev log loss %s, acc %s" % (cv_i, i, dev_log_loss, dev_acc))
                iter_loss_list.append(dev_log_loss)
                iter_acc_list.append(dev_acc)

                test_acc, test_log_loss = self.predict_data_log_loss_acc(test_x, test_y, test_batch_size)
                logger.info("cv %d epoch %d,   test log loss %s, acc %s" % (cv_i, i, test_log_loss, test_acc))
                iter_loss_list.append(test_log_loss)
                iter_acc_list.append(test_acc)

                log_loss_history.append(iter_loss_list)
                acc_history.append(iter_acc_list)

            # Log Best Epoch
            log_loss_history = np.array(log_loss_history)
            acc_history = np.array(acc_history)

            # Log Best Batch
            batch_log_loss_history = np.array(batch_log_loss_history)
            batch_acc_history = np.array(batch_acc_history)

            self.log_to_file("Epoch", log_loss_history, acc_history, cv_iter=cv_i)
            self.log_to_file("Batch", batch_log_loss_history, batch_acc_history, cv_iter=cv_i)

            # record best epoch
            best_loss_epoch = np.argmin(log_loss_history[:, 1])
            best_acc_epoch = np.argmax(acc_history[:, 1])
            cv_epoch_acc.append([log_loss_history[best_acc_epoch, 2], acc_history[best_acc_epoch, 2]])
            cv_epoch_log_loss.append([log_loss_history[best_loss_epoch, 2], acc_history[best_loss_epoch, 2]])

            # record best batch
            best_loss_batch = np.argmin(batch_log_loss_history[:, 1])
            best_acc_batch = np.argmax(batch_acc_history[:, 1])
            cv_batch_acc.append([batch_log_loss_history[best_acc_batch, 2],
                                 batch_acc_history[best_acc_batch, 2]])
            cv_batch_log_loss.append([batch_log_loss_history[best_loss_batch, 2],
                                      batch_acc_history[best_loss_batch, 2]])

        cv_epoch_acc = np.array(cv_epoch_acc)
        cv_epoch_log_loss = np.array(cv_epoch_log_loss)
        logger.info("%s Times CV Best Epoch Dev Acc test  log loss %s, acc %s"
                    % (cv_times, np.mean(cv_epoch_acc[:, 0]), np.mean(cv_epoch_acc[:, 1])))
        logger.info("%s Times CV Best Epoch Dev Log Loss test  log loss %s, acc %s"
                    % (cv_times, np.mean(cv_epoch_log_loss[:, 0]), np.mean(cv_epoch_log_loss[:, 1])))

    def train(self, train, dev=None, test=None, to_predict=None, max_iter=5, batch_size=128, test_batch_size=1000,
              pre_test_batch=25, predict_path=None):
        train_x, train_y = train
        self.set_train_data(train)
        train_index = align_batch_size(range(len(train_y)), batch_size)
        train_x_length = np.sum((train_x > 0), axis=1)
        num_batch = len(train_index) / batch_size
        batch_list = range(num_batch)
        log_loss_history, acc_history = list(), list()
        batch_log_loss_history, batch_acc_history = list(), list()
        logger.info("start training")
        batch_count = 0
        best_dev_acc = 0
        for i in xrange(max_iter):
            iter_loss_list = list()
            iter_acc_list = list()
            batch_list = np.random.permutation(batch_list)
            for j in batch_list:
                set_dropout_on(True)
                batch_count += 1
                indexs = train_index[j * batch_size: (j + 1) * batch_size]
                max_len = np.max(train_x_length[indexs])
                self.train_batch(indexs, max_len)
                if batch_count % pre_test_batch == 0:
                    set_dropout_on(False)
                    batch_log_loss, batch_acc = [batch_count], [batch_count]
                    if dev is not None:
                        dev_x, dev_y = dev
                        dev_acc, dev_log_loss = self.predict_data_log_loss_acc(dev_x, dev_y, test_batch_size)
                        batch_log_loss.append(dev_log_loss)
                        batch_acc.append(dev_acc)
                        if dev_acc > best_dev_acc:
                            best_dev_acc = dev_acc
                            save_model("model/%s.best.model" % predict_path, self)
                        logger.info("batch %d,   dev log loss %s, acc %s" % (batch_count, dev_log_loss, dev_acc))
                    if test is not None:
                        test_x, test_y = test
                        test_acc, test_log_loss = self.predict_data_log_loss_acc(test_x, test_y, test_batch_size)
                        batch_log_loss.append(test_log_loss)
                        batch_acc.append(test_acc)
                        logger.info("batch %d,  test log loss %s, acc %s" % (batch_count, test_log_loss, test_acc))
                    batch_log_loss_history.append(batch_log_loss)
                    batch_acc_history.append(batch_acc)
            set_dropout_on(False)
            train_acc, train_log_loss = self.predict_data_log_loss_acc(train_x, train_y, test_batch_size)
            iter_loss_list.append(train_log_loss)
            iter_acc_list.append(train_acc)
            iter_l2_loss, iter_l2_norm = self.get_l2_loss()
            logger.info("epoch %d, param l2 losss %s, l2 norm %s" % (i, iter_l2_loss, iter_l2_norm))
            logger.info("epoch %d, train log loss %s, acc %s" % (i, train_log_loss, train_acc))
            if dev is not None:
                dev_x, dev_y = dev
                dev_acc, dev_log_loss = self.predict_data_log_loss_acc(dev_x, dev_y, test_batch_size)
                logger.info("epoch %d,   dev log loss %s, acc %s" % (i, dev_log_loss, dev_acc))
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    save_model("model/%s.best.model" % predict_path, self)
                iter_loss_list.append(dev_log_loss)
                iter_acc_list.append(dev_acc)
            if test is not None:
                test_x, test_y = test
                test_acc, test_log_loss = self.predict_data_log_loss_acc(test_x, test_y, test_batch_size)
                logger.info("epoch %d,  test log loss %s, acc %s" % (i, test_log_loss, test_acc))
                iter_loss_list.append(test_log_loss)
                iter_acc_list.append(test_acc)
            log_loss_history.append(iter_loss_list)
            acc_history.append(iter_acc_list)

        # Log Best Epoch
        log_loss_history = np.array(log_loss_history)
        acc_history = np.array(acc_history)

        # Log Best Batch
        batch_log_loss_history = np.array(batch_log_loss_history)
        batch_acc_history = np.array(batch_acc_history)
        self.log_to_file("Epoch", log_loss_history, acc_history)
        self.log_to_file("Batch", batch_log_loss_history, batch_acc_history)
        save_model("model/%s.final.model" % predict_path, self)

    @staticmethod
    def log_to_file(name, log_loss_hist, acc_hist, cv_iter=None):
        """
        :param name: Iter Name (Epoch/Batch)
        :param log_loss_hist: log loss history
        :param acc_hist: acc history
        :param cv_iter: cross validation iter
        :return:
        """
        if log_loss_hist.shape[0] == 0 or acc_hist.shape[0] == 0:
            return
        best_loss_index = np.argmin(log_loss_hist[:, 1])
        best_acc_index = np.argmax(acc_hist[:, 1])
        prefix = "Best Dev" if cv_iter is None else "CV %d Best Dev" % cv_iter
        if name != 'Batch':
            # Batch is too much
            logger.info("%s Log Loss %s %s, train log loss %s, acc %s"
                        % (prefix, name, best_loss_index,
                           log_loss_hist[best_loss_index, 0], acc_hist[best_loss_index, 0]))
        logger.info("%s Log Loss %s %s, dev   log loss %s, acc %s"
                    % (prefix, name, best_loss_index,
                       log_loss_hist[best_loss_index, 1], acc_hist[best_loss_index, 1]))
        logger.info("%s Log Loss % s%s, test  log loss %s, acc %s"
                    % (prefix, name, best_loss_index,
                       log_loss_hist[best_loss_index, 2], acc_hist[best_loss_index, 2]))
        if name != 'Batch':
            # Batch is too much
            logger.info("%s Acc %s %s, train log loss %s, acc %s"
                        % (prefix, name, best_acc_index,
                           log_loss_hist[best_acc_index, 0], acc_hist[best_acc_index, 0]))
        logger.info("%s Acc %s %s, dev   log loss %s, acc %s"
                    % (prefix, name, best_acc_index,
                       log_loss_hist[best_acc_index, 1], acc_hist[best_acc_index, 1]))
        logger.info("%s Acc %s %s, test  log loss %s, acc %s"
                    % (prefix, name, best_acc_index,
                       log_loss_hist[best_acc_index, 2], acc_hist[best_acc_index, 2]))


def text_classify_main(train_file, dev_file=None, test_file=None, embedding_file=None,
                       encoder='lstm', word_dim=300, hidden_dim='168', dropout=0.5, batch_size=128,
                       optimizer='adadelta', lr=0.95, predict_name=None, norm_limit=-1, regularization_weight=0.0001,
                       split_symbol='\t'):
    # Load Dictionary
    train_d = load_sentence_dict(train_file, max_len=MAX_LEN, low_case=LOW_CASE,
                                 remove_stop=REMOVE_STOP, split_symbol=split_symbol)
    dictionary = train_d
    if dev_file is not None:
        dev_d = load_sentence_dict(dev_file, max_len=MAX_LEN, low_case=LOW_CASE,
                                   remove_stop=REMOVE_STOP, split_symbol=split_symbol)
        dictionary = dictionary + dev_d
    if test_file is not None:
        test_d = load_sentence_dict(test_file, max_len=MAX_LEN, low_case=LOW_CASE,
                                    remove_stop=REMOVE_STOP, split_symbol=split_symbol)
        dictionary = dictionary + test_d
    dictionary.cut_by_top(TOPK)
    # Load Data
    train_y, train_x = load_sentence(train_file, dictionary, max_len=MAX_LEN, low_case=LOW_CASE,
                                     remove_stop=REMOVE_STOP, split_symbol=split_symbol)
    label_num = np.max(train_y) + 1
    if dev_file is not None:
        dev_y, dev_x = load_sentence(dev_file, dictionary, max_len=MAX_LEN, low_case=LOW_CASE,
                                     remove_stop=REMOVE_STOP, split_symbol=split_symbol)
    else:
        # No Dev File, Random Select Dev from Train Data
        train_data_index = np.random.permutation(np.arange(train_x.shape[0]))
        split_num = int(len(train_x) * (1 - DEV_RATIO))
        train_index = train_data_index[:split_num]
        dev_index = train_data_index[split_num:]
        dev_x, dev_y = train_x[dev_index, :], train_y[dev_index]
        train_x, train_y = train_x[train_index, :], train_y[train_index]
    if test_file is not None:
        test_y, test_x = load_sentence(test_file, dictionary, max_len=MAX_LEN, low_case=LOW_CASE,
                                       remove_stop=REMOVE_STOP, split_symbol=split_symbol)
    else:
        test_x, test_y = None, None
    # Define Classifer
    classifier = TextClassifier(dictionary.word_index, pretrain_name=embedding_file, word_dim=word_dim,
                                hidden=hidden_dim, label_num=label_num,
                                dropout=dropout, encoder=encoder, optimizer_name=optimizer, lr=lr,
                                norm_lim=norm_limit, regularization_weight=regularization_weight,
                                label2index_filename=LABEL2INDEX)
    if CROSS_VALIDATION_TIMES > 1:
        classifier.cross_validation_train([train_x, train_y], cv_times=CROSS_VALIDATION_TIMES, batch_size=batch_size,
                                          max_iter=MAX_ITER, pre_test_batch=PRE_TEST_BATCH, model_path=predict_name)
    else:
        # Train Classifier
        classifier.train([train_x, train_y],
                         dev=[dev_x, dev_y],
                         test=[test_x, test_y],
                         batch_size=batch_size, predict_path=predict_name,
                         max_iter=MAX_ITER,
                         pre_test_batch=PRE_TEST_BATCH,
                         )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='lstm', help='Encoder Name: cbow lstm bilstm cnn gru bigru '
                                                                    'recurrent birecurrent, Default: lstm')
    parser.add_argument('--seed', type=int, default=1993, help='Random Seed, Default: 1993')
    parser.add_argument('--batch', type=int, default=25, help='Batch Size, Default: 25')
    parser.add_argument('--word', type=int, default=300, help='Word Dim, Default: 300')
    parser.add_argument('--hidden', type=str, default='168', help='Hidden Dim for Encoder/Hidden Layer,'
                                                                  'Default: `168` Encoder(168) and No Hidden Layer,'
                                                                  '168_168_50 for Encoder(168) Hidden Layer (168, 50),')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Rate, Default: 0.5')
    parser.add_argument('--pre', type=str, default="google", help='Pre-Train Embedding File: google glove '
                                                                  'random(no pre), '
                                                                  'Default: Google')
    parser.add_argument('--optimizer', type=str, default="adadelta",
                        help='Optimizer: sgd adagrad adadelta momentum adam, Default: adadelta')
    parser.add_argument('--regular', type=float, default=0.0001, help='Regularization Weight, Default: 10e-4')
    parser.add_argument('--conv', type=str, default="3_4_5", help='Filter Size for Conv, Default: 3_4_5')
    parser.add_argument('--norm', type=int, default=-1, dest='norm_limit', help='L2 Norm Limit for Clip, Default: -1')
    parser.add_argument('--lr', type=float, default=0.95, help='Learning Rate, Default: 0.95')
    parser.add_argument('--emblr', type=float, default=0.1, help='Embedding Learning Rate(SGD), Default: 0.1, '
                                                                 'less than 0 mean using global optimizer')
    parser.add_argument('--train', type=str, default=None, help='Train File, Default: train.data')
    parser.add_argument('--dev', type=str, default=None, help='Dev File, Default: dev.data')
    parser.add_argument('--test', type=str, default=None, help='Test File, Default: test.data')
    parser.add_argument('--epoch', type=int, default=10, help='Max Num Epoch, Default: 10')
    parser.add_argument('--test-batch', type=int, default=1000, dest='test_batch', help='Test Pre Batch Size, '
                                                                                        'Default: 1000')
    parser.add_argument('--cross', type=int, default=0, dest='cross', help='Cross Validation Times, Default: 5')
    parser.add_argument('--prefix', type=str, default=None, dest='prefix', help='Model Name Prefix')
    parser.add_argument('--label', type=str, default=None, dest='label', help='Label Index File, Default is None. '
                                                                               'Each line is "label\tindex"')

    args = parser.parse_args()
    if args.pre == 'google':
        pre_embedding_file = "GoogleNews-vectors-negative300.bin"
    elif args.pre == 'glove':
        pre_embedding_file = "Glove.300.bin"
    elif args.pre == 'random':
        pre_embedding_file = None
    else:
        pre_embedding_file = args.pre
    model_name = "%s_seed%s_drop%s_batch%s_word%s_hidden%s_%s%s" % (args.encoder, args.seed, args.dropout, args.batch,
                                                                    args.word, args.hidden, args.optimizer, args.lr)
    if args.conv != "3_4_5":
        CONV_FILTER_SIZES = [int(v) for v in args.conv.split('_')]
        model_name += args.conv
    if args.pre is not None:
        if os.sep in args.pre:
            model_name += "_%s" % args.pre.split(os.sep)[-1]
        else:
            model_name += "_%s" % args.pre
    else:
        model_name += "_random"
    if args.norm_limit > 0:
        model_name += "_nlimit%s" % args.norm_limit
    if args.regular > 0:
        model_name += "_regular%s" % args.regular
    if args.emblr < 0:
        EMBEDDING_LR = -1
    elif args.emblr != EMBEDDING_LR:
        model_name += "_emblr%s" % args.emblr
    else:
        model_name += "_emblr%s" % EMBEDDING_LR
    if args.cross > 1:
        model_name += "_cross%s" % args.cross
        CROSS_VALIDATION_TIMES = args.cross
    if args.prefix is not None:
        model_name = "%s_%s" % (args.prefix, model_name)
    pre_logger(model_name)
    np.random.seed(args.seed)

    if args.epoch != MAX_ITER:
        MAX_ITER = args.epoch
        logger.info("Max Epoch Iter: %s" % MAX_ITER)
    if args.emblr != EMBEDDING_LR:
        EMBEDDING_LR = args.emblr
        logger.info("Embedding Learning Rate: %s" % EMBEDDING_LR)
    if args.test_batch != PRE_TEST_BATCH:
        PRE_TEST_BATCH = args.test_batch
    if args.label is not None:
        LABEL2INDEX = args.label

    if CROSS_VALIDATION_TIMES > 1:
        args.dev = None
        args.test = None
    text_classify_main(train_file=args.train, dev_file=args.dev, test_file=args.test,
                       embedding_file=pre_embedding_file, encoder=args.encoder, word_dim=args.word,
                       hidden_dim=args.hidden, dropout=args.dropout, batch_size=args.batch,
                       predict_name=model_name, optimizer=args.optimizer, lr=args.lr, norm_limit=args.norm_limit,
                       regularization_weight=args.regular)
