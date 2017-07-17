import logging
from abc import abstractmethod, ABCMeta
from collections import OrderedDict

import theano.tensor as T

from utils import shared_zero_matrix, shared_scalar

__author__ = 'roger'
logger = logging.getLogger(__name__)


class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, lr, norm_lim=9, verbose=True):
        self.lr = lr
        self.norm_lim = norm_lim
        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Init LR:    %s' % self.lr)
            if self.norm_lim is None:
                logger.debug('No Norm Limit')
            else:
                logger.debug('Norm Limit: %s' % self.norm_lim)

    @staticmethod
    def get_grad(loss, params):
        """
        get param's grad in params
        :param loss:
        :param params: list/tuple
        :return:
        """
        grad_params = []
        for param in params:
            gp = T.grad(loss, param)
            grad_params.append(gp)
        return grad_params

    @staticmethod
    def get_params_str(params):
        param_str_list = list()
        for param in params:
            param_str_list.append("%s%s" % (param.name, param.get_value().shape))
        return "[%s]" % (", ".join(param_str_list))

    def get_update(self, loss, params, norm_exc_params=None):
        """
        :param loss:
        :param params:
        :param norm_exc_params: the name list of the params without norm
        :return:
        """
        logger.info("Update Parameters: %s" % self.get_params_str(params))
        grad_params = self.get_grad(loss, params=params)
        param_updates, optimizer_updates = self.get_iter(params, grad_params)
        if self.norm_lim > 0:
            param_updates = self.norm_limit_param(param_updates, norm_exc_params)
        updates = param_updates.copy()
        updates.update(optimizer_updates)
        return updates

    @abstractmethod
    def get_iter(self, params, grad_params):
        """

        :param params:              list/tuple
        :param grad_params:         list/tuple
        :return: param_updates      OrderedDict({}) key: param, value: update value
                 optimizer_updates  OrderedDict({}) key: acc in optmizer, value: update value
        """
        return OrderedDict({}), OrderedDict({})

    def norm_limit_param(self, param_update, norm_except_params=None):
        """
        :param param_update: OrderedDict({}) key: param, value: update value
        :param norm_except_params:
        :return:
        """
        updates = OrderedDict({})
        for param, stepped_param in param_update.iteritems():
            param_name = param.name
            if self.norm_lim > 0 \
                    and (param.get_value(borrow=True).ndim == 2) \
                    and norm_except_params is not None \
                    and (param_name not in norm_except_params):
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(self.norm_lim))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param
        return updates


class SGDOptimizer(Optimizer):
    def __init__(self, lr=0.05, norm_lim=-1):
        super(SGDOptimizer, self).__init__(lr, norm_lim)

    def get_iter(self, params, grad_params):
        """
        get params' update values
        :param params: list/ tuple
        :param grad_params: list/ tuple
        :return: param_updates      OrderedDict({}) key: param, value: update value
                 optimizer_updates  OrderedDict({}) key: acc in optmizer, value: update value
        """
        param_updates = OrderedDict({})
        optimizer_updates = OrderedDict({})
        for param, gp in zip(params, grad_params):
            stepped_param = param - gp * self.lr
            param_updates[param] = stepped_param
        return param_updates, optimizer_updates


class SGDMomentumOptimizer(Optimizer):

    def __init__(self, lr=0.05, momentum=0.9, norm_lim=9):
        super(SGDMomentumOptimizer, self).__init__(lr, norm_lim)
        self.momentum = momentum
        logger.debug('Momentum: %s' % self.momentum)

    def get_iter(self, params, grad_params):
        """
        get params' update values
        Optmization Section in DEEP LEARNING book
        :param params: list/ tuple
        :param grad_params: list/ tuple
        :return: param_updates      OrderedDict({}) key: param, value: update value
                 optimizer_updates  OrderedDict({}) key: acc in optmizer, value: update value
        """
        params_updates = OrderedDict({})
        optimizer_updates = OrderedDict({})
        velocity = OrderedDict({})
        for param in params:
            velocity[param] = shared_zero_matrix(param.get_value().shape, name="vel_%s" % param.name)
        for param, gp in zip(params, grad_params):
            vel_para = velocity[param]
            up_vel_para = self.momentum * vel_para - self.lr * gp
            step = up_vel_para
            stepped_param = param + step
            optimizer_updates[vel_para] = up_vel_para
            params_updates[param] = stepped_param
        return params_updates, optimizer_updates


class AdaGradOptimizer(Optimizer):
    def __init__(self, lr=0.95, norm_lim=9, epsilon=1e-7):
        super(AdaGradOptimizer, self).__init__(lr, norm_lim)
        self.epsilon = epsilon

    def get_iter(self, params, grad_params):
        """
        get params' update values
        Optmization Section in DEEP LEARNING book
        :param params: list/ tuple
        :param grad_params: list/ tuple
        :return: param_updates      OrderedDict({}) key: param, value: update value
                 optimizer_updates  OrderedDict({}) key: acc in optmizer, value: update value
        """
        params_updates = OrderedDict({})
        optimizer_updates = OrderedDict({})
        accumulators = OrderedDict({})
        for param in params:
            accumulators[param] = shared_zero_matrix(param.get_value().shape, name="acc_%s" % param.name)
        for param, gp in zip(params, grad_params):
            exp_sr = accumulators[param]
            up_exp_sr = exp_sr + T.sqr(gp)
            step = (self.lr / (T.sqrt(up_exp_sr) + self.epsilon)) * gp
            stepped_param = param - step
            optimizer_updates[exp_sr] = up_exp_sr
            params_updates[param] = stepped_param
        return params_updates, optimizer_updates


class AdaDeltaOptimizer(Optimizer):
    """
    https://arxiv.org/abs/1212.5701
    """
    def __init__(self, lr=1, decay_rate=0.95, norm_lim=16, epsilon=1e-7):
        super(AdaDeltaOptimizer, self).__init__(lr, norm_lim)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        logger.debug('Decay Rate: %s' % self.decay_rate)

    def get_iter(self, params, grad_params):
        """
        get params' update values
        :param params: list/ tuple
        :param grad_params: list/ tuple
        :return: param_updates      OrderedDict({}) key: param, value: update value
                 optimizer_updates  OrderedDict({}) key: acc in optmizer, value: update value
        """
        params_updates = OrderedDict({})
        optimizer_updates = OrderedDict({})
        rho = self.decay_rate
        epsilon = self.epsilon
        exp_sqr_grads = OrderedDict({})
        exp_sqr_ups = OrderedDict({})
        for param in params:
            exp_sqr_grads[param] = shared_zero_matrix(param.get_value().shape, name="exp_grad_%s" % param.name)
            exp_sqr_ups[param] = shared_zero_matrix(param.get_value().shape, name="exp_ups_%s" % param.name)
        for param, gp in zip(params, grad_params):
            exp_sg = exp_sqr_grads[param]
            exp_su = exp_sqr_ups[param]
            up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
            step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp * self.lr
            stepped_param = param + step
            optimizer_updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
            optimizer_updates[exp_sg] = up_exp_sg
            params_updates[param] = stepped_param
        return params_updates, optimizer_updates


class AdamOptimizer(Optimizer):
    """
    Adaptive Moments
    https://arxiv.org/pdf/1412.6980.pdf
    """

    def __init__(self, lr=0.001, first_decay_rate=0.9, second_decay_rate=0.999, norm_lim=16, epsilon=1e-8):
        super(AdamOptimizer, self).__init__(lr, norm_lim)
        self.first_decay_rate = first_decay_rate
        self.second_decay_rate = second_decay_rate
        self.epsilon = epsilon
        logger.debug('First Decay Rate: %s' % self.first_decay_rate)
        logger.debug('Second Decay Rate: %s' % self.second_decay_rate)

    def get_iter(self, params, grad_params):
        """
        get params' update values
        :param params: list/ tuple
        :param grad_params: list/ tuple
        :return: param_updates      OrderedDict({}) key: param, value: update value
                 optimizer_updates  OrderedDict({}) key: acc in optmizer, value: update value
        """
        params_updates = OrderedDict({})
        optimizer_updates = OrderedDict({})
        epsilon = self.epsilon
        first_moment_bias = OrderedDict({})
        second_moment_bias = OrderedDict({})
        rho1 = self.first_decay_rate
        rho2 = self.second_decay_rate
        acc_rho1 = shared_scalar(value=rho1, name="adam_acc_rho1")
        acc_rho2 = shared_scalar(value=rho2, name="adam_acc_rho2")
        for param in params:
            first_moment_bias[param] = shared_zero_matrix(param.get_value().shape, name="fir_mom_bias%s" % param.name)
            second_moment_bias[param] = shared_zero_matrix(param.get_value().shape, name="sec_mom_bias%s" % param.name)
        for param, gp in zip(params, grad_params):
            first_mb = first_moment_bias[param]
            second_mb = second_moment_bias[param]
            # Update biased first moment estimate
            up_first_mb = rho1 * first_mb + (1 - rho1) * gp
            # Update biased second moment estimate
            up_second_mb = rho2 * second_mb + (1 - rho2) * T.sqr(gp)
            # Correct bias in first moment
            correct_first_mb = up_first_mb / (1 - acc_rho1)
            # Correct bias in second moment
            correct_second_mb = up_second_mb / (1 - acc_rho2)
            # Compute step
            step = correct_first_mb / (T.sqrt(correct_second_mb) + epsilon)
            # Apply Update
            stepped_param = param - step * self.lr
            optimizer_updates[first_mb] = up_first_mb
            optimizer_updates[second_mb] = up_second_mb
            optimizer_updates[acc_rho1] = acc_rho1 * rho1
            optimizer_updates[acc_rho2] = acc_rho2 * rho2
            params_updates[param] = stepped_param
        return params_updates, optimizer_updates
