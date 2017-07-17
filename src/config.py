import ConfigParser

__author__ = 'roger'


class BRAEConfig(object):
    """
    Class for the configuration of BRAE (JiaJun Zhang et al., 2014 ACL)
    """
    def __init__(self, filename):
        self._cf_parser = ConfigParser.ConfigParser()
        self._cf_parser.read(filename)
        (self.activation, self.dim, self.normalize, self.weight_rec, self.weight_sem,
         self.weight_l2, self.alpha, self.n_epoch, self.batch_size,
         self.dropout, self.random_seed, self.min_count) = self.parse()
        self.optimizer = OptimizerConfig(filename)

    def parse(self):
        activation = self._cf_parser.get("functions", "activation")

        dim = self._cf_parser.getint("architectures", "dim")
        normalize = self._cf_parser.getboolean("architectures", "normalize")
        weight_rec = self._cf_parser.getfloat("architectures", "weight_rec")
        weight_sem = self._cf_parser.getfloat("architectures", "weight_sem")
        weight_l2 = self._cf_parser.getfloat("architectures", "weight_l2")
        alpha = self._cf_parser.getfloat("architectures", "alpha")

        n_epoch = self._cf_parser.getint("parameters", "n_epoch")
        batch_size = self._cf_parser.getint("parameters", "batch_size")
        dropout = self._cf_parser.getfloat("parameters", "dropout")
        random_seed = self._cf_parser.getint("parameters", "random_seed")
        min_count = self._cf_parser.getfloat("parameters", "min_count")

        return (activation, dim, normalize, weight_rec, weight_sem, weight_l2, alpha,
                n_epoch, batch_size, dropout, random_seed, min_count)


class BLERAEConfig(BRAEConfig):
    def __init__(self, filename):
        super(BLERAEConfig, self).__init__(filename)
        self.beta = self._cf_parser.getfloat("architectures", "beta")


class BRAEISOMAPConfig(BRAEConfig):
    def __init__(self, filename):
        super(BRAEISOMAPConfig, self).__init__(filename)
        self.beta = self._cf_parser.getfloat("architectures", "beta")
        self.trans_num = self._cf_parser.getint("architectures", "trans_num")


class GBRAEConfig(BRAEConfig):
    def __init__(self, filename):
        super(GBRAEConfig, self).__init__(filename)
        self.beta = self._cf_parser.getfloat("architectures", "beta")
        self.gama = self._cf_parser.getfloat("architectures", "gama")
        self.delta = self._cf_parser.getfloat("architectures", "delta")
        self.trans_num = self._cf_parser.getint("architectures", "trans_num")


class OptimizerConfig(object):
    """
    Class for the configuration of Optimizer
    """
    def __init__(self, filename):
        self._cf_parser = ConfigParser.ConfigParser()
        self._cf_parser.read(filename)
        self.name, self.param = self.parse()

    def parse(self):
        name = self._cf_parser.get("optimizer", "optimizer")
        opt_param = self.get_opt_param(name)
        return name, opt_param

    def get_opt_param(self, optimizer):
        param = dict()
        if optimizer.lower() == "sgd":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
        elif optimizer.lower() == "sgdmomentum":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
            param["momentum"] = self._cf_parser.getfloat("optimizer", "momentum")
        elif optimizer.lower() == "adagrad":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
        elif optimizer.lower() == "adadelata":
            param["decay_rate"] = self._cf_parser.getfloat("optimizer", "decay_rate")
        else:
            raise NotImplementedError
        return param
