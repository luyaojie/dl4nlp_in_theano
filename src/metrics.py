import numpy as np


def prob_to_accuracy(y_true, y_prob):
    pred_label = np.argmax(y_prob, axis=1)
    return accuracy_score(y_true=y_true, y_pred=pred_label)


def prob_to_log_loss(y_true, y_prob, labels=None):
    return log_loss(y_true=y_true, y_pred_prob=y_prob, labels=labels)


def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    import sklearn.metrics
    return sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred,
                                          normalize=normalize, sample_weight=sample_weight)


def log_loss(y_true, y_pred_prob, normalize=True, sample_weight=None, labels=None):
    """
    :param y_true: array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.
    :param y_pred_prob: array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.
    :param normalize: bool, optional (default=True)
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.
    :param sample_weight: array-like of shape = [n_samples], optional
        Sample weights.
    :param labels : array-like, optional (default=None)
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.
        .. versionadded:: 0.18
    :return:
    """
    import sklearn.metrics
    return sklearn.metrics.log_loss(y_true=y_true, y_pred=y_pred_prob, normalize=normalize,
                                    sample_weight=sample_weight, labels=labels)
