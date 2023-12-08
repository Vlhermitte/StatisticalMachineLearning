import numpy as np
class LossCrossEntropy(object):
    def __init__(self, name):
        super(LossCrossEntropy, self).__init__()
        self.name = name

    def forward(self, X, T):
        """
        Forward message. (multinominal cross-entropy)
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, 1)
        """
        # TODO IMPLEMENT
        return -np.sum(T * np.log(X)) / X.shape[0]


    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: delta vector from the loss layer, shape (n_samples, n_inputs)
        """
        # TODO IMPLEMENT
        return -T / X


class LossCrossEntropyForSoftmaxLogits(object):
    def __init__(self, name):
        super(LossCrossEntropyForSoftmaxLogits, self).__init__()
        self.name = name

    def forward(self, X, T):
        # TODO IMPLEMENT
        return -np.sum(X * T) / X.shape[0]

    def delta(self, X, T):
        # TODO IMPLEMENT
        return X - T
