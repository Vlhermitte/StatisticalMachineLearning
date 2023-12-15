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

    def softmax(self, X):
        # Numerically stable softmax
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X, T):
        # TODO IMPLEMENT
        # Compute softmax probabilities
        softmax_probs = self.softmax(X)
        # Compute cross-entropy loss
        cross_entropy_loss = -np.sum(T * np.log(softmax_probs + 1e-15)) / X.shape[0]
        return cross_entropy_loss

    def delta(self, X, T):
        # TODO IMPLEMENT (use numerically stable version)
        softmax_probs = self.softmax(X)
        return softmax_probs - T
