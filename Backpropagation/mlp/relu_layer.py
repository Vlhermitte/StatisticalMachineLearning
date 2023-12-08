import numpy as np
class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        # TODO IMPLEMENT
        return np.maximum(X, 0)

    def delta(self, Y, delta_next):
        # TODO IMPLEMENT
        return delta_next * (Y > 0)
