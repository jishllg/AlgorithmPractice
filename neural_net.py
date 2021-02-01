import numpy

class Neural_Net(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
        self.weights = [np.random.randn(i, j) for j, i in sizes[1:]]