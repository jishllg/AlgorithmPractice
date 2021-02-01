import numpy as np

# Neural net class definition
class Neural_Net(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
        self.weights = [np.random.randn(i, j) for j, i in zip(sizes[:-1], sizes[1:])]

# Processes through the network layer by layer
    def feed_forward(self, activation):
        for biases, weights in zip(self.biases, self.weights):
            activation = sigmoid(np.dot(weights, activation) + biases)
        return activation

# Calibrates the function using stochastic gradient descent
    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[j:j + mini_batch_size]
                for j in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch %(epoch)s: %(correct)s / %(total)s." %
                        {'epoch': i, 'correct': self.evaluate(test_data), 'total': n_test})

    def update_mini_batch(self, mini_batch, eta):
        del_biases = [np.zeros(bias.shape) for bias in self.biases]
        del_weights = [np.zeros(weight.shape) for weight in self.weights]
        for x, y in mini_batch:
            delta_del_biases, delta_del_weights = self.backprop(x, y)
            del_biases = [db + ddb for db, ddb in zip(del_biases, delta_del_biases)]
            del_weights = [dw + ddw for dw, ddw in zip(del_weights, delta_del_weights)]
        self.biases = [bias - (eta / len(mini_batch)) * del_bias
                        for bias, del_bias in zip(self.biases, del_biases)]
        self.weights = [weight - (eta / len(mini_batch)) * del_weight
                        for weight, del_weight in zip(self.weights, del_weights)]

# Sigmoid link function
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))



net = Neural_Net([5, 5, 5])

print(net.weights)