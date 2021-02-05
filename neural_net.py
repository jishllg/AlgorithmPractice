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

# Creates mini batches to iterate through for backpropagation
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

# Backpropagation algorithm
    def backprop(self, x, y):
        del_biases = [np.zeros(bias.shape) for bias in self.biases]
        del_weights = [np.zeros(bias.weight) for weight in self.weights]
        activation = x
        activations = [x]
        zs = []
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        del_biases[-1] = delta
        del_weights[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            del_biases[-i] = delta
            del_weights[-i] = np.dot(delta, activations[-i - 1].transpose())
        return(del_biases, del_weights)

# Calculate vector of partial derivatives for the output activations
    def cost_derivative(self, output_activations, y):
        return(output_activations - y)

# Sigmoid link function
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

# Sigmoid link function derivative
def sigmoid_prime(scores):
    return sigmoid(scores) * (1 - sigmoid(scores))
