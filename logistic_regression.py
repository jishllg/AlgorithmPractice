import numpy as np
import matplotlib.pyplot as plt

# Setting generation parameters
seed = 15
np.random.seed(seed)
num_observations = 5000
params = str(seed) + "_" + str(num_observations)

# Generating data
x1 = np.random.multivariate_normal([0, 0], [[1, 0.75],[0.75, 1]], 
                                    num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75],[0.75, 1]], 
                                    num_observations)

# Compiling data into arrays
simulated_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations), 
                            np.ones(num_observations)))

# Plotting generated data
plt.figure(figsize = (12, 8))
plt.scatter((simulated_features[:, 0]), simulated_features[:, 1], 
            c = simulated_labels, alpha = 0.4)
plt.show()

# Sigmoid link function
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

# Log Likelihood equation
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
    return ll

# The real deal
def logistic_regression(features, target, num_steps, learning_rate, 
                        add_intercept = False, print_step = 10000):

    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
    
    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log likelihood every so often
        if step % print_step == 0:
            print(log_likelihood(features, target, weights))

    return weights

# Load weights from file if they exist
try:
    weights = np.genfromtxt("weights_" + params + ".csv", delimiter=",")
    print("Weights imported.")

# Running the model and saving weights
except:
    print("Weights file not found. Generating weights. . .")
    weights = logistic_regression(simulated_features, simulated_labels,
                num_steps = 300000, learning_rate = 5e-5, add_intercept = True)
    np.savetxt("weights_" + params + ".csv", weights, delimiter=",")

# Checking accuracy
data_with_intercept = np.hstack((np.ones((simulated_features.shape[0], 1)),
                                 simulated_features))
final_scores = np.dot (data_with_intercept, weights)
predictions = np.round(sigmoid(final_scores))
correct_predictions = 0
for i in range(len(predictions)):
    if predictions[i] == simulated_labels[i]:
        correct_predictions += 1
accuracy = correct_predictions / len(predictions)
print("Algorithm Accuracy:", accuracy)

# Plotting data by accuracy
plt.figure(figsize = (12, 8))
plt.scatter((simulated_features[:, 0]), simulated_features[:, 1], 
            c = predictions == simulated_labels - 1, alpha = 0.8, s = 50)
plt.show()