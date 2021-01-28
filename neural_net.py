import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Setting generation parameters
seed = 12
np.random.seed(seed)
num_observations = 5000
params = str(seed) + "_" + str(num_observations)

# Generating data
x1 = np.random.multivariate_normal([0, 0], [[2, 0.75],[0.75, 2]], 
                                    num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75],[0.75, 1]], 
                                    num_observations)
x3 = np.random.multivariate_normal([2, 8], [[0, 0.75],[0.75, 0]], 
                                    num_observations)

# Compiling data into arrays
simulated_features = np.vstack((x1, x2, x3)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations), 
                            np.ones(num_observations), 
                            np.ones(num_observations) + 1))

# Plotting generated data
plt.figure(figsize = (12, 8))
plt.scatter((simulated_features[:, 0]), simulated_features[:, 1], 
            c = simulated_labels, alpha = 0.4)
plt.show()