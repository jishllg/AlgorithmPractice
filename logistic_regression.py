import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, 0.75],[0.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75],[0.75, 1]], num_observations)

simulated_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

plt.figure(figsize=(12, 8))
plt.scatter((simulated_features[:, 0]), simulated_features[:, 1], c = simulated_labels, alpha = 0.4)
plt.show()