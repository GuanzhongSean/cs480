#!/bin/python3.10
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
# from ucimlrepo import fetch_ucirepo
import perceptron as per

# Fetch dataset
# spambase = fetch_ucirepo(id=94)  # fetch from UCI
# data = read_csv('datasets/spambase/spambase.data',
#                 header=None)  # fetch from local

# Data
# X = spambase.data.features
# y = spambase.data.targets
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# X = X.to_numpy()
# y = np.where(y == 0, -1, 1)

# From shuffled data
X = read_csv('datasets/spambase/spambase_X.csv', header=None)
y = read_csv('datasets/spambase/spambase_y.csv', header=None)
X = X.to_numpy().T
y = y.to_numpy()

n, d = X.shape
w = w_shuffled = np.zeros(d)
max_pass = 500

w, b, mistakes = per.perceptron(X, y, w, 0, max_pass)
w_shuffled, b_shulffed, mistakes_shuffled = per.perceptron_shuffled(
    X, y, w_shuffled, 0, max_pass)

# Plot
plt.plot(range(1, max_pass + 1), mistakes)
plt.xlabel('Number of Passes')
plt.ylabel('Number of Mistakes')
plt.title('Perceptron Mistakes vs. Passes')
plt.savefig('perceptron_mistakes_vs_passes.png')
plt.close()

plt.plot(range(1, max_pass + 1), mistakes_shuffled)
plt.xlabel('Number of Passes')
plt.ylabel('Number of Mistakes')
plt.title('Perceptron Mistakes vs. Passes (Shuffled)')
plt.savefig('perceptron_mistakes_vs_passes_shuffled.png')
plt.close()
