#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, .1)

print("y is:")
print(y_train)

def kernel(x1, x2, tau):
    return np.exp(-1 * (x1 - x2) * (x1 - x2) / tau)

def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    # TODO: your code here
    y_test = []
    for x_new in x_test:

        # find all kernel distances between x and xdata
        dists = [kernel(x_old, x_new, tau=1) for x_old in x_train]

        # find indices of min k dists
        min_dists = sorted(dists, reverse=True)[0:k]
        min_indices = []
        for dist in min_dists:
            index = dists.index(dist)
            min_indices.append(index)
            dists[index] = -1 # make sure repeated point doesn't get chosen twice

        # choose k points from data
        closest_y = [y_train[i] for i in min_indices]
        print("x is", x_new, "closest y are", closest_y)
        
        y_test.append(sum(closest_y) / k)
    return y_test


def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0,3])
    
    y_test = predict_knn(k=k)
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.legend()
    plt.title("KNN Predictions with k = " + str(k))
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for k in (1, 3, len(x_train)-1):
    plot_knn_preds(k)