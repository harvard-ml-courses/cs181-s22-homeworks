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

def takeSecond(elem):
    _, kernel_reg = elem
    return kernel_reg

def predict_knn(k, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k.""" 
    y_test = np.zeros(len(x_test))
    
    # List of lists: will store the distances between every x_test point and x_0 to x_6
    k_dist = []

    for x in x_test:
        lst = []
        for i, (x_n, y_n) in enumerate(data):
            distance = - (x_n - x) ** 2
            kernel = pow(np.e, distance)
            lst.append((i, kernel))
        k_dist.append(lst)
    
    for lst in k_dist:
        lst.sort(key=takeSecond, reverse=True)
    print(k_dist)
    for n, lst in enumerate(k_dist):
        kernel_reg = 0
        for i, (j, dist) in enumerate(lst):
            if i <= k:
                x, y = data[j]
                kernel_reg += y
            else:
                y_test[n] = (1/k) * kernel_reg
    print(y_test)
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