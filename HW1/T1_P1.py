#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np
import matplotlib.pyplot as plt

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

def compute_loss(tau):
    
    loss = 0

    for i, (x_n, y_n) in enumerate(data):
        neg_sum = 0
        for j, (x_i, y_i) in enumerate(data):
            if i != j:
                distance = - (x_i - x_n) ** 2
                neg_sum += pow(np.e, distance / tau) * y_i
            
        loss += (y_n - neg_sum) ** 2

    return loss

def plot():
    plot_data = { "0.01" : [], "2" : [], "100" : [] }
    
    for tau in (0.01, 2, 100):
        kernel_reg = 0
        for x in np.arange(0., 12.1, 0.1):
            kernel_reg = 0
            for (x_n, y_n) in data:
                distance = - (x_n - x) ** 2
                kernel_reg += pow(np.e, distance / tau) * y_n
            plot_data[str(tau)].append(kernel_reg)

    line1, = plt.plot(np.arange(0., 12.1, 0.1), plot_data["0.01"], label="0.01")
    line2, = plt.plot(np.arange(0., 12.1, 0.1), plot_data["2"], label="2")
    line3, = plt.plot(np.arange(0., 12.1, 0.1), plot_data["100"], label="100")
    leg = plt.legend(loc="upper right")
    plt.xlabel("x^*")
    plt.ylabel("f(x^*)")
    plt.show()

plot()

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))