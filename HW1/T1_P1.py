#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

N = len(data)
xdata = [f for (f,s) in data]
ydata = [s for (f,s) in data]

def kernel(x1, x2, tau):
    return np.exp(-1 * (x1 - x2) * (x1 - x2) / tau)

def compute_loss(tau):
    # TODO

    loss = 0
    for n in range(N):
        f = sum([kernel(xdata[m], xdata[n], tau) * ydata[m] for m in range(N) if m != n])
        loss += (ydata[n] - f) ** 2
    
    return loss

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))

import matplotlib.pyplot as plt

def f(xn, tau):
    return sum([kernel(xdata[m], xn, tau) * ydata[m] for m in range(N)])

xrange = np.arange(0, 12, 0.1)
for t in [0.01, 2, 100]:
    plt.plot(xrange, [f(x, t) for x in xrange])
    plt.title("Kernel-Based Regression predictions for τ = " + str(t))
    plt.savefig('Q1_tau' + str(t) + '.png')
    plt.show()

plt.scatter(xdata, ydata)
for t in [0.01, 2, 100]:
    plt.plot(xrange, [f(x, t) for x in xrange], label="τ = " + str(t))
plt.title("Kernel-Based Regression predictions")
plt.legend()
plt.savefig('Q1_all')
plt.show()