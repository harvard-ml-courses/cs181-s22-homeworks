#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
    
    if part == 'a':
        parta = []
        for x in xx:
            parta.append([x ** j for j in range(6)])
        return np.array(parta)

    if part == 'b':
        partb = []
        for x in xx:
            partb.append([1] + [np.exp((-(x-mu) ** 2) / 25) for mu in range(1960, 2015, 5)])
        return np.array(partb)

    if part == 'c':
        partc = []
        for x in xx:
            partc.append([1] + [np.cos(x / j) for j in range(1, 6)])
        return np.array(partc)

    if part == 'd':
        partd = []
        for x in xx:
            partd.append([1] + [np.cos(x / j) for j in range(1, 26)])
        return np.array(partd)
        
    return None

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)

for part in ['a', 'b', 'c', 'd']:
    grid_X = make_basis(grid_years, part)
    w = find_weights(make_basis(years, part), Y)
    grid_Yhat = np.dot(grid_X, w)

    # TODO: plot and report sum of squared error for each basis
    loss = sum([(Y[n] - np.dot(make_basis(years, part), w)[n]) ** 2 for n in range(len(years))])
    print("Loss for part", part, "is", loss)

    # Plot the data and the regression line.
    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("LSQ Regression with basis for part " + part + ")")
    plt.text(1985, 35, "Loss =" + str(loss))
    plt.savefig("Q4.1_part" + part + ".png")
    plt.show()

grid_sunspots = np.linspace(min(sunspot_counts[years<last_year]), max(sunspot_counts[years<last_year]), 200)
for part in ['a', 'c', 'd']:
    grid_X = make_basis(grid_sunspots, part, False)
    w = find_weights(make_basis(sunspot_counts[years<last_year], part, False), Y[years<last_year])
    grid_Yhat = np.dot(grid_X, w)

    # TODO: plot and report sum of squared error for each basis
    loss = sum([(Y[n] - np.dot(make_basis(sunspot_counts[years<last_year], part, False), w)[n]) ** 2 for n in range(len(sunspot_counts[years<last_year]))])
    print("Loss for part", part, "is", loss)

    # Plot the data and the regression line.
    plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_sunspots, grid_Yhat, '-')
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("LSQ Regression with basis for part " + part + ")")
    
    if part == 'a':
        plt.text(80, 33, "Loss =" + str(loss))
    if part == 'c':
        plt.text(95, 30, "Loss =" + str(loss))
    if part == 'd':
        plt.text(90, -200, "Loss =" + str(loss))
    

    plt.savefig("Q4.2_part" + part + ".png")
    plt.show()