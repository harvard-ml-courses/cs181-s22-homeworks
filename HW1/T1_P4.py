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
# plt.figure(1)
# plt.plot(years, republican_counts, 'o')
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.figure(2)
# plt.plot(years, sunspot_counts, 'o')
# plt.xlabel("Year")
# plt.ylabel("Number of Sunspots")
# plt.figure(3)
# plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
# plt.xlabel("Number of Sunspots")
# plt.ylabel("Number of Republicans in Congress")
# plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part,is_years=True):
#DO NOT CHANGE LINES 65-69: re-scaling the data
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20

    return_arr = []
    
    # Problem 4.1
    if part == 'a' and is_years:
        for year in xx:
            year_vec = np.array([1, year, year ** 2, year ** 3, year ** 4, year ** 5])
            return_arr.append(year_vec)

    if part == 'b' and is_years:
        for year in xx:
            year_lst = [1]
            for mu in np.arange(1960., 2011., 5.):
                year_lst.append(pow(np.e, (- (year - mu) ** 2) / 25.))
            year_vec = np.array(year_lst)
            return_arr.append(year_vec)
    
    if part == 'c' and is_years:
        for year in xx:
            year_lst = [1]
            for j in np.arange(1., 6., 1.):
                year_lst.append(np.cos(year/j))
            year_vec = np.array(year_lst)
            return_arr.append(year_vec)

    if part == 'd' and is_years:
        for year in xx:
            year_lst = [1]
            for j in np.arange(1., 26., 1.):
                year_lst.append(np.cos(year/j))
            year_vec = np.array(year_lst)
            return_arr.append(year_vec)

    # Problem 4.2
    if part == 'a' and not is_years:
        for i, count in enumerate(xx):
            if i < 13: # years before 1985
                count_vec = np.array([1, count, count ** 2, count ** 3, count ** 4, year ** 5])
                return_arr.append(count_vec)
    
    if part == 'c' and not is_years:
        for i, count in enumerate(xx):
            if i < 13: # years before 1985
                count_lst = [1]
                for j in np.arange(1., 6., 1.):
                    count_lst.append(np.cos(count/j))
                count_vec = np.array(count_lst)
                return_arr.append(count_vec)
    
    if part == 'd' and not is_years:
        for i, count in enumerate(xx):
            if i < 13: # years before 1985
                count_lst = [1]
                for j in np.arange(1., 26., 1.):
                    count_lst.append(np.cos(count/j))
                count_vec = np.array(count_lst)
                return_arr.append(count_vec)

    return np.vstack(return_arr)

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)

squared_errors = []

for part in ["a", "b", "c", "d"]:
    grid_X = make_basis(grid_years, part)
    grid_Yhat  = np.dot(grid_X, find_weights(make_basis(years, part), Y))

    # TODO: plot and report sum of squared error for each basis
    sum_errors = 0
    # Yhat = np.dot(grid_X, find_weights(X, Y))

    # for i, count in enumerate(Y):
    #     sum_errors += (count - Yhat[i]) ** 2
    # squared_errors.append(sum_errors)

    # Plot the data and the regression line.
    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.show()

for error in squared_errors:
    print(f"Loss " + error)