import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
from T1_P4 import make_basis, find_weights


'''
Instructions for using this Autograder:
1. Make sure it is in the same immediate directory/folder as your implementation file, which *must* be called T1_P4.py
2. Run this only after you have implemented the functions make_basis and find_weights for parts (a) and (d), or at least put in some dummy output for these parts so that the Autograder can run.
3. The test data used here switches the x and y axes from the original problem. The weights you see below are hence not the correct answer.
'''

test_x = np.array([36., 34., 32., 36., 43., 44., 42., 38., 38., 41., 53., 54., 53.,
       45., 45., 44., 43., 52., 55., 55., 50., 51., 55., 49.])
test_y = np.array([1960., 1962., 1964., 1966., 1968., 1970., 1972., 1974., 1976.,
       1978., 1980., 1982., 1984., 1986., 1988., 1990., 1992., 1994.,
       1996., 1998., 2000., 2002., 2004., 2006.])

w_a = [2.31899640e-09, -8.88289383e-08,  3.18990977e-06, -1.01821630e-04,
        2.43732322e-03,  4.30702950e-05]

w_d = [ 1.23650309e+05,  1.02356914e+00,  1.32627256e+02, -5.56954926e+03,
       -4.38872871e+04,  3.47537671e+04, -3.55957031e+05,  4.37324953e+05,
       -7.25200996e+04, -3.37987910e+05, -2.54755410e+05, -6.95818301e+04,
        7.81688135e+04,  1.54475561e+05,  1.70672254e+05,  1.48118670e+05,
        1.05422707e+05,  5.53613877e+04,  5.64504187e+03, -3.94460747e+04,
       -7.81371162e+04, -1.09949648e+05, -1.35043025e+05, -1.54227879e+05,
       -1.68128359e+05, -1.77586744e+05]

for part in ['a','d']:
    X = make_basis(test_x,part) 
    if part == 'a':
        if X.shape != (24,6):
            print("For part a, your X is the wrong shape. Shape should be (24,6) and you have ", X.shape)
    elif part == 'd':
        if X.shape != (24,26):
            print("For part d, your X is the wrong shape. Shape should be (24,26) and you have ", X.shape)
    w = find_weights(X,test_y)
    if part == 'a':
        if len(w_a)!=len(w):
            print("Your weights for part a have the wrong shape. Length should be ", len(w_a), " and you have ", len(w))
        else:
            parta_checker = np.array_equal(np.around(w_a,-1),np.around(w,-1))
            if parta_checker:
                parta_checker = "Pass"
            else:
                parta_checker = "Fail"
    elif part == 'd':
        if len(w_d)!=len(w):
            print("Your weights for part d have the wrong shape. Length should be ", len(w_d), " and you have ", len(w))
        else:
            partd_checker = np.array_equal(np.around(w_d,-1),np.around(w,-1))
            if partd_checker:
                partd_checker = "Pass"
            else:
                partd_checker = "Fail"
        
print("Your test case results for parts a and d respectively are:", parta_checker, partd_checker) 