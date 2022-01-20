#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample code of HW4-Part II, Problem 1
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import linalg

myfile = open('hw4_part2_problem1_data.pickle', 'rb')
mydict = pickle.load(myfile)

X_train = mydict['X_train']
X_test = mydict['X_test']
Y_train = mydict['Y_train']
Y_test = mydict['Y_test']

predictive_mean = np.empty(X_test.shape[0])
predictive_std = np.empty(X_test.shape[0])
sigma = 0.1
sigma_f = 1.0
ls = 2.0

#-------- Your code (~10 lines) ---------
N = X_train.shape[0]
M = X_test.shape[0]

var = np.square(sigma)
var_f = np.square(sigma_f)
ls_square_2 = 2.0 * np.square(ls)

I = np.identity(N)
Var_I = np.dot(var, I)

Cov_M = np.empty([N, N])
for i in range(N):
    for j in range(N):
        Cov_M[i][j] = var_f * np.exp(-1 * np.square(X_train[i] - X_train[j]) / ls_square_2 ) + Var_I[i][j]

Xk = np.empty([1, N])
Y = np.transpose([Y_train])
Cov_M_inverse = np.linalg.inv(Cov_M + Var_I)
Cov_M_inverse_Y = np.dot(Cov_M_inverse, Y)

for i in range(M):
    for j in range(N):
        Xk[0][j] = var_f * np.exp(-1 * np.square(X_test[i][0] - X_train[j][0]) / ls_square_2 )
    predictive_mean[i] = np.dot(Xk, Cov_M_inverse_Y)
    predictive_std[i] = np.sqrt(var_f + var - np.dot(np.dot(Xk, Cov_M_inverse), np.transpose(Xk)))

#---------- End of your code -----------

# Optional: Visualize the training data, testing data, and predictive distributions
fig = plt.figure()
plt.plot(X_train, Y_train, linestyle='', color='b', markersize=5, marker='+',label="Training data")
plt.plot(X_test, Y_test, linestyle='', color='orange', markersize=2, marker='^',label="Testing data")
plt.plot(X_test, predictive_mean, linestyle=':', color='green')
plt.fill_between(X_test.flatten(), predictive_mean - predictive_std, predictive_mean + predictive_std, color='green', alpha=0.13)
plt.fill_between(X_test.flatten(), predictive_mean - 2*predictive_std, predictive_mean + 2*predictive_std, color='green', alpha=0.07)
plt.fill_between(X_test.flatten(), predictive_mean - 3*predictive_std, predictive_mean + 3*predictive_std, color='green', alpha=0.04)
plt.xlabel("X")
plt.ylabel("Y")
#download
plt.savefig('partI.png')
#view
plt.show()