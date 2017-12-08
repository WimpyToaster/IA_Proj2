"""
Grupo 029

Antonio Terra 84702
Diogo D'Andrade 84709

"""

import numpy as np
from sklearn import datasets, tree, linear_model, svm, gaussian_process, neighbors, ensemble
from sklearn.preprocessing import PolynomialFeatures

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
import timeit

def mytraining(X,Y):

	
	reg = 
	
	#reg = KernelRidge(kernel='rbf', gamma=0.1, alpha=0.001)
	
	
	reg.fit(X,Y)
	return reg

def myprediction(X,reg):

	Ypred = reg.predict(X)

	return Ypred


def meh(X, Y):
	temp = []
	for i in range(len(X)):
		temp.append([X[i][0], Y[i][0]])
	return temp

def meh1(X):
	return tuple((x[0],) for x in X)