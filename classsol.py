"""
Grupo 029

Antonio Terra 84702
Diogo D'Andrade 84709

"""

import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

import sys


def features(X):

	F = np.zeros((len(X),5))
	for x in range(0,len(X)):
		F[x,0] = len(X[x])        # length
		F[x,1] = ord(X[x][0])     # ascii 1st letter
		F[x,2] = hash(X[x])       # hash
		F[x,3] = coutVowels(X[x]) # num vowels
		F[x,4] = F[x,0] % 2       # even/odd
	return F     

def mytraining(f,Y):
	
	clf = tree.DecisionTreeClassifier(min_samples_split = 2) # perfect
	
	#clf = neighbors.KNeighborsClassifier(2, 'uniform')      # perfect
	clf = clf.fit(f, Y)
	return clf

def myprediction(f, clf):
	Ypred = clf.predict(f)

	return Ypred

def coutVowels(mip):
	a = 0
	for v in "aeiou":
		a += mip.lower().count(v)
	return a
