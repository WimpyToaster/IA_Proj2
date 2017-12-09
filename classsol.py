"""
Grupo 029

Antonio Terra 84702
Diogo D'Andrade 84709

"""


import numpy as np
from sklearn import neighbors, datasets, tree, linear_model
from sklearn.model_selection import GridSearchCV

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
	
	
	treeC = tree.DecisionTreeClassifier(min_samples_split=2)
	neigC = neighbors.KNeighborsClassifier(2, 'uniform')
	
	treeParams = {"criterion":["gini", "entropy"], "splitter":["best", "random"], "min_samples_split":[1.0, 2, 3, 4, 5], "min_samples_leaf":[1, 2, 3, 4, 5, 10], "min_weight_fraction_leaf":[0.0, 0.1, 0.5], "max_leaf_nodes":[None, 10, 100], "min_impurity_decrease":[0.0, 0.1, 0.2]}
	neigParams = {"n_neighbors":[1,2,3,4,5,6], "weights":["uniform", "distance"], "algorithm":["auto", "ball_tree", "kd_tree", "brute"], "leaf_size":[1, 2, 5, 10, 20, 30], "p":[1, 2]}

	clf = GridSearchCV(treeC, cv = 2, param_grid=treeParams)
	clf = GridSearchCV(neigC, cv = 2, param_grid=neigParams)

	clf.fit(f, Y)

	return clf

def myprediction(f, clf):
	Ypred = clf.predict(f)

	return Ypred

def coutVowels(mip):
	a = 0
	for v in "aeiou":
		a += mip.lower().count(v)
	return a

#0.259615384615
#0.221153846154