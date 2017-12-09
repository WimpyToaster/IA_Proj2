"""
Grupo 029

Antonio Terra 84702
Diogo D'Andrade 84709

"""


import numpy as np
from sklearn import datasets, tree, linear_model, neighbors, gaussian_process
from sklearn.gaussian_process.kernels import *
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import timeit

def mytraining(X,Y):

	scores = ['neg_mean_squared_error']

	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()

		clf = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5, param_grid={"alpha": [10, 1e0, 0.1, 1e-2, 1e-3, 1e-4], "gamma": np.logspace(-2, 2, 5)}, scoring="neg_mean_squared_error")
		#clf = GridSearchCV(neighbors.KNeighborsRegressor(), cv=5, param_grid={"n_neighbors":[1, 2, 5], "weights":["uniform", "distance"], "algorithm":["auto", "ball_tree"], "leaf_size":[30], "p":[2], "metric":["minkowski"]}, scoring='neg_mean_squared_error')
		#clf = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6], "gamma": np.logspace(-2, 2, 5)}, scoring="neg_mean_squared_error")

		clf.fit(X, Y.ravel())
		

		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Best score:")
		print()
		print(clf.best_score_)
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']

	return clf

def myprediction(X,reg):

	Ypred = reg.predict(X)

	return Ypred

