import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

def features(X):

	F = np.zeros((len(X),5))
	for x in range(0,len(X)):
		F[x,0] = len(X[x])        # length
		F[x,1] = ord(X[x][0])
		F[x,2] = ord(X[x][-1])
		F[x,3] = coutLetters(X[x])
		#F[x,1] = coutVowels(X[x]) # vowels
		#F[x,3] = F[x,0] % 2       # even/odd
		#F[x,3] = F[x, 1] / F[x, 0]
	return F     

def mytraining(f,Y):
	#reg = linear_model.Perceptron(max_iter = 10000, tol=None, penalty=None, alpha=0.00001) #.23
	#reg = linear_model.LogisticRegression() #.209
	#reg = linear_model.RidgeCV(alphas=[.1]) #nop
	#reg = tree.DecisionTreeClassifier(min_samples_split = 8) #201
	reg = neighbors.KNeighborsClassifier(10, 'distance') #.21
	reg = reg.fit(f, Y)
	return reg
    
def mytrainingaux(f,Y,par):
    
	return clf

def myprediction(f, clf):
	Ypred = clf.predict(f)

	return Ypred

def coutVowels(mip):
	a = 0
	for v in "aeiou":
		a += mip.lower().count(v)
	return a

def coutLetters(mip):
	a = 0
	for x in mip:
		a += ord(x)
	return a