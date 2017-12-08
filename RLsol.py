"""
Grupo 029

Antonio Terra 84702
Diogo D'Andrade 84709

"""


import numpy as np

def Q2pol(Q, epsilon = 0.1):
	B = [[sum(y), sum(y)] for y in Q]
	MAX = [max(x) for x in Q]
	
	for i in range(len(Q)):
		temp = np.random.random_sample()
		if (temp < epsilon):
			for j in range( len(Q[0]) ):
				Q[i][j] = Q[i][j] / B[i][j]
		else:
			Q[i] = [0 if Q[i][x] != MAX[i] else 1 for x in range(len(Q[i]))]

	return Q
	
class myRL:

	def __init__(self, nS, nA, gamma):
		self.nS = nS # estados
		self.nA = nA # acoes por estado
		self.gamma = gamma
		self.Q = np.zeros((nS,nA)) # matriz [S][A]
		
	def traces2Q(self, trace):
		# trace [estado i, acao executada, estado seg, recompensa] 
		learning_rate = 0.3 
		self.Q = np.zeros((self.nS,self.nA))
		nQ = np.zeros((self.nS,self.nA))
		while True:            
			for t in trace:
				nQ[int(t[0]),int(t[1])] = (1 - learning_rate) * nQ[int(t[0]), int(t[1])] + learning_rate * (t[3] + self.gamma *  max(nQ[int(t[2]),:]))
				
			if(np.allclose(self.Q, nQ, 0.000001)):
				return nQ

			self.Q = np.copy(nQ)

		return self.Q
 



			