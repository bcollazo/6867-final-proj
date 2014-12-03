import numpy as np
import math

def sign(x):
	if x < 0:
		return -1
	else:
		return 1

class KernelPerceptron():
	def __init__(self, kernel):
		self.alphas = None
		self.kernel = kernel

	def fit(self, X, Y, T=10):
		alphas = np.zeros(len(X))
		
		self.alphas = alphas

	def predict(self, X):
		return np.array([sign(self.v.dot(x)) for x in X])

	def value_predict(self, X):
		return np.array([self.v.dot(x) for x in X])

class VotedPerceptron():
	def __init__(self):
		self.weights = []
		self.vectors = []
		self.k = 0

	def fit(self, X, Y, T=10):
		k = 0
		vs = [np.zeros(X[0].shape)]
		cs = [0]
		for j in xrange(T):
			for i in xrange(len(X)):
				x = X[i]
				y = Y[i]

				g = sign(vs[k].dot(x))
				if g == y:
					cs[k] += 1
				else:
					v = vs[k] + y*x
					vs.append(v)
					cs.append(1)
					k += 1
		self.k = k
		self.weights = cs
		self.vectors = vs

	def predict(self, X):
		guesses = [sign(sum([self.weights[j]*
			sign(x.dot(self.vectors[j])) 
			for j in xrange(self.k)])) for x in X]
		return np.array(guesses)

	def value_predict(self, X):
		return [sum([self.weights[j]*
			sign(x.dot(self.vectors[j]))
			for j in xrange(self.k)]) for x in X]