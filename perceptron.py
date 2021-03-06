import numpy as np
import math
from util import *

class Perceptron():
	def __init__(self):
		self.v = None

	def fit(self, X, Y, T=GLOBAL_EPOCH):
		v = np.zeros(X[0].shape)
		for j in xrange(T):
			for i in xrange(len(X)):
				x = X[i]
				y = Y[i]
				
				g = sign(v.dot(x))
				if g != y:
					v = v + RATE*y*x
		self.v = v

	def predict(self, X):
		return vsign(self.value_predict(X))

	def value_predict(self, X):
		return np.array([self.v.dot(x) for x in X])

class VotedPerceptron():
	def __init__(self):
		self.weights = []
		self.vectors = []
		self.k = 0

	def fit(self, X, Y, T=GLOBAL_EPOCH):
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
					v = vs[k] + RATE*y*x
					vs.append(v)
					cs.append(1)
					k += 1
		self.k = k
		self.weights = cs
		self.vectors = vs

	def predict(self, X):
		return vsign(self.value_predict(X))

	def value_predict(self, X):
		return [sum([self.weights[j]*
			sign(x.dot(self.vectors[j]))
			for j in xrange(self.k)]) for x in X]