import numpy as np
import math

def sign(x):
	if x < 0:
		return -1
	else:
		return 1

vsign = np.vectorize(sign)

class KernelPerceptron():
	def __init__(self, kernel):
		self.alphas = None
		self.kernel = kernel
		self.kMatrix = {}

	def fit(self, X, Y, T=10):
		self.X = X
		self.Y = Y
		n = len(X)
		self.n = n

		for i in xrange(n):
			for j in xrange(i, n):
				val = self.kernel(X[i], X[j])
				self.kMatrix[(i,j)] = val
				self.kMatrix[(j,i)] = val

		alphas = np.zeros(n)
		for epoch in xrange(T):
			for j in xrange(n):
				x, y = X[j], Y[j]

				kernels = [self.kMatrix[(i,j)] for i in xrange(n)]
				kernels = np.array(kernels)
				kernels = Y*kernels
				g = sign(alphas.dot(kernels))
				if g != y:
					alphas[j] = alphas[j] + 1

		self.alphas = alphas

	def predict(self, X):
		return vsign(self.value_predict(X))

	def value_predict(self, X):
		return np.array([sign(self.alphas.dot(
			self.Y*np.array([self.kernel(self.X[i], x) for i in range(self.n)])))
			for x in X])

class KernelVotedPerceptron():
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
		# guesses = [sign(sum([self.weights[j]*
		# 	sign(x.dot(self.vectors[j])) 
		# 	for j in xrange(self.k)])) for x in X]
		# return np.array(guesses)
		return vsign(self.value_predict(X))

	def value_predict(self, X):
		return [sum([self.weights[j]*
			sign(x.dot(self.vectors[j]))
			for j in xrange(self.k)]) for x in X]