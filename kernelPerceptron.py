import numpy as np
import math
from util import *
import random

class KernelPerceptron():
	def __init__(self, kernel=KERNEL):
		self.alphas = None
		self.kernel = kernel
		self.kMatrix = {}

	def fit(self, X, Y, T=GLOBAL_EPOCH):
		self.X = X
		self.Y = Y
		n = len(X)
		self.n = n

		alphas = np.zeros(n)
		for epoch in xrange(T):
			print "EPOCH", epoch
			for j in xrange(n):
				# if random.random() < 0.01: print epoch, j
				x, y = X[j], Y[j]

				# kernels = [self.kMatrix[(i,j)] for i in xrange(n)]
				kernels = [self.kernel(X[i], x) for i in xrange(n)]
				kernels = np.array(kernels)
				kernels = Y*kernels
				g = sign(alphas.dot(kernels))
				if g != y:
					alphas[j] = alphas[j] + 1

		self.alphas = alphas

	def predict(self, X):
		return vsign(self.value_predict(X))

	def value_predict(self, X):
		return np.array([self.alphas.dot(
			self.Y*np.array([self.kernel(self.X[i], x) for i in range(self.n)]))
			for x in X])

class KernelVotedPerceptron():
	def __init__(self, kernel=KERNEL):
		self.kernel = KERNEL
		self.u = []
		self.c = []
		self.k = 0

	def fit(self, X, Y, T=GLOBAL_EPOCH):
		self.X = X
		self.Y = Y

		k = 0
		c = [0]
		u = []
		for epoch in xrange(T):
			for i in xrange(len(X)):

				g = sign(sum([Y[u[j]]*self.kernel(X[u[j]], X[i]) for j in xrange(k)]))
				
				if g == Y[i]:
					c[k] += 1
				else:
					u.append(i)
					c.append(1)
					k += 1
		self.k = k
		self.c = np.array(c[:-1])
		self.u = u

	def predict(self, X):
		return vsign(self.value_predict(X))

	def value_predict(self, X):
		guesses = []
		for x in X:
			signs = np.array([sign(sum([self.Y[self.u[j]]*
				self.kernel(self.X[self.u[j]], x) for j in xrange(i)])) 
				for i in xrange(self.k)])
			# print self.c.shape, signs.shape 
			g = sign(self.c.dot(signs))
			guesses.append(g)
		return np.array(guesses)
		