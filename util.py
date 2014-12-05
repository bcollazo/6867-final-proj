from mnist import *
from spect import *
import numpy as np

# Perceptron Settings
GLOBAL_EPOCH = 1
RATE = 1
# Kernel Settings
DEGREE = 1
def linear(x, z):
	return x.dot(z)
def poly_kernel(x, z, d=DEGREE):
	return x.dot(z)**d
KERNEL = poly_kernel

# Helper Functions ===
def sign(x):
	if x < 0:
		return -1
	else:
		return 1

vsign = np.vectorize(sign)

class Obj():
	def __init__(self, x, y):
		self.target = y
		self.data = x

def load_data(n, dataset='training'):
	# if n == 1:
	# 	x, y = load_mnist(dataset=dataset, path="data/")
	if n == "SPECT":
		x, y = load_spect(dataset=dataset)
	elif n == "WINEQUALITY":
		x = np.loadtxt('datasets/winequality-red.csv', delimiter=";")
		y = x[:,-1]
		x = x[:,:-1]
	elif n == "WINE":
		x = np.loadtxt('datasets/winequality-red.csv', delimiter=";")
		y = x[:,-1]
		x = x[:,:-1]
	else:
		x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
		y = np.array([[1], [1], [1], [-1]])

	print x
	print y
	return Obj(x, y)

	# x_train = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
	# y_train = np.array([1,1,1,-1])
	# x_test = x_train
	# y_test = y_train


def countErrors(g, y):
	t = np.equal(g, y)
	return sum([1 for i in t if not i])


class MultiClassifier():
	def __init__(self, clf, epoch):
		self.classifiers = []
		self.classes = []
		self.clf = clf
		self.epoch = epoch

	def fit(self, X, Y):
		classifiers = []
		classes = np.unique(Y)
		for val in classes:
			# print "Building labels", val
			new_labels = []
			for i in xrange(len(Y)):
				if Y[i] == val:
					new_labels.append(1)
				else:
					new_labels.append(-1)
			clf = self.clf()
			# print "Training", val
			clf.fit(X, np.array(new_labels), self.epoch)
			classifiers.append(clf)
		self.classes = classes
		self.classifiers = classifiers

	def predict(self, X):
		guesses = []
		for clf in self.classifiers:
			guesses.append(clf.value_predict(X))
		predictions = []
		for i in xrange(len(X)):
			m_val = None
			for j in xrange(len(self.classes)): #classes
				if m_val == None or guesses[j][i] > m_val:
					m_val = guesses[j][i]
					m_class_index = j
			predictions.append(self.classes[m_class_index])
		return np.array(predictions)
