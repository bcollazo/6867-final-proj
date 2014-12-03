from mnist import *
from spect import *
import numpy as np

def load_data(n, dataset):
	print "Loading Data..."
	if n == 1:
		x, y = load_mnist(dataset=dataset, path="data/")
	elif n == 2:
		x, y = load_spect(dataset=dataset)
	else:
		x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
		y = np.array([[1], [1], [1], [-1]])

	return x, y

	# x_train = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
	# y_train = np.array([1,1,1,-1])
	# x_test = x_train
	# y_test = y_train


def countErrors(g, y):
	t = np.equal(g, y)
	return sum([1 for i in t if not i])


class MultiClassifier():
	def __init__(self, clf):
		self.classifiers = []
		self.classes = []
		self.clf = clf

	def fit(self, X, Y):
		classifiers = []
		classes = np.unique(Y)
		for val in classes:
			print "Training", val
			new_labels = []
			for i in xrange(len(Y)):
				if Y[i] == val:
					new_labels.append(1)
				else:
					new_labels.append(-1)
			clf = self.clf()
			clf.fit(X, np.array(new_labels))
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