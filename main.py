#!/usr/bin/env python
from util import *
from perceptron import *
from kernelPerceptron import *
from time import time
import numpy as np
import math
from sklearn import svm, linear_model
from sklearn.datasets import load_iris, fetch_mldata
from sklearn.cross_validation import train_test_split

# DATASET_NAME = 'MNIST original'
# DATASET_NAME = 'iris'
DATASET_NAME = 'australian'

def train(clf, x_train, y_train):
	a = time()
	clf.fit(x_train, y_train)
	b = time()
	print "Training Time:", b-a

def test(clf, x_test, y_test):
	a = time()
	g = clf.predict(x_test)
	b = time()
	print "Prediction Time:", b-a
	error = countErrors(g, y_test)
	print "Errors: %d (%2.f %% of test_dataset)"%(error, error * 100 / float(y_test.shape[0]))

def linear(x, z):
	return x.dot(z)

def main():
	print "Loading Data..."
	dataset = fetch_mldata(DATASET_NAME)
	x_train, x_test, y_train, y_test = train_test_split(dataset.data,
		dataset.target)
	print x_train.shape, y_train.shape
	print x_test.shape, y_test.shape
	print np.unique(y_train)

	# print "=== LinearSVM"
	# clf = svm.LinearSVC()
	# train(clf, x_train, y_train)
	# test(clf, x_test, y_test)

	print "=== Perceptron"
	clf = Perceptron()
	# clf = MultiClassifier(Perceptron)
	train(clf, x_train, y_train)
	test(clf, x_test, y_test)

	print "=== VotedPerceptron"
	clf = VotedPerceptron()
	# clf = MultiClassifier(VotedPerceptron)
	train(clf, x_train, y_train)
	test(clf, x_test, y_test)

	print "=== KernelPerceptron"
	clf = KernelPerceptron(linear)
	train(clf, x_train, y_train)
	test(clf, x_test, y_test)


if __name__ == '__main__':
	main()