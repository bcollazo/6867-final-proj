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

# DATASET_NAME = 'MNIST original'	# Big Dataset, Multiclass
# DATASET_NAME = 'iris'	# Small Dataset, Multiclass
# DATASET_NAME = 'australian'	# Small Dataset, Binary
# DATASET_NAME = 'SensIT Vehicle (combined)'	# Big Dataset, Multiclass, Low Dimension
# DATASET_NAME = 'leukemia'	# Small Dataset, High Dimension, Binary
# DATASET_NAME = 'WINE'
# DATASET_NAME = 'WINEQUALITY'
# DATASET_NAME = 'SPECT'


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

def main():
	print "Loading Data...", DATASET_NAME
	if DATASET_NAME not in ["WINE", "SPECT", "WINEQUALITY"]:
		dataset = fetch_mldata(DATASET_NAME)
	else:
		dataset = load_data(DATASET_NAME)
	x_train, x_test, y_train, y_test = train_test_split(dataset.data,
		dataset.target)
	print x_train.shape, y_train.shape
	print x_test.shape, y_test.shape
	print np.unique(y_train)
	print "Epochs =", GLOBAL_EPOCH
	print "Learning Rate =", RATE
	print "Degree =", DEGREE

	print "=== LinearSVM"
	clf = svm.LinearSVC()
	train(clf, x_train, y_train)
	test(clf, x_test, y_test)

	print "=== SVM"
	clf = svm.SVC()
	train(clf, x_train, y_train)
	test(clf, x_test, y_test)

	print "=== SklearnPercepton"
	clf = linear_model.Perceptron()
	train(clf, x_train, y_train)
	test(clf, x_test, y_test)

	print "=== Perceptron"
	# clf = Perceptron()
	clf = MultiClassifier(Perceptron)
	train(clf, x_train, y_train)
	test(clf, x_test, y_test)

	print "=== VotedPerceptron"
	# clf = VotedPerceptron()
	clf = MultiClassifier(VotedPerceptron)
	train(clf, x_train, y_train)
	test(clf, x_test, y_test)

	print "=== KernelPerceptron"
	# clf = KernelPerceptron(linear)
	clf = MultiClassifier(KernelPerceptron)
	train(clf, x_train, y_train)
	test(clf, x_test, y_test)

	print "=== KernelVotedPerceptron"
	# clf = KernelVotedPerceptron(linear)
	clf = MultiClassifier(KernelVotedPerceptron)
	train(clf, x_train, y_train)
	test(clf, x_test, y_test)

if __name__ == '__main__':
	for e in [1,3,5]:
		for r in [0.1,1,10]:
			for d in [1,3,5]:
				GLOBAL_EPOCH = e
				RATE = r
				DEGREE = d
				print "=+=+=+=+=+=+=+=+=+=+=+=+"
				print "=+=+=+=+=+=+=+=+=+=+=+=+"
				main()
