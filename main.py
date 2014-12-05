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
import matplotlib.pyplot as plt
from collections import defaultdict

# DATASET_NAME = 'MNIST original'	# Big Dataset, Multiclass
# DATASET_NAME = 'iris'	# Small Dataset, Multiclass
# DATASET_NAME = 'australian'	# Medium Dataset, Binary
# DATASET_NAME = 'SensIT Vehicle (combined)'
DATASET_NAME = 'leukemia'	# Small Dataset, High Dimension, Binary
# DATASET_NAME = 'WINE'
# DATASET_NAME = 'WINEQUALITY'
# DATASET_NAME = 'SPECT'
# DATASET_NAME = 'Climate Model Simulation Crashes'


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
	return error

state = random.randint(0, 10)

def main(epoch):
	print "Loading Data...", DATASET_NAME
	if DATASET_NAME not in ["WINE", "SPECT", "WINEQUALITY"]:
		dataset = fetch_mldata(DATASET_NAME)
	else:
		dataset = load_data(DATASET_NAME)
	x_train, x_test, y_train, y_test = train_test_split(dataset.data,
		dataset.target, random_state=state)
	print x_train.shape, y_train.shape
	print x_test.shape, y_test.shape
	print np.unique(y_train)
	print "Epochs =", GLOBAL_EPOCH
	print "Learning Rate =", RATE
	print "Degree =", DEGREE
	errors = {}
	exit()

	print "=== LinearSVM"
	clf = svm.LinearSVC()
	train(clf, x_train, y_train)
	e = test(clf, x_test, y_test)
	errors["LinearSVM"] = e

	print "=== SVM"
	clf = svm.SVC()
	train(clf, x_train, y_train)
	e = test(clf, x_test, y_test)
	errors["SVM"] = e

	print "=== SklearnPercepton"
	clf = linear_model.Perceptron()
	train(clf, x_train, y_train)
	e = test(clf, x_test, y_test)
	errors["SklearnPerceptron"] = e

	print "=== Perceptron"
	# clf = Perceptron()
	clf = MultiClassifier(Perceptron, epoch)
	train(clf, x_train, y_train)
	e = test(clf, x_test, y_test)
	errors["Perceptron"] = e

	print "=== VotedPerceptron"
	# clf = VotedPerceptron()
	clf = MultiClassifier(VotedPerceptron, epoch)
	train(clf, x_train, y_train)
	e = test(clf, x_test, y_test)
	errors["VotedPerceptron"] = e

	print "=== KernelPerceptron"
	# clf = KernelPerceptron(linear)
	clf = MultiClassifier(KernelPerceptron, epoch)
	train(clf, x_train, y_train)
	e = test(clf, x_test, y_test)
	errors["KernelPerceptron"] = e

	print "=== KernelVotedPerceptron"
	# clf = KernelVotedPerceptron(linear)
	clf = MultiClassifier(KernelVotedPerceptron, epoch)
	train(clf, x_train, y_train)
	e = test(clf, x_test, y_test)
	errors["KernelVotedPerceptron"] = e

	return errors

if __name__ == '__main__':
	graphs = defaultdict(list)
	plt.title('Epoch Effect')
	plt.axis([1, 10, 0, 30])
	global GLOBAL_EPOCH

	for e in [1,3,5]:
		for r in [0.1]:
			for d in [5]:
				GLOBAL_EPOCH = e
				RATE = r
				DEGREE = d
				# print "=+=+=+=+=+=+=+=+=+=+=+=+"
				# print "=+=+=+=+=+=+=+=+=+=+=+=+"
				errors = main(e)
				for k,v in errors.iteritems():
					graphs[k].append((e, v))

	for k,v in graphs.iteritems():
		if k == "KernelVotedPerceptron": v = graphs["VotedPerceptron"]
		x = [i[0] for i in v]
		y = [i[1] for i in v]
		plt.plot(x, y, label=k)
	plt.legend()
	plt.show()
