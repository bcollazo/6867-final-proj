import numpy as np

def load_spect(dataset="training", path="data/"):
	if dataset == "training":
		x = np.loadtxt(path + "SPECT.train.txt", delimiter=",")
	else:
		x = np.loadtxt(path + "SPECT.test.txt", delimiter=",")
	y = x[:,0]
	for i in xrange(len(y)):
		if y[i] == 0:
			y[i] = -1

	return x[:,1:], y.reshape((len(x), 1))