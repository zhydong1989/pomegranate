from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_true
from nose.tools import assert_equal
from nose.tools import assert_greater
from nose.tools import assert_raises
from nose.tools import assert_not_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
import random
import pickle
import numpy as np

numpy.random.seed(0)

def test_kmeans_init():
	centroids = [[2, 3], [5, 7]]
	model = Kmeans(2, centroids)
	assert_equal(model.d, 2)
	assert_equal(model.k, 2)
	assert_array_equal(model.centroids, centroids)


def test_kmeans_from_samples():
	X = numpy.concatenate([numpy.random.normal(0, 1, size=(25, 3)), 
						   numpy.random.normal(8, 1, size=(25, 3))])

	model = Kmeans.from_samples(2, X, init='random')
	centroids = [[ 0.1070838,   0.14769405, -0.2569194 ],
 				 [ 8.45162528,  8.27646348,  8.02635454]]

	assert_array_almost_equal(model.centroids, centroids)

	model = Kmeans.from_samples(2, X, init='first-k')
	centroids = [[ 8.2924351,   8.1156335,   7.88007494],
 				 [ 0.14404818,  0.18963403, -0.23232582]]
 	assert_array_almost_equal(model.centroids, centroids)
	
	model = Kmeans.from_samples(2, X, init='kmeans++')
	centroids = [[ 8.45162528,  8.27646348,  8.02635454],
 				 [ 0.1070838,   0.14769405, -0.2569194 ]]
 	assert_array_almost_equal(model.centroids, centroids)

	model = Kmeans.from_samples(2, X, init='kmeans||')
	centroids = [[ 0.1070838,   0.14769405, -0.2569194 ],
 				 [ 8.45162528,  8.27646348,  8.02635454]]

 	assert_array_almost_equal(model.centroids, centroids)


def test_kmeans_predict():
	X = numpy.concatenate([numpy.random.normal(0, 1, size=(25, 3)), 
						   numpy.random.normal(8, 1, size=(25, 3))])

	model = Kmeans.from_samples(2, X, init='random')
	y_hat = model.predict(X)
	y = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	assert_array_equal(y, y_hat)

	model = Kmeans.from_samples(2, X, init='first-k')
	y_hat = model.predict(X)
	y = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	assert_array_equal(1-y, y_hat)

	model = Kmeans.from_samples(2, X, init='kmeans++')
	y_hat = model.predict(X)
	y = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	assert_array_equal(1-y, y_hat)

	model = Kmeans.from_samples(2, X, init='kmeans||')
	y_hat = model.predict(X)
	y = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	assert_array_equal(1-y, y_hat)


def test_kmeans_predict_large():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 10)) for i in range(10)])

	model = Kmeans.from_samples(10, X, init='kmeans++')
	y_hat = model.predict(X)
	y = [0]*100 + [8]*100 + [6]*100 + [5]*100 + [7]*100 + [2]*100 + [4]*100 + [3]*100 + [9]*100 + [1]*100

	assert_array_equal(y, y_hat)


def test_kmeans_fit():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)

	centroids = (numpy.ones((5, 3)).T * numpy.arange(5) * 3).T

	model = Kmeans(5, centroids, n_init=1)
	model.fit(X)

	centroids = [[  0.0213785,    0.04556998,   0.02620603],
				 [  2.9460307,    2.95655842,   3.01425851],
				 [  6.03862437,   5.96177857,   5.98935301],
				 [  9.1014985,    9.07597708,   8.9594636 ],
				 [ 12.08125299,  12.09561284,  12.04601822]]

	assert_array_almost_equal(model.centroids, centroids)


def test_kmeans_multiple_init():
	X = numpy.concatenate([numpy.random.normal(i, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)

	model1 = Kmeans.from_samples(5, X, init='kmeans++', n_init=1)
	model2 = Kmeans.from_samples(5, X, init='kmeans++', n_init=25)
	
	dist1 = model1.distance(X).min(axis=1).sum()
	dist2 = model2.distance(X).min(axis=1).sum()

	assert_greater(dist1, dist2)

	model1 = Kmeans.from_samples(5, X, init='random', n_init=1)
	model2 = Kmeans.from_samples(5, X, init='random', n_init=25)
	
	dist1 = model1.distance(X).min(axis=1).sum()
	dist2 = model2.distance(X).min(axis=1).sum()

	assert_greater(dist1, dist2)

	model1 = Kmeans.from_samples(5, X, init='first-k', n_init=1)
	model2 = Kmeans.from_samples(5, X, init='first-k', n_init=25)
	
	dist1 = model1.distance(X).min(axis=1).sum()
	dist2 = model2.distance(X).min(axis=1).sum()

	assert_equal(dist1, dist2)


def test_kmeans_ooc_from_samples():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)

	model1 = Kmeans.from_samples(5, X, init='first-k', batch_size=500)
	model2 = Kmeans.from_samples(5, X, init='first-k', batch_size=None)

	assert_array_equal(model1.centroids, model2.centroids)

	model = Kmeans.from_samples(5, X, init='random', batch_size=100)
	model = Kmeans.from_samples(5, X, init='kmeans++', batch_size=100)
	model = Kmeans.from_samples(5, X, init='kmeans||', batch_size=100)


def test_kmeans_ooc_fit():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)

	centroids = (numpy.ones((5, 3)).T * numpy.arange(5) * 3).T

	model1 = Kmeans(5, centroids, n_init=1)
	model1.fit(X)

	model2 = Kmeans(5, centroids, n_init=1)
	model2.fit(X, batch_size=10)

	model3 = Kmeans(5, centroids, n_init=1)
	model3.fit(X, batch_size=1)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_array_almost_equal(model1.centroids, model3.centroids)


def test_kmeans_minibatch_from_samples():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)

	model1 = Kmeans.from_samples(5, X, init='first-k', batch_size=10)
	model2 = Kmeans.from_samples(5, X, init='first-k', batch_size=None)
	model3 = Kmeans.from_samples(5, X, init='first-k', batch_size=10, batches_per_epoch=10)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_raises(AssertionError, assert_array_equal, model1.centroids, model3.centroids)


def test_kmeans_minibatch_fit():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)

	centroids = (numpy.ones((5, 3)).T * numpy.arange(5) * 3).T

	model1 = Kmeans(5, centroids)
	model1.fit(X, batch_size=10)

	model2 = Kmeans(5, centroids)
	model2.fit(X, batch_size=None)

	model3 = Kmeans(5, centroids)
	model3.fit(X, batch_size=10, batches_per_epoch=10)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_raises(AssertionError, assert_array_equal, model1.centroids, model3.centroids)


def test_kmeans_nan_from_samples():
	X = numpy.concatenate([numpy.random.normal(0, 1, size=(25, 3)), 
						   numpy.random.normal(8, 1, size=(25, 3))])
	idxs = numpy.random.choice(numpy.arange(150), replace=False, size=50)
	i, j = idxs / 3, idxs % 3
	X[i, j] = numpy.nan

	model = Kmeans.from_samples(2, X, init='random')
	centroids = [[ 7.836165,  8.552842,  7.732688],
       			 [ 0.28689 , -0.1444  , -0.55996 ]]

	assert_array_almost_equal(model.centroids, centroids)

	model = Kmeans.from_samples(2, X, init='first-k')
	centroids = [[ 0.284408, -0.153738, -0.57606 ],
       			 [ 7.676938,  8.371684,  7.660732]]

 	assert_array_almost_equal(model.centroids, centroids)
	
	model = Kmeans.from_samples(2, X, init='kmeans++')
	centroids = [[ 0.28689 , -0.1444  , -0.55996 ],
       			 [ 7.836165,  8.552842,  7.732688]]

 	assert_array_almost_equal(model.centroids, centroids)

	model = Kmeans.from_samples(2, X, init='kmeans||')
	centroids = [[ 7.836165,  8.552842,  7.732688],
       			 [ 0.28689 , -0.1444  , -0.55996 ]]

 	assert_array_almost_equal(model.centroids, centroids)


def test_kmeans_fit():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(3)])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(300), replace=False, size=150)
	i, j = idxs / 3, idxs % 3
	X[i, j] = numpy.nan

	centroids = (numpy.ones((3, 3)).T * numpy.arange(3) * 3).T

	model = Kmeans(3, centroids)
	model.fit(X)

	centroids = [[-0.118461,  0.08878 , -0.021584],
     		     [ 3.00117 ,  2.886438,  3.048999],
     		     [ 6.005934,  6.051744,  6.053779]]

	assert_array_almost_equal(model.centroids, centroids)


def test_kmeans_nan_predict():
	X = numpy.concatenate([numpy.random.normal(0, 1, size=(25, 3)), 
						   numpy.random.normal(8, 1, size=(25, 3))])
	idxs = numpy.random.choice(numpy.arange(150), replace=False, size=25)
	i, j = idxs / 3, idxs % 3
	X[i, j] = numpy.nan

	model = Kmeans.from_samples(2, X, init='random')
	y_hat = model.predict(X)
	y = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	assert_array_equal(y, y_hat)

	model = Kmeans.from_samples(2, X, init='first-k')
	y_hat = model.predict(X)
	y = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	assert_array_equal(1-y, y_hat)

	model = Kmeans.from_samples(2, X, init='kmeans++')
	y_hat = model.predict(X)
	y = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	assert_array_equal(y, y_hat)

	model = Kmeans.from_samples(2, X, init='kmeans||')
	y_hat = model.predict(X)
	y = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	assert_array_equal(1-y, y_hat)

def test_kmeans_nan_large_predict():
	X = numpy.concatenate([numpy.random.normal(0, 1, size=(25, 3)), 
						   numpy.random.normal(8, 1, size=(25, 3))])
	idxs = numpy.random.choice(numpy.arange(150), replace=False, size=100)
	i, j = idxs / 3, idxs % 3
	
	X_nan = X.copy()
	X_nan[i, j] = numpy.nan
	y = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1])

	model = Kmeans.from_samples(2, X, init='random')
	y_hat = model.predict(X_nan)
	assert_array_equal(y, y_hat)

	model = Kmeans.from_samples(2, X, init='first-k')
	y_hat = model.predict(X_nan)
	assert_array_equal(y, y_hat)

	model = Kmeans.from_samples(2, X, init='kmeans++')
	y_hat = model.predict(X_nan)
	assert_array_equal(y, y_hat)

	model = Kmeans.from_samples(2, X, init='kmeans||')
	y_hat = model.predict(X_nan)
	y = numpy.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0])

	assert_array_equal(y, y_hat)


def test_kmeans_nan_multiple_init():
	X = numpy.concatenate([numpy.random.normal(i, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(1500), replace=False, size=250)
	i, j = idxs / 3, idxs % 3
	X[i, j] = numpy.nan

	model1 = Kmeans.from_samples(5, X, init='kmeans++', n_init=1)
	model2 = Kmeans.from_samples(5, X, init='kmeans++', n_init=25)
	
	dist1 = model1.distance(X).min(axis=1).sum()
	dist2 = model2.distance(X).min(axis=1).sum()

	assert_greater(dist1, dist2)

	model1 = Kmeans.from_samples(5, X, init='random', n_init=1)
	model2 = Kmeans.from_samples(5, X, init='random', n_init=25)
	
	dist1 = model1.distance(X).min(axis=1).sum()
	dist2 = model2.distance(X).min(axis=1).sum()

	assert_greater(dist1, dist2)

	model1 = Kmeans.from_samples(5, X, init='first-k', n_init=1)
	model2 = Kmeans.from_samples(5, X, init='first-k', n_init=25)
	
	dist1 = model1.distance(X).min(axis=1).sum()
	dist2 = model2.distance(X).min(axis=1).sum()

	assert_equal(dist1, dist2)


def test_kmeans_ooc_nan_from_samples():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(1500), replace=False, size=500)
	i, j = idxs / 3, idxs % 3
	X[i, j] = numpy.nan

	model1 = Kmeans.from_samples(5, X, init='first-k', batch_size=500)
	model2 = Kmeans.from_samples(5, X, init='first-k', batch_size=None)

	assert_array_equal(model1.centroids, model2.centroids)

	model = Kmeans.from_samples(5, X, init='random', batch_size=100)
	model = Kmeans.from_samples(5, X, init='kmeans++', batch_size=100)
	model = Kmeans.from_samples(5, X, init='kmeans||', batch_size=100)


def test_kmeans_ooc_nan_fit():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(1500), replace=False, size=500)
	i, j = idxs / 3, idxs % 3
	X[i, j] = numpy.nan

	centroids = (numpy.ones((5, 3)).T * numpy.arange(5) * 3).T

	model1 = Kmeans(5, centroids, n_init=1)
	model1.fit(X)

	model2 = Kmeans(5, centroids, n_init=1)
	model2.fit(X, batch_size=10)

	model3 = Kmeans(5, centroids, n_init=1)
	model3.fit(X, batch_size=1)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_array_almost_equal(model1.centroids, model3.centroids)


def test_kmeans_minibatch_nan_from_samples():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(1500), replace=False, size=500)
	i, j = idxs / 3, idxs % 3
	X[i, j] = numpy.nan

	model1 = Kmeans.from_samples(5, X, init='first-k', batch_size=10)
	model2 = Kmeans.from_samples(5, X, init='first-k', batch_size=None)
	model3 = Kmeans.from_samples(5, X, init='first-k', batch_size=10, batches_per_epoch=10)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_raises(AssertionError, assert_array_equal, model1.centroids, model3.centroids)


def test_kmeans_minibatch_nan_fit():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(5)])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(1500), replace=False, size=500)
	i, j = idxs / 3, idxs % 3
	X[i, j] = numpy.nan

	centroids = (numpy.ones((5, 3)).T * numpy.arange(5) * 3).T

	model1 = Kmeans(5, centroids)
	model1.fit(X, batch_size=10)

	model2 = Kmeans(5, centroids)
	model2.fit(X, batch_size=None)

	model3 = Kmeans(5, centroids)
	model3.fit(X, batch_size=10, batches_per_epoch=10)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_raises(AssertionError, assert_array_equal, model1.centroids, model3.centroids)
