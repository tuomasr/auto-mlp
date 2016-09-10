import numpy as np


def stand_data(x):
    """ Standardize each column of x to have zero mean and unit variance
    """
    return (x-x.mean(axis=0))/x.std(axis=0)


def norm_data(x, lb, ub):
    """ Normalize each column of x to the interval [lb, ub]
    """
    # normalize to range [0, 1]
    x = (x-x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    # scale the data to [lb, ub]
    return (ub-lb)*x + lb


def norm_data_reverse(x, lb, ub, x_min, x_max):
    """ Scale normalized data in the interval [lb, ub] back to original values
    """
    return (x_max-x_min)/(ub-lb)*(x-lb) + x_min


def stand_data_reverse(x, x_mean, x_std):
    """ Scale normalized data in the interval [lb, ub] back to original values
    """
    return x*x_std+x_mean


def whiten(X, fudge=1E-18):
    # the matrix X should be observations-by-components
    # get the covariance matrix
    Xcov = np.dot(X.T, X)

    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d+fudge))

    # whitening matrix
    W = np.dot(np.dot(V, D), V.T)

    # multiply by the whitening matrix
    X_white = np.dot(X, W)

    return X_white, W


def train_test_split(X, y, test_size=0.1, report_shapes=False):
    """ Split data to training and testing parts
    """

    # leave the last test_size percent of the data for testing
    n_trn = int(X.shape[0]*(1-test_size))

    X_train = X[:n_trn, :]
    X_test = X[n_trn:, :]
    y_train = y[:n_trn]
    y_test = y[n_trn:]

    # report the dimensions of the train and test datasets
    if report_shapes:
        print('X_train dimensions:', X_train.shape)
        print('X_test dimensions:', X_test.shape)
        print('y_train dimensions:', y_train.shape)
        print('y_test dimensions:', y_test.shape)

    return (X_train, y_train), (X_test, y_test)
