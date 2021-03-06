import numpy as np

def row_wise_equal(a, b):
    return np.sum(np.all(a == b, axis=1))

def split_dataset(X, Y, test, valid=0):
    n_samples = X.shape[0]
    train_size = round(n_samples * (1 - test - valid))
    test_size = round(n_samples * test)
    valid_size = round(n_samples * valid)

    return X[:train_size], Y[:train_size], X[train_size:train_size + test_size], Y[train_size:train_size + test_size], X[-valid_size:], Y[-valid_size:]