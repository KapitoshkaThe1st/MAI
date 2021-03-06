import random
from functools import reduce

import numpy as np

def treshold_function(x):
    return 0 if x < 0 else 1

class Perceptron:
    def __init__(self, activation_func=treshold_function):
        self.activation_func = activation_func

    def fit(self, x, y, *, n_epochs=20):
        w_len = len(x[0])
        self.w = np.random.sample(w_len)
        self.b = np.random.sample()
        for _ in range(n_epochs):
            for i, train_example in enumerate(x):
                error = y[i] - self.predict(train_example)
                self.w += error * train_example
                self.b += error

    def predict(self, x):
        return self.activation_func(np.dot(self.w, x) + self.b)

    def get_weights(self):
        return *self.w, self.b

if __name__ == "__main__":
    pass
