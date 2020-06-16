import math
import numpy as np

from collections import Counter

class LogisticRegression():
    def __init__(self, learning_rate = 0.01, grad_iters=1000):
        self.lr = learning_rate
        self.gi = grad_iters

    def __sigmoid(self, x):
        return 1.0 / (1.0 + np.e ** (-x))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def __add_intercept(self, X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def fit(self, X, y):
        X = self.__add_intercept(X)
        self.w = np.zeros(X.shape[1])

        for _ in range(self.gi):
            h = self.__sigmoid(np.dot(X, self.w))
            g = np.dot(X.T, (h - y)) / y.size
            self.w -= self.lr * g
        pass

    def __predict_probability(self, X):
        X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.__predict_probability(X) >= threshold

class KNN():
    def __init__(self, n_neighbors=5):
        self.nn = n_neighbors
    
    def fit(self, X, y):
        self.X = X
        self.y = y.reshape((y.shape[0], 1))

    def __get_distances(self, p):
        t = self.X - p
        return np.sqrt((t*t).sum(1))

    def predict(self, X):

        n = X.shape[0]
        y_pred = np.zeros((n, 1))
        for i in range(n):
            d = self.__get_distances(X[i])
            y_sorted = self.y[np.argsort(d)].flatten()

            if y_sorted[:self.nn].sum() > self.nn / 2:
                y_pred[i] = 1.0
        return y_pred

class Node():
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTree():
    def __init__(self, max_depth=1, rf=False):
        self.max_depth = max_depth
        self.rf = rf

    def fit(self, X, y, max_features=None):
        self.n_classes_ = len(set(y))
        if not self.rf:
            n_features_ = X.shape[1]
        else:
            ind = np.random.choice(X.shape[0], X.shape[0])
            X, y = X[tuple([ind])], y[tuple([ind])]
            if max_features is None:
                n_features_ = np.sqrt(X.shape[1]).astype(int)
            else:
                n_features_ = max_features
        self.features_ = np.sort(np.random.choice(X.shape[1], n_features_,
                                                  replace=False))
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in self.features_:
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i)
                                 for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


class RandomForest():
    def __init__(self, max_depth=5, n_estimators=100, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.forest = [None] * n_estimators

    def fit(self, X, y):
        for i in range(self.n_estimators):
            self.forest[i] = DecisionTree(
                self.max_depth, rf=True) 
            self.forest[i].fit(X, y)

    def predict(self, X):
        most_common = np.zeros(X.shape[0])
        preds = np.zeros((self.n_estimators, X.shape[0]))
        for i in range(self.n_estimators):
            preds[i] = self.forest[i].predict(X)
        for i in range(len(most_common)):
            most_common[i] = Counter(preds[:, i]).most_common(1)[0][0]
        return most_common.astype(int)