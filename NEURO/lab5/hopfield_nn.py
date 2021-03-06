import numpy as np

class hopfield_nn:
    def fit(self, x_train):
        n_examples = x_train.shape[0]
        n_neurons = x_train.shape[1]
        self.W = np.zeros((n_neurons, n_neurons))
        for ex in x_train:
            self.W += np.outer(ex, ex)
        self.W /= n_neurons

    def predict_single(self, x):
        print(f'{x.shape=}')
        print(f'{self.W.shape=}')

        return np.sign(x @ self.W)

    def predict(self, x, epochs=600):
        n_examples = x.shape[0]
        converged = [False] * n_examples
        n_converged = 0

        cur = np.copy(x)
        for _ in range(epochs):
            for i in range(n_examples):
                if converged[i]:
                    continue

                pred = self.predict_single(cur[i])

                if (cur[i] == pred).all():
                    converged[i] = True
                    n_converged += 1

                if n_converged == n_examples:
                    break

                cur[i] = pred

        return cur

if __name__ == '__main__':
    x_train = np.array([[1, 1, -1, -1, -1], [1, 1, 1, 1, 1], [-1, -1, -1, -1, -1]])
    x_to_pr = np.array([[1, 1, -1, 1, -1], [1, -1, 1, 1, 1], [-1, 1, -1, 1, -1]])

    classifier = hopfield_nn()
    classifier.fit(x_train)

    predicted = classifier.predict(x_to_pr)
    print(predicted)