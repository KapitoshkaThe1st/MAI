import numpy as np

def poslin(x):
    return np.maximum(0, x)

class hamming_nn:
    def fit(self, x_train):
        Q, R = x_train.shape
        self.IW = x_train.T
        self.b1 = np.full((1, Q), R)
        eps = 1 / (Q - 1)
        self.LW = np.full((Q, Q), -eps)
        for i in range(Q):
            self.LW[i][i] = 1.0 

    def predict_single(self, x):
        return poslin(x @ self.LW)

    def predict(self, x, epochs=600):
        n_examples = x.shape[0]
        converged = [False] * n_examples
        n_converged = 0

        cur = x @ self.IW + self.b1
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
    x_to_pr = np.array([[1, 1, -1, 1, -1], [1, -1, 1, 1, 1], [-1, -1, -1, 1, -1]])

    classifier = hamming_nn()
    classifier.fit(x_train)

    predicted = classifier.predict(x_to_pr)
    print(predicted)