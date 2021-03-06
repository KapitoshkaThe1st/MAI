import numpy as np

def radbas(x):
    return np.exp(-(x ** 2))

class rbf:
    def fit(self, x_train, y_train, spread=2):
        class_number = np.argmax(y_train, axis=1)
        y_train_sorted_index = class_number.argsort()

        self.unique_y = np.unique(y_train, axis=0)
        unique_y_sorted_index = np.argsort(np.argmax(self.unique_y, axis=1))
        self.unique_y = self.unique_y[unique_y_sorted_index]

        self.n_train = x_train.shape[0]
        self.n_classes = np.unique(class_number).shape[0]

        self.x_train = x_train[y_train_sorted_index]
        self.y_train = y_train[y_train_sorted_index]

        self.spread = spread

        A = np.zeros((self.n_train, self.n_train))
        for i in range(self.n_train):
            for j in range(i+1, self.n_train):
                dist = np.sqrt(((self.x_train[i] - self.x_train[j]) ** 2).sum())
                A[i][j] = dist
                A[j][i] = dist

        A = radbas(0.8326 / self.spread * A)
        A_stacked = np.hstack([A, np.ones(self.n_train)[np.newaxis].T])
        wb = np.linalg.pinv(A_stacked) @ self.y_train

        self.W = wb[:-1:]
        self.b = wb[-1]

    def predict(self, x):
        n_samples = x.shape[0]
            
        res = []
        for i in range(n_samples):
            dist = np.zeros(self.n_train)
            for j in range(self.n_train):
                dist[j] = np.sqrt(((x[i] - self.x_train[j]) ** 2).sum())

            a1 = radbas(0.8326 / self.spread * dist)
            
            res.append(a1 @ self.W + self.b)

        return np.array(res)


if __name__ == '__main__':
    x_train = np.array([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]])
    y_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    classifier = rbf()

    classifier.fit(x_train, y_train)

    res = classifier.predict(np.array([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]]))
    print(f'{res=}')