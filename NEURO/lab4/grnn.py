import numpy as np

def radbas(x):
    return np.exp(-(x ** 2))

eps = 1e-309

class grnn:
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

    def predict(self, x):
        n_samples = x.shape[0]
            
        res = []
        for i in range(n_samples):
            dist = np.zeros(self.n_train)
            for j in range(self.n_train):
                dist[j] = np.sqrt(((x[i] - self.x_train[j]) ** 2).sum())

            a1 = radbas(0.8326 / self.spread * dist)

            a1_sum = a1.sum()
            if a1_sum < eps:
                a1_sum = eps

            res.append(a1 @ self.y_train / a1_sum)

        return np.array(res)

if __name__ == '__main__':
    x_train = np.array([[0], [1], [2], [3], [4], [5]])
    y_train = np.sin(x_train)
    print(f'{y_train=}')

    classifier = grnn()

    classifier.fit(x_train, y_train, spread=0.3)

    res = classifier.predict(x_train)

    print(f'{res=}')