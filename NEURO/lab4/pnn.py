import numpy as np

def radbas(x):
    return np.exp(-(x ** 2))

class pnn:
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
        y_train_argmax = np.argmax(self.y_train, axis=1)
        for i in range(n_samples):
            dist = np.zeros(self.n_train)
            for j in range(self.n_train):
                dist[j] = np.sqrt(((x[i] - self.x_train[j]) ** 2).sum())

            a1 = radbas(0.8326 / self.spread * dist)
            
            max = 0
            max_index = 0
            for k in range(self.n_classes):
                cur_sum = a1[y_train_argmax == np.argmax(self.unique_y[k])].sum()
                if cur_sum > max:
                    max = cur_sum
                    max_index = k

            res.append(self.unique_y[max_index])

        return np.array(res)

if __name__ == '__main__':
    x_train = np.array([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]])
    y_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    classifier = pnn()

    classifier.fit(x_train, y_train)

    res = classifier.predict(np.array([[2, 3.001]]))
    print(f'{res=}')
