import numpy as np

class ADALINE:
    def fit_one(self, x, y):
        prediction = self.predict(x)
        error = y - prediction

        self.W += self.lr * np.dot(x[np.newaxis].T, error[np.newaxis])
        self.b += self.lr * error

    def fit(self, x_train, y_train, learning_rate=0.01, epochs=20):
        self.lr = learning_rate
        x_dim = x_train.shape[1]
        y_dim = y_train.shape[1]

        self.W = np.random.sample((x_dim, y_dim))
        self.b = np.random.random(y_dim)

        for i in range(epochs):
            for x, y in zip(x_train, y_train):
                self.fit_one(x, y)

    def predict(self, x_to_predict):
        res = np.dot(x_to_predict, self.W) + self.b
        return res

    def weights(self):
        return self.W
    
    def biases(self):
        return self.b

if __name__ == "__main__":
    x_train_list = [[1, -1, -1], [1, 1, -1]]
    y_train_list = [[-1, 2], [1, -4]]

    x_train = np.array(x_train_list)
    y_train = np.array(y_train_list)

    print(x_train)
    print(y_train)

    classifier = ADALINE()
    classifier.fit(x_train, y_train, learning_rate=0.2, epochs=20)

    print()
    print(classifier.predict(x_train))