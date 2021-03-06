import numpy as np

def sigmoid(t, derivative=False):
    return sigmoid(t) * (1 - sigmoid(t)) if derivative else 1 / (1 + np.exp(-t))

def purelin(t, derivative=False):
    return np.full(t.shape, 1) if derivative else t

def tansig(t, derivative=False):
    return 1 / (np.cosh(t) ** 2) if derivative else np.tanh(t)

def relu(t, derivative=False):
    return np.where(t > 0, 1, 0) if derivative else np.maximum(0, t)

def softmax(t):
    return np.exp(t) / np.exp(t).sum()

def norm2(x):
    return np.sqrt((x**2).sum())

def mse(A, B):
    return ((A - B) ** 2).sum() / A.shape[0]

def rmse(A, B):
    return np.sqrt(mse(A, B))

def normalize(x, area):
    res = x.copy()
    if res.ndim == 1:
        for i, (min, max) in enumerate(area):
            res[i] = (res[i]) / (max - min)
    else:
        for i, (min, max) in enumerate(area):
            res[:,i] = (res[:,i]) / (max - min)
    
    return res

eps = 1e-8

class NeuralNetwork:
    def __init__(self, optimizer='traingd', area=None):
        self.area = area
        self.optimizer = optimizer
        self.W = []
        self.b = []
        self.f = []

    def add_layer(self, n_neurons, input_layer=False, activation_function=sigmoid):
        if input_layer:
            self.prev_layer_n_neurons = n_neurons
        else:
            self.W.append(np.random.sample((self.prev_layer_n_neurons, n_neurons)))
            self.b.append(np.random.random(n_neurons))
            self.f.append(activation_function)
            self.prev_layer_n_neurons = n_neurons

    def traingd(self, x_train, y_train, epochs):
        n_samples = x_train.shape[0]

        n_layers = len(self.W)
        self.a = [0] * n_layers 
        self.o = [0] * n_layers 

        for _ in range(epochs):

            dw = [np.zeros(w.shape) for w in self.W]
            db = [np.zeros(b.shape) for b in self.b]

            for x, y in zip(x_train, y_train):
                y_pred = self.predict(x, training=True)

                for i in range(n_layers-1, -1, -1):
                    o = x if i == 0 else self.o[i-1]

                    if i == n_layers-1:
                        delta = ((y_pred - y) * self.f[-1](self.a[-1], derivative=True))
                    else:
                        delta = (delta @ self.W[i+1].T) * self.f[i](self.a[i], derivative=True)

                    dw[i] += o[np.newaxis].T @ delta[np.newaxis]
                    db[i] += delta

            for i in range(n_layers):
                self.W[i] -= self.lr * dw[i] / n_samples
                self.b[i] -= self.lr * db[i] / n_samples

    def trainrp(self, x_train, y_train, epochs):
        n_layers = len(self.W)
        self.a = [0] * n_layers 
        self.o = [0] * n_layers 

        eta_minus = 0.5
        eta_plus = 1.2

        dij_max = 50
        dij_min = 1e-6

        prev_dedw = [np.full(w.shape, 0.001) for w in self.W]
        prev_dedb = [np.full(b.shape, 0.001) for b in self.b]

        prev_dw = [np.full(w.shape, 0) for w in self.W]
        prev_db = [np.full(b.shape, 0) for b in self.b]

        dij_w = [np.full(w.shape, 0.1) for w in self.W]
        dij_b = [np.full(b.shape, 0.1) for b in self.b]

        for ep in range(epochs):
            print(f'epoch: {ep} RMSE: {rmse(y_train, self.predict(x_train, training=True))}')
            # dedw, dedb -- градиенты ошибки по весам и байесам соответственно
            dedw = [np.zeros(w.shape) for w in self.W]
            dedb = [np.zeros(b.shape) for b in self.b]

            for x, y in zip(x_train, y_train):
                y_pred = self.predict(x, training=True)

                delta = ((y_pred - y) * self.f[-1](self.a[-1], derivative=True))

                dedw[-1] += self.o[-2][np.newaxis].T @ delta[np.newaxis]
                dedb[-1] += delta

                for i in range(n_layers-2, -1, -1):
                    o = x if i == 0 else self.o[i-1]
                    delta = self.f[i](self.a[i], derivative=True) * (delta @ self.W[i+1].T)
                    dedw[i] += o[np.newaxis].T @ delta[np.newaxis]
                    dedb[i] += delta

            for i in range(n_layers):

                prev_cur_dedw = (dedw[i] * prev_dedw[i]) >= 0
                prev_cur_dedb = (dedb[i] * prev_dedb[i]) >= 0

                dij_w[i] = np.clip(np.where(prev_cur_dedw, eta_plus, eta_minus) * dij_w[i], dij_min, dij_max)
                dij_b[i] = np.clip(np.where(prev_cur_dedb, eta_plus, eta_minus) * dij_b[i], dij_min, dij_max)

                prev_dw[i] = np.where(prev_cur_dedw, -np.sign(dedw[i]) * dij_w[i], -prev_dw[i])
                prev_db[i] = np.where(prev_cur_dedb, -np.sign(dedb[i]) * dij_b[i], -prev_db[i])

                self.W[i] += prev_dw[i]
                self.b[i] += prev_db[i]

                # если произведение производных меньше 0, то "обнуляем" производную
                prev_dedw[i] = np.where(prev_cur_dedw > 0, dedw[i], 0)
                prev_dedb[i] = np.where(prev_cur_dedb > 0, dedb[i], 0)

    def traingdx(self, x_train, y_train, epochs):
        n_samples = x_train.shape[0]

        n_layers = len(self.W)
        self.a = [0] * n_layers 
        self.o = [0] * n_layers 

        vw = [np.zeros(w.shape) for w in self.W]
        vb = [np.zeros(b.shape) for b in self.b]

        cache_wgrad = [np.zeros(w.shape) for w in self.W]
        cache_bgrad = [np.zeros(b.shape) for b in self.b]

        for ep in range(epochs):
            print(f'epoch: {ep} RMSE: {rmse(y_train, self.predict(x_train, training=True))}')
            dw = [np.zeros(w.shape) for w in self.W]
            db = [np.zeros(b.shape) for b in self.b]

            for x, y in zip(x_train, y_train):
                y_pred = self.predict(x, training=True)

                for i in range(n_layers-1, -1, -1):
                    o = x if i == 0 else self.o[i-1]

                    if i == n_layers-1:
                        delta = ((y_pred - y) * self.f[-1](self.a[-1], derivative=True))
                    else:
                        delta = (delta @ self.W[i+1].T) * self.f[i](self.a[i], derivative=True)

                    dw[i] += o[np.newaxis].T @ delta[np.newaxis]
                    db[i] += delta

            beta = 0.8

            for i in range(n_layers):
                db[i] /= n_samples
                dw[i] /= n_samples

                # моментики
                vw[i] = beta * vw[i] + (1 - beta) * dw[i]
                vb[i] = beta * vb[i] + (1 - beta) * db[i]

                # как в ADAGRAD'е
                cache_wgrad[i] += dw[i] ** 2
                cache_bgrad[i] += db[i] ** 2

                self.W[i] -= self.lr *  1 / (np.sqrt(cache_wgrad[i]) + eps) * vw[i]
                self.b[i] -= self.lr * 1 / (np.sqrt(cache_bgrad[i]) + eps) * vb[i]
                
    def trainoss(self, x_train, y_train, epochs):
        n_samples = x_train.shape[0]

        n_layers = len(self.W)
        self.a = [0] * n_layers 
        self.o = [0] * n_layers 

        prev_step_w = dw = [np.zeros(w.shape) for w in self.W]
        prev_step_b = [np.zeros(b.shape) for b in self.b]

        prev_dw = dw = [np.zeros(w.shape) for w in self.W]
        prev_db = [np.zeros(b.shape) for b in self.b]

        n_weights = 0
        for i in range(n_layers):
            n_weights += self.W[i].size + self.b[i].size

        for epoch in range(epochs):
            print(f'epoch: {epoch} RMSE: {rmse(y_train, self.predict(x_train, training=True))}')
            rmse_val = rmse(self.predict(x_train), y_train)

            if rmse_val < eps:
                break

            dw = [np.zeros(w.shape) for w in self.W]
            db = [np.zeros(b.shape) for b in self.b]

            for x, y in zip(x_train, y_train):
                y_pred = self.predict(x, training=True)

                for i in range(n_layers-1, -1, -1):
                    o = x if i == 0 else self.o[i-1]

                    if i == n_layers-1:
                        delta = ((y_pred - y) * self.f[-1](self.a[-1], derivative=True))
                    else:
                        delta = (delta @ self.W[i+1].T) * self.f[i](self.a[i], derivative=True)

                    dw[i] += o[np.newaxis].T @ delta[np.newaxis]
                    db[i] += delta

            for i in range(n_layers):
                dw[i] /= n_samples
                db[i] /= n_samples

                if epoch == 0:
                    dwi = -dw[i]
                    dbi = -db[i]

                    prev_dw[i] = dw[i]
                    prev_db[i] = db[i]

                    prev_step_w[i] = self.lr * dwi
                    prev_step_b[i] = self.lr * dbi

                    self.W[i] += prev_step_w[i]
                    self.b[i] += prev_step_b[i]
                else:
                    yw = dw[i] - prev_dw[i] 
                    yb = db[i] - prev_db[i] 

                    yy = (yw ** 2).sum() + (yb ** 2).sum()
 
                    sy = np.clip((prev_step_w[i] * yw).sum() + (prev_step_b[i] * yb).sum(), 1e-6, 100)

                    sg = (prev_step_w[i] * dw[i]).sum() + (prev_step_b[i] * db[i]).sum()

                    yg = (dw[i] * yw).sum() + (db[i] * yb).sum()

                    Ac = -(1 + yy / sy) * sg / sy + yg / sy

                    Bc = sg / sy

                    dwi = -dw[i] + Ac * prev_step_w[i] + Bc * yw 
                    dbi = -db[i] + Ac * prev_step_b[i] + Bc * yb  

                    prev_dw[i] = dw[i]
                    prev_db[i] = db[i]

                    prev_W = self.W[i]
                    prev_B = self.b[i]

                    def func(alpha):
                        self.W[i] = prev_W + alpha * dwi
                        self.b[i] = prev_B + alpha * dbi

                        res =  rmse(y_train, self.predict(x_train, training=True))

                        self.W[i] = prev_W
                        self.b[i] = prev_B

                        return res

                    # простейший линейный поиск минимума
                    min_rmse = 1000
                    best_alpha = 0
                    for a in np.linspace(0, 2, 100):
                        val = func(a)
                        if val < min_rmse:
                            min_rmse = val
                            best_alpha = a
                    
                    prev_step_w[i] = best_alpha * dwi
                    prev_step_b[i] = best_alpha * dbi

                    self.W[i] += prev_step_w[i]
                    self.b[i] += prev_step_b[i]

    def fit(self, x_train, y_train, learning_rate=0.01, epochs=200):
        x_normalized = x_train
        if self.area:
            x_normalized = normalize(x_train, self.area)

        self.lr = learning_rate
        if self.optimizer == 'traingd':
            self.traingd(x_normalized, y_train, epochs)
        elif self.optimizer == 'trainrp':
            self.trainrp(x_normalized, y_train, epochs)
        elif self.optimizer == 'traingdx':
            self.traingdx(x_normalized, y_train, epochs)
        elif self.optimizer == 'trainoss':
            self.trainoss(x_normalized, y_train, epochs)
        else:
            raise Exception(f"There is no optimizer '{self.optimizer}''")

    def predict(self, x, training=False):
        res = x
        if self.area and not training:
            res = normalize(x, self.area)

        n_layers = len(self.W)
        for i in range(n_layers):
            a = np.dot(res, self.W[i]) + self.b[i]
            self.a[i] = a
            res = self.f[i](a)
            self.o[i] = res
        
        return res

    def weights(self):
        return self.W
    
    def biases(self):
        return self.b

if __name__ == "__main__":
    np.random.seed(0)

    X = np.array([[1, -1, -1], [1, 1, -1]])
    Y = np.array([[-0.5, 0.5], [0.5, -0.5]])

    x_train = X
    y_train = Y
    
    classifier = NeuralNetwork(optimizer='traingdx')

    classifier.add_layer(3, input_layer=True)
    classifier.add_layer(5, activation_function=tansig)
    classifier.add_layer(2, activation_function=tansig)

    classifier.fit(x_train, y_train, learning_rate=0.1, epochs=300)

    print()
    y_pred = classifier.predict(x_train)
    print(y_pred)
    print(f'RMSE={rmse(y_train, y_pred)}')

    