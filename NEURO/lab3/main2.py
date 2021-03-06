import numpy as np
import matplotlib.pyplot as plt

from multilayer_nn import *
from utils import *

def phi(t):
    return np.sin(-2*t*t + 7*t)

t_begin = 0
t_end = 3.5
h = 0.01

n_points = int((t_end - t_begin) / h) + 1
x = np.linspace(t_begin, t_end, n_points).reshape((-1,1))

y = phi(x)

x_train, y_train, x_test, y_test, x_valid, y_valid = split_dataset(x, y, 0.0, 0.1)

approximator = NeuralNetwork(optimizer='traingdx')

approximator.add_layer(1, input_layer=True)
approximator.add_layer(10, activation_function=tansig)
approximator.add_layer(1, activation_function=purelin)

approximator.fit(x_train, y_train, learning_rate=2, epochs=9000)

print("WEIGHTS:")
print(approximator.weights())
print("BIASES:")
print(approximator.biases())

y_pred = approximator.predict(x_train)
print(f'RMSE on train set: {rmse(y_train, y_pred)}')

error = y_pred - y_train

fig, ax = plt.subplots()
ax.plot(x_train, y_train, 'g', label='reference')
ax.plot(x_train, y_pred, 'b', label='approximation')
ax.plot(x_train, error, 'r', label='error')

ax.set(xlabel='t1', ylabel='x1', title='Approximation on train set')
ax.grid()
ax.legend()

plt.savefig('2.png')
plt.show()

y_pred = approximator.predict(x_valid)
print(f'RMSE on validation set: {rmse(y_valid, y_pred)}')

error = y_pred - y_valid

fig, ax = plt.subplots()
ax.plot(x_valid, y_valid, 'g', label='reference')
ax.plot(x_valid, y_pred, 'b', label='approximation')
ax.plot(x_valid, error, 'r', label='error')

ax.set(xlabel='t1', ylabel='x1', title='Approximation on validation set')
ax.grid()
ax.legend()

plt.savefig('3.png')
plt.show()
