import numpy as np
import matplotlib.pyplot as plt

from grnn import grnn
from utils import *

def mse(A, B):
    return ((A - B) ** 2).sum() / A.shape[0]

def rmse(A, B):
    return np.sqrt(mse(A, B))

def phi(t):
    return np.sin(-2*t*t + 7*t)

t_begin = 0
t_end = 3.5
h = 0.01

n_points = int((t_end - t_begin) / h) + 1
x = np.linspace(t_begin, t_end, n_points).reshape((-1,1))

y = phi(x)

x_train, y_train, x_test, y_test, x_valid, y_valid = split_dataset(x, y, 0.0, 0.1)

approximator = grnn()

approximator.fit(x_train, y_train, spread=h)

y_pred = approximator.predict(x_train)
print(f'RMSE on train set: {rmse(y_train, y_pred)}')

fig, ax = plt.subplots()
ax.plot(x_train, y_train, 'g', label='reference')
ax.plot(x_train, y_pred, 'b', label='approximation')

ax.set(xlabel='t1', ylabel='x1', title='Approximation on train set')
ax.grid()
ax.legend()

plt.savefig('5.png')
plt.show()

error = y_pred - y_train
fig, ax = plt.subplots()
ax.plot(x_train, error, 'r', label='error')

ax.set(xlabel='t1', ylabel='x1', title='Approximation on train set')
ax.grid()
ax.legend()

plt.savefig('6.png')
plt.show()

y_pred = approximator.predict(x_valid)

print(f'RMSE on validation set: {rmse(y_valid, y_pred)}')

fig, ax = plt.subplots()
ax.plot(x_valid, y_valid, 'g', label='reference')
ax.plot(x_valid, y_pred, 'b', label='approximation')

ax.set(xlabel='t1', ylabel='x1', title='Approximation on validation set')
ax.grid()
ax.legend()

plt.savefig('7.png')
plt.show()

error = y_pred - y_valid
fig, ax = plt.subplots()
ax.plot(x_valid, error, 'r', label='error')

ax.set(xlabel='t1', ylabel='x1', title='Approximation on validation set')
ax.grid()
ax.legend()

plt.savefig('8.png')
plt.show()

# проверка для разреженных данных
x_train, y_train, x_test, y_test, x_valid, y_valid = split_dataset(x_train, y_train, 0.2, 0)

approximator = grnn()

approximator.fit(x_train, y_train, spread=h)

y_pred = approximator.predict(x_train)
print(f'RMSE on train set with sparse data: {rmse(y_train, y_pred)}')

error = y_pred - y_train

fig, ax = plt.subplots()
ax.plot(x_train, y_train, 'g', label='reference')
ax.plot(x_train, y_pred, 'b', label='approximation')
ax.plot(x_train, error, 'r', label='error')

ax.set(xlabel='t1', ylabel='x1', title='Approximation on train set')
ax.grid()
ax.legend()

plt.savefig('9.png')
plt.show()