from linear_NN import ADALINE

import numpy as np
import math as m
import matplotlib.pyplot as plt

# --- Данные для задания (вариант №8) ---
t_begin1 = 1
t_end1 = 6
h1 = 0.025

def phi1(t):
    return np.sin(t*t - 10*t + 3)

t_begin2 = 0
t_end2 = 3.5
h2 = 0.01

def phi2(t):
    return np.sin(-2*t*t + 7*t)

def phi(t):
    return 1/8 * np.sin(-2*t*t + 7*t - m.pi)

# --- Вспомогательные функции ---

def gen_data(x, D):
    x_train_list = []
    y_train_list = []
    for i in range(D, x.shape[0]-1):
        x_train_list.append(x[i-D:i])
        y_train_list.append([x[i+1]])

    x_train = np.array(x_train_list)
    y_train = np.array(y_train_list)

    return x_train, y_train

def gen_filter_data(x, y, D):
    x_train_list = []
    y_train_list = []
    for i in range(D, x.shape[0]):
        x_train_list.append(x[i-D:i])
        y_train_list.append([y[i]])

    x_train = np.array(x_train_list)
    y_train = np.array(y_train_list)

    return x_train, y_train

# --- Метрики ---

def mse(A, B):
    return ((A - B) ** 2).sum() / A.shape[0]

def rmse(A, B):
    return np.sqrt(mse(A, B))

# --- Задание №1 ---
print('--- TASK 1 ---')

n1 = int((t_end1 - t_begin1) / h1) + 1
t1 = np.linspace(t_begin1, t_end1, num=n1)

x1 = phi1(t1)

D = 5
x_train, y_train = gen_data(x1, D)

approximator = ADALINE()
approximator.fit(x_train, y_train, learning_rate=0.01, epochs=50)

print("WEIGHTS:")
print(approximator.weights())
print("BIASES:")
print(approximator.biases())

predicted = approximator.predict(x_train)
expected = y_train

rmse_metric = rmse(predicted, expected)
print(f'RMSE: {rmse_metric}')

fig, ax = plt.subplots()
ax.plot(t1[D:-1], expected, 'g', label='reference')
ax.plot(t1[D:-1], predicted, 'b', label='predicted')

ax.set(xlabel='t1', ylabel='x1', title='Approximation')
ax.grid()
ax.legend()

plt.savefig('ex1.png')

# --- Задание №2 ---
print('--- TASK 2 ---')

D = 3
x_train, y_train = gen_data(x1, D)

predictor = ADALINE()

predictor.fit(x_train, y_train, epochs=600)
print("WEIGHTS:")
print(approximator.weights())
print("BIASES:")
print(approximator.biases())

predicted = predictor.predict(x_train)
expected = y_train

error = predicted - expected

fig, ax = plt.subplots()
ax.plot(t1[D:-1], expected, 'g', label='reference')
ax.plot(t1[D:-1], predicted, 'b', label='predicted')
ax.plot(t1[D:-1], error, 'r', label='error')

ax.set(xlabel='t1', ylabel='x1', title='Approximation after training')
ax.grid()
ax.legend()

plt.savefig('ex2.png')

rmse_metric = rmse(predicted, expected)
print(f'RMSE on train data: {rmse_metric}')

print(f'{D=}')

n_append = 10
t1 = np.append(t1, np.array([t_end1 + (i+1) * h1 for i in range(n_append)]))
x1 = phi1(t1)

x, y = gen_data(x1, D)

predicted = predictor.predict(x)
expected = y

fig, ax = plt.subplots()

ax.plot(t1[-n_append:], expected[-n_append:], 'g', label='reference')
ax.plot(t1[-n_append:], predicted[-n_append:], 'b', label='predicted')

ax.set(xlabel='t1', ylabel='x1', title='Prediction for next 10 measurments')
ax.grid()
ax.legend()

rmse_metric = rmse(expected[-n_append:], predicted[-n_append:])
print(f'RMSE on appended data: {rmse_metric}')

plt.savefig('ex3.png')

# задание № 3
print('--- TASK 3 ---')

n2 = int((t_end2 - t_begin2) / h2) + 1
print(f'{n2=}')
t2 = np.linspace(t_begin2, t_end2, num=n2)

x2 = phi2(t2)
y2 = phi(t2)

D = 4
x_train, y_train = gen_filter_data(x2, y2, D)

print(x_train[:5])
print(y_train[:5])

adaptive_filter = ADALINE()
adaptive_filter.fit(x_train, y_train, learning_rate=0.01, epochs=50)

print("WEIGHTS:")
print(approximator.weights())
print("BIASES:")
print(approximator.biases())

predicted = adaptive_filter.predict(x_train)
expected = y_train

error = predicted - expected

fig, ax = plt.subplots()
ax.plot(t2[D:], expected, 'g', label='reference')
ax.plot(t2[D:], predicted, 'b', label='filtered')
ax.plot(t2[D:], error, 'r', label='error')

ax.set(xlabel='t1', ylabel='x1', title='')
ax.grid()
ax.legend()

plt.savefig('ex4.png')

rmse_metric = rmse(predicted, expected)
print(f'RMSE: {rmse_metric}')