import math as m
import numpy as np

l1 = 2
l2 = 4

def f(x, y):
    c = np.pi / l1
    return (- 1.0 / np.tanh(c * l2) * np.sinh(c * y) + np.cosh(c * y)) * np.sin(c * x)

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

n = 50

x = np.linspace(0, l1, num=n)
y = np.linspace(0, l2, num=n)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

print(Z)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Зависимость искомой функции u(x,y) от x и y')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
plt.tight_layout()


plt.show()