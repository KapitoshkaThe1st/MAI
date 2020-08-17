import math as m

inf = 100

E = 2e11
ro = 7700

# a = m.sqrt(E / ro)
a = 1

l = 5
eps = 0.01

z = 1

def mu1(t):
    return m.sin(z*t)

def f(x, t):
    return m.sin(z**2 * t)

def integral(f, a, b, h=eps):
    w = b-a
    n = m.ceil(w / h)
    if n == 0:
        # return w * f(a + w/2)
        return 0
    hh = w / n
    
    s = f(a) * 0.5
    for i in range(1, n):
        s += f(a + hh*i)
    s += f(b) * 0.5

    return s * hh

def series(f, a, b):
    s = 0.0
    for i in range(a, b+1):
        s += f(i)

    return s

def deriv(f, x, h=eps):
    return (f(x + h) - f(x - h)) / (2 *h)

def deriv2(f, x, h=eps):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h*h)

def g1(x):
    # return -mu1(0)
    return 0

def g2(x):
    return -1

def int_g1(k):
    def ff(xi):
        return g1(xi) * m.sin((2*k+1)/(2*l)*m.pi * xi)
    return integral(ff, 0, l)

def int_g2(k):
    def ff(xi):
        return g2(xi) * m.sin((2*k+1)/(2*l)*m.pi * xi)
    return integral(ff, 0, l)

def v(x, t):
    def ff(k):
        ig2 = int_g2(k)
        ig1 = int_g1(k)
        # print(f"ig1:{ig1} ig2:{ig2}")
        return (4 / ((2*k+1)*m.pi*a) * ig2 * m.sin((2*k+1)/(2*l)*m.pi*a*t) + 2/l * ig1 * m.cos((2*k+1)/(2*l)*m.pi*a*t)) * m.sin((2*k+1)/(2*l)*m.pi*x)
    return series(ff, 0, inf)

def int_f(tau, k):
    def ff(xi):
        return f(xi, tau) * m.sin((2*k+1)/(2*l)*m.pi * xi)
    return integral(ff, 0, l)

def int_int(t, k):
    def ff(tau):
        return int_f(tau, k) * m.sin((2*k+1)/(2*l)*m.pi*a*(t-tau))
    return integral(ff, 0, t)

def w(x, t):
    def ff(k):
        return int_int(t, k) / (2*k+1) * m.sin((2*k+1)/(2*l)*m.pi * x)
    return 4 / (a*m.pi) * series(ff, 0, inf)

def u1(x, t):
    return mu1(t)

def u(x, t):
    u1v = u1(x, t)
    vv = v(x, t)
    wv = w(x, t)

    print(f"u1: {u1v} v: {vv} w: {wv}")
    return u1v + vv + wv

import matplotlib.pyplot as plt
import numpy as np

max_t = 3
n = 5

tvals = np.linspace(eps, max_t-eps, num=n)
xvals = np.linspace(eps, l-eps, num=n)

print(tvals)
print(xvals)

# yvals = [u(0, t) for t in tvals]
# yvals2 = [mu1(t) + 1 for t in tvals]

# plt.plot(tvals, yvals, color='green')
# plt.plot(tvals, yvals2, color='red')
# plt.show()

# yvals = [deriv(lambda x: u(x, t), l) for t in tvals]
# yvals2 = [0 for t in tvals]

# plt.plot(tvals, yvals, color='green')
# plt.plot(tvals, yvals2, color='red')
# plt.show()

# yvals = [u(x, 2) for x in xvals]
# # yvals2 = [0.1 for x in xvals]

# plt.plot(xvals, yvals, color='green')
# # plt.plot(xvals, yvals2, color='red')
# plt.show()

# yvals = [deriv(lambda t: u(x, t), 0) for x in xvals]
# yvals2 = [0.01 for x in xvals]

# plt.plot(xvals, yvals, color='green')
# plt.plot(xvals, yvals2, color='red')
# plt.show()

print()

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from cycler import cycler

colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
          for i in range(1000)]
plt.rc('axes', prop_cycle=(cycler('color', colors)))

print("[", end='')
plt.title("Зависимость температуры u от координаты x при различных t")
for i in range(3):
    if i > 0:
        print(',')
    t = i

    yvals = [u(x, t) for x in xvals]
    print(yvals, end='')
    plt.plot(xvals, yvals, label=f"t={t}")

print("]")
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.tight_layout()
plt.show()
