import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

__eps = 1e-6

def bilinear_interpolation(x, y, points):
    points = sorted(points)
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
    def f(x, y):
        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')

        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1) + 0.0)

    result = np.full((x.shape[0], y.shape[0]), 0.0)
    # print(f'bi:{x.shape=} {y.shape=}')
    # print(f'bi:{result.shape=}')
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            # print(f'{i=} {j=}')
            result[i][j] = f(x[i], y[j])

    return result


def libman(a, b, h1, h2, x_begin, x_end, y_begin, y_end, phi1, phi2, psi1, psi2, eps=__eps):
    nx = round((x_end - x_begin) / h1) + 1
    ny = round((y_end - y_begin) / h2) + 1

    x = np.linspace(x_begin, x_end, nx)
    y = np.linspace(y_begin, y_end, ny)
    
    u = np.full((nx, ny), 0.0)

    u[:,0] = psi1(x)
    u[:,ny-1] = psi2(x)

    u[0] = phi1(y)
    u[nx-1] = phi2(y)

    for j in range(ny):
        u[:, j] = np.interp(x, (x[0], x[-1]), (phi1(y[j]), phi2(y[j])))

    for i in range(nx):
        u[i] += np.interp(y, (y[0], y[-1]), (psi1(x[i]), psi2(x[i])))

    u -= bilinear_interpolation(x, y, [(x[0], y[0], psi1(x[0])),
                                       (x[0], y[-1], psi2(x[0])),
                                       (x[-1], y[0], psi1(x[-1])),
                                       (x[-1], y[-1], psi2(x[-1]))])

    u_ = u.copy()

    coef = 1 / (-2 / (h1 ** 2) - 2 / (h2 ** 2) - b)
    a1_coef = a / (2 * h1) - 1 / (h1 ** 2)
    a2_coef = -a / (2 * h1) - 1 / (h1 ** 2)
    b_coef = -1 / (h2 ** 2)
    iters = 0
    while True:
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u_[i][j] = (a1_coef * u[i+1][j] + a2_coef * u[i-1][j] + b_coef * (u[i][j+1] + u[i][j-1])) * coef

        e = np.abs(u - u_).max()
        iters += 1
        if e < eps:
            return u_, iters

        u_[:,0] = psi1(x)
        u_[:,ny-1] = psi2(x)

        u_[0] = phi1(y)
        u_[nx-1] = phi2(y)

        u, u_ = u_, u

    return u, iters


def seidel(a, b, h1, h2, x_begin, x_end, y_begin, y_end, phi1, phi2, psi1, psi2, eps=__eps):
    nx = round((x_end - x_begin) / h1) + 1
    ny = round((y_end - y_begin) / h2) + 1

    x = np.linspace(x_begin, x_end, nx)
    y = np.linspace(y_begin, y_end, ny)

    u = np.full((nx, ny), 0.0)

    u[:,0] = psi1(x)
    u[:,ny-1] = psi2(x)

    u[0] = phi1(y)
    u[nx-1] = phi2(y)

    for j in range(ny):
        u[:, j] = np.interp(x, (x[0], x[-1]), (phi1(y[j]), phi2(y[j])))

    for i in range(nx):
        u[i] += np.interp(y, (y[0], y[-1]), (psi1(x[i]), psi2(x[i])))

    u -= bilinear_interpolation(x, y, [(x[0], y[0], psi1(x[0])),
                                       (x[0], y[-1], psi2(x[0])),
                                       (x[-1], y[0], psi1(x[-1])),
                                       (x[-1], y[-1], psi2(x[-1]))])

    u_ = u.copy()

    coef = 1 / (2 / (h1**2) + 2 / (h2**2) + b)
    iters = 0
    while True:
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u_[i][j] = ((1 / (h1**2) - a / (2 * h1)) * u[i+1][j] + (1 / (h1**2) + a / (2 * h1)) * u_[i-1][j]
                + 1 / (h2**2) * u[i][j+1] + 1 / (h2**2) * u_[i][j-1]) * coef

        e = np.abs(u - u_).max()
        iters += 1
        if e < eps:
            return u_, iters

        u, u_ = u_, u

    return u, iters


def relaxation(a, b, h1, h2, x_begin, x_end, y_begin, y_end, phi1, phi2, psi1, psi2, w, eps=__eps):
    nx = round((x_end - x_begin) / h1) + 1
    ny = round((y_end - y_begin) / h2) + 1

    x = np.linspace(x_begin, x_end, nx)
    y = np.linspace(y_begin, y_end, ny)

    u = np.full((nx, ny), 0.0)

    u[:,0] = psi1(x)
    u[:,ny-1] = psi2(x)

    u[0] = phi1(y)
    u[nx-1] = phi2(y)

    for j in range(ny):
        u[:, j] = np.interp(x, (x[0], x[-1]), (phi1(y[j]), phi2(y[j])))

    for i in range(nx):
        u[i] += np.interp(y, (y[0], y[-1]), (psi1(x[i]), psi2(x[i])))

    u -= bilinear_interpolation(x, y, [(x[0], y[0], psi1(x[0])),
                                       (x[0], y[-1], psi2(x[0])),
                                       (x[-1], y[0], psi1(x[-1])),
                                       (x[-1], y[-1], psi2(x[-1]))])

    u_ = u.copy()

    coef = 1 / (2 / (h1**2) + 2 / (h2**2) + b)
    iters = 0
    while True:
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u_[i][j] = w * ((1 / (h1**2) - a / (2 * h1)) * u[i+1][j] + (1 / (h1**2) + a / (2 * h1)) * u_[i-1][j]
                + 1 / (h2**2) * u[i][j+1] + 1 / (h2**2) * u_[i][j-1]) * coef + (1 - w) * u[i][j]

        e = np.abs(u - u_).max()
        iters += 1
        if e < eps:
            return u_, iters

        u, u_ = u_, u

    return u, iters

# if __name__ == '__main__':

#     def phi1(y):
#         return np.cos(y)
    
#     def phi2(y):
#         return 0

#     def psi1(x):
#         return np.exp(-x) * np.cos(x)
    
#     def psi2(x):
#         return 0

#     def u_ref(x, y):
#         return np.exp(-x) * np.cos(x) * np.cos(y)

#     x_begin = 0
#     x_end = np.pi / 2

#     y_begin = 0
#     y_end = np.pi / 2

#     # h1 = 0.01
#     # h2 = 0.02

#     h1 = 0.05
#     h2 = 0.05

#     nx = round((x_end - x_begin) / h1) + 1
#     ny = round((y_end - y_begin) / h2) + 1

#     x = np.linspace(x_begin, x_end, nx)
#     y = np.linspace(y_begin, y_end, ny)

#     u = libman(-2, -3, h1, h2, x_begin, x_end, y_begin, y_end, phi1, phi2, psi1, psi2)
#     # u = seidel(-2, -3, h1, h2, x_begin, x_end, y_begin, y_end, phi1, phi2, psi1, psi2)
#     # u = relaxation(-2, -3, h1, h2, x_begin, x_end, y_begin, y_end, phi1, phi2, psi1, psi2, 1.7)

#     n1 = nx // 2
#     n2 = ny // 2

#     print(f'max_err={max(np.abs(u[:, n2] - u_ref(x, y[n2])).max(), np.abs(u[n1] - u_ref(x[n1], y)).max())}')

#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots()
#     ax.plot(x, u[:, n2], 'g', label='result')
#     ax.plot(x, u_ref(x, y[n2]), 'r', label='reference')
#     ax.legend()

#     ax.set(xlabel='x', ylabel='u',
#         title='title')
#     ax.grid()
#     plt.show()

#     fig, ax = plt.subplots()
#     ax.plot(y, u[n1], 'g', label='result')
#     ax.plot(y, u_ref(x[n1], y), 'r', label='reference')
#     ax.legend()

#     ax.set(xlabel='y', ylabel='u',
#         title='title')
#     ax.grid()

#     # fig.savefig("test.png")
#     plt.show()
    
if __name__ == '__main__':

    def phi1(y):
        return np.cos(y)
    
    def phi2(y):
        return 0

    def psi1(x):
        return np.cos(x)
    
    def psi2(x):
        return 0

    def u_ref(x, y):
        return np.cos(x) * np.cos(y)

    x_begin = 0
    x_end = np.pi / 2

    y_begin = 0
    y_end = np.pi / 2

    # h1 = 0.01
    # h2 = 0.02

    h1 = 0.05
    h2 = 0.05

    nx = round((x_end - x_begin) / h1) + 1
    ny = round((y_end - y_begin) / h2) + 1

    x = np.linspace(x_begin, x_end, nx)
    y = np.linspace(y_begin, y_end, ny)

    u = libman(0, -2, h1, h2, x_begin, x_end, y_begin, y_end, phi1, phi2, psi1, psi2)
    # u = seidel(-2, -3, h1, h2, x_begin, x_end, y_begin, y_end, phi1, phi2, psi1, psi2)
    # u = relaxation(-2, -3, h1, h2, x_begin, x_end, y_begin, y_end, phi1, phi2, psi1, psi2, 1.7)

    n1 = nx // 2
    n2 = ny // 2

    print(f'max_err={max(np.abs(u[:, n2] - u_ref(x, y[n2])).max(), np.abs(u[n1] - u_ref(x[n1], y)).max())}')

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(x, u[:, n2], 'g', label='result')
    ax.plot(x, u_ref(x, y[n2]), 'r', label='reference')
    ax.legend()

    ax.set(xlabel='x', ylabel='u',
        title='title')
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(y, u[n1], 'g', label='result')
    ax.plot(y, u_ref(x[n1], y), 'r', label='reference')
    ax.legend()

    ax.set(xlabel='y', ylabel='u',
        title='title')
    ax.grid()

    # fig.savefig("test.png")
    plt.show()
    