import numpy as np

__eps = 0.001

def deriv1(f, x):
    return (f(x + __eps) - f(x - __eps)) / (2 * __eps)

def deriv2(f, x):
    return (f(x + __eps) + f(x - __eps) - 2*f(x)) / (__eps**2)

def tdma(a, b, c, d):
    n = b.shape[0]

    p = np.zeros(n-1)
    q = np.zeros(n)

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        denum = a[i-1] * p[i-1] + b[i]

        if i < n-1:
            p[i] = -c[i] / denum
        q[i] = (d[i] - a[i-1] * q[i-1]) / denum

    res = np.zeros(n)
    res[n-1] = q[n-1]

    for i in range(n-2, -1, -1):
        res[i] = p[i] * res[i+1] + q[i]

    return res

def implicit_hyperbolic_11_2p1o(a, b, c, q, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi1, psi2, f):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi1(x)
    u[1] = tau * psi2(x) + u[0]

    a_coef = b / (2 * h) - a / (h ** 2)
    b_coef = 1 / (tau ** 2) + 2 * a / (h ** 2) + q / (2 * tau) - c
    c_coef = -b / (2 * h) - a / (h ** 2)

    A = np.full(nx-3, a_coef)
    B = np.full(nx-2, b_coef)
    C = np.full(nx-3, c_coef)
    D = np.zeros(nx-2)

    for k in range(1, nt-1):
        D[0] = 2 / (tau ** 2) * u[k][1] - 1 / (tau ** 2) * u[k-1][1] + q / (2 * tau) * u[k-1][1] + f(x[1], t[k+1]) - a_coef * phi1(t[k+1])
        for j in range(2, nx-2):
            D[j-1] = 2 / (tau ** 2) * u[k][j] - 1 / (tau ** 2) * u[k-1][j] + q / (2 * tau) * u[k-1][j] + f(x[j], t[k+1])
        D[nx-3] = 2 / (tau ** 2) * u[k][nx-2] - 1 / (tau ** 2) * u[k-1][nx-2] + q / (2 * tau) * u[k-1][nx-2] + f(x[nx-2], t[k+1]) - c_coef * phi2(t[k+1])

        u[k+1, 1:-1] = tdma(A, B, C, D)
        u[k+1][0] = phi1(t[k+1])
        u[k+1][nx-1] = phi2(t[k+1])

    return u

def implicit_hyperbolic_11_2p2o(a, b, c, q, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi1, psi2, f):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi1(x)
    u[1] = (1 + tau**2 / 2 * c) * u[0] + (tau - tau**2 / 2 * q) * psi2(x) + a * tau**2 / 2 * deriv2(psi1, x) + b * tau**2 / 2 * deriv1(psi1, x) + tau**2 / 2 * f(x, 0)

    a_coef = b / (2 * h) - a / (h ** 2)
    b_coef = 1 / (tau ** 2) + 2 * a / (h ** 2) + q / (2 * tau) - c
    c_coef = -b / (2 * h) - a / (h ** 2)

    A = np.full(nx-3, a_coef)
    B = np.full(nx-2, b_coef)
    C = np.full(nx-3, c_coef)
    D = np.zeros(nx-2)

    for k in range(1, nt-1):
        D[0] = 2 / (tau ** 2) * u[k][1] - 1 / (tau ** 2) * u[k-1][1] + q / (2 * tau) * u[k-1][1] + f(x[1], t[k+1]) - a_coef * phi1(t[k+1])
        for j in range(2, nx-2):
            D[j-1] = 2 / (tau ** 2) * u[k][j] - 1 / (tau ** 2) * u[k-1][j] + q / (2 * tau) * u[k-1][j] + f(x[j], t[k+1])
        D[nx-3] = 2 / (tau ** 2) * u[k][nx-2] - 1 / (tau ** 2) * u[k-1][nx-2] + q / (2 * tau) * u[k-1][nx-2] + f(x[nx-2], t[k+1]) - c_coef * phi2(t[k+1])

        u[k+1, 1:-1] = tdma(A, B, C, D)
        u[k+1][0] = phi1(t[k+1])
        u[k+1][nx-1] = phi2(t[k+1])

    return u

if __name__ == '__main__':
    def phi1(t):
        return 0

    def phi2(t):
        return 0

    def psi1(x):
        return 0

    def psi2(x):
        return 2 * np.exp(-x) * np.sin(x)

    def f(x, t):
        return 0

    def u_ref(x, t):
        return np.exp(-t - x) * np.sin(x) * np.sin(2 * t)

    x_begin = 0
    x_end = np.pi

    t_begin = 0
    t_end = 2

    h = 0.01
    tau = 0.01

    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = implicit_hyperbolic_11_2p1o(1, 2, -3, 2, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi1, psi2, f)
    u_ = implicit_hyperbolic_11_2p2o(1, 2, -3, 2, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi1, psi2, f)

    n = 10

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(x, u[n], 'g', label='result')
    ax.plot(x, u_ref(x, t_begin + t[n]), 'r', label='reference')
    ax.legend()

    ax.set(xlabel='x', ylabel='u',
        title='title')
    ax.grid()

    max_err = np.abs(u[n] - u_ref(x, t[n])).max()
    print(f'{max_err=}')
    max_err_ = np.abs(u_[n] - u_ref(x, t[n])).max()
    print(f'{max_err_=}')

    fig.savefig("test.png")
    plt.show()
    