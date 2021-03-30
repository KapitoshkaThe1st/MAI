import numpy as np

__eps = 0.001

def deriv1(f, x):
    return (f(x + __eps) - f(x - __eps)) / (2 * __eps)

def deriv2(f, x):
    return (f(x + __eps) + f(x - __eps) - 2*f(x)) / (__eps**2)

def cross_hyperbolic_11_2p1o(a, b, c, q, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi1, psi2, f):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi1(x)
    u[1] = tau * psi2(x) + u[0]

    u[:, 0] = phi1(t)
    u[:, nx-1] = phi2(t)

    sigma = a * tau * tau / (h ** 2)

    if sigma > 1:
        print(f"WARNING: {sigma=} but it should be <= 1 to provide algorithm stability!")

    coef = b * tau**2 / (2 * h) 
    for k in range(1, nt-1):
        for j in range(1, nx-1):
            u[k+1][j] = ((sigma + coef) * u[k][j+1] + (sigma - coef) * u[k][j-1] + (2 + c * tau**2 - 2 * sigma) * u[k][j] + tau**2 * f(x[j], t[k]) + -(1 - q * tau / 2) * u[k-1][j]) / (1 + q * tau / 2)


    return u

def cross_hyperbolic_11_2p2o(a, b, c, q, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi1, psi2, f):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi1(x)
    u[1] = (1 + tau**2 / 2 * c) * u[0] + (tau - tau**2 / 2 * q) * psi2(x) + a * tau**2 / 2 * deriv2(psi1, x) + b * tau**2 / 2 * deriv1(psi1, x) + tau**2 / 2 * f(x, 0)
    u[:, 0] = phi1(t)
    u[:, nx-1] = phi2(t)

    sigma = a * tau * tau / (h ** 2)
    
    if sigma > 1:
        print(f"WARNING: {sigma=} but it should be <= 1 to provide algorithm stability!")


    coef = b * tau**2 / (2 * h) 
    for k in range(1, nt-1):
        for j in range(1, nx-1):
            u[k+1][j] = ((sigma + coef) * u[k][j+1] + (sigma - coef) * u[k][j-1] + (2 + c * tau**2 - 2 * sigma) * u[k][j] + tau**2 * f(x[j], t[k]) + -(1 - q * tau / 2) * u[k-1][j]) / (1 + q * tau / 2)


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
    t_end = 0.5

    h = 0.001
    tau = 0.0001

    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)
    
    t_fix = t_end / 2
    n = round((t_fix - t_begin) / tau)

    u = cross_hyperbolic_11_2p1o(1, 2, -3, 2, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi1, psi2, f)
    u_ = cross_hyperbolic_11_2p2o(1, 2, -3, 2, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi1, psi2, f)

    print(f'{u.shape=}')

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(x, u[n], 'g', label='result')
    ax.plot(x, u_ref(x, t_begin + tau * n), 'r', label='reference')
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
    