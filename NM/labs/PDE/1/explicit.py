import numpy as np
import math

def explicit_parabolic_21_2p1o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi(x)

    sigma = a * tau / (h ** 2)
    if sigma > 0.5:
        print(f"WARNING: {sigma=} but it should be <= 0.5 to provide algorithm stability!")

    coef = 1 + c * tau - 2 * sigma

    for k in range(1, nt):
        for j in range(1, nx-1):
            u[k][j] = sigma * (u[k-1][j+1] + u[k-1][j-1]) + coef * u[k-1][j]

        u[k][0] = u[k][1] - h * phi1(t[k])
        u[k][-1] = phi2(t[k])
    
    return u

def explicit_parabolic_21_3p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi(x)

    sigma = a * tau / (h ** 2)
    if sigma > 0.5:
        print(f"WARNING: {sigma=} but it should be <= 0.5 to provide algorithm stability!")

    coef = 1 + c * tau - 2 * sigma

    for k in range(1, nt):
        for j in range(1, nx-1):
            u[k][j] = sigma * (u[k-1][j+1] + u[k-1][j-1]) + coef * u[k-1][j]

        u[k][0] = (2 * h * phi1(t[k]) - 4 * u[k][1] + u[k][2]) / -3
        u[k][-1] = phi2(t[k])
    
    return u

def explicit_parabolic_21_2p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi(x)

    sigma = a * tau / (h ** 2)
    if sigma > 0.5:
        print(f"WARNING: {sigma=} but it should be <= 0.5 to provide algorithm stability!")

    coef = 1 + c * tau - 2 * sigma

    for k in range(1, nt):
        for j in range(1, nx-1):
            u[k][j] = sigma * (u[k-1][j+1] + u[k-1][j-1]) + coef * u[k-1][j]

        u[k][0] = (2 * a / h * u[k][1] + h / tau * u[k-1][0] - phi1(t[k]) * 2 * a) / (2 * a / h + h / tau - c * h)
        u[k][-1] = phi2(t[k])
    
    return u

if __name__ == '__main__':
    c = -0.5
    a = 0.5

    def phi1(t):
        return np.exp((c - a) * t)

    def phi2(t):
        return np.exp((c - a) * t)

    def psi(x):
        return np.sin(x)

    def u_ref(x, t):
        return np.exp((c - a) * t) * np.sin(x)


    x_begin = 0
    x_end = np.pi / 2

    t_begin = 0
    t_end = 2

    h = 0.05
    tau = 0.0005

    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    # u = explicit_parabolic_21_2p1o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi)
    # u = explicit_parabolic_21_3p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi)
    u = explicit_parabolic_21_2p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi)

    n = 1500

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(x, u[n])
    ax.plot(x, u_ref(x, t_begin + tau * n))

    ax.set(xlabel='x', ylabel='u',
        title='title')
    ax.grid()

    fig.savefig("test.png")
    plt.show()
