import numpy as np

def implicit_parabolic_21_2p1o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi(x)

    sigma = a * tau / (h ** 2)
    if sigma <= 0.5:
        print(f"WARNING: {sigma=} but it should be > 0.5 to provide algorithm stability!")

    for k in range(1, nt):
        A = np.zeros(nx-3)
        B = np.zeros(nx-2)
        C = np.zeros(nx-3)
        D = np.zeros(nx-2)

        B[0] = c * tau - sigma - 1
        C[0] = sigma
        D[0] = sigma * h * phi1(t[k]) - u[k-1][1]

        for i in range(1, nx - 3):
            A[i-1] = sigma
            B[i] =  c * tau - 2 * sigma - 1
            C[i] = sigma
            D[i] = -u[k-1][i + 1]

        A[nx-4] = sigma
        B[nx-3] = c * tau - 2 * sigma - 1
        D[nx-3] = -u[k-1][nx-2] - sigma * phi2(t[k])

        u[k,1:-1] = tdma(A, B, C, D)
        u[k][0] = u[k][1] - h * phi1(t[k])
        u[k][-1] = phi2(t[k])

    return u

def implicit_parabolic_21_3p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi(x)

    sigma = a * tau / (h ** 2)
    if sigma <= 0.5:
        print(f"WARNING: {sigma=} but it should be > 0.5 to provide algorithm stability!")

    for k in range(1, nt):
        A = np.zeros(nx-3)
        B = np.zeros(nx-2)
        C = np.zeros(nx-3)
        D = np.zeros(nx-2)

        B[0] = c * tau - 2 / 3 * sigma - 1
        C[0] = 2 / 3 * sigma
        D[0] = 2 / 3 * sigma * h * phi1(t[k]) - u[k-1][1]

        for i in range(1, nx - 3):
            A[i-1] = sigma
            B[i] =  c * tau - 2 * sigma - 1
            C[i] = sigma
            D[i] = -u[k-1][i + 1]

        A[nx-4] = sigma
        B[nx-3] = c * tau - 2 * sigma - 1
        D[nx-3] = -u[k-1][nx-2] - sigma * phi2(t[k])

        u[k,1:-1] = tdma(A, B, C, D)
        u[k][0] = u[k][1] - h * phi1(t[k])
        u[k][-1] = phi2(t[k])

    return u


def implicit_parabolic_21_2p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi(x)

    sigma = a * tau / (h ** 2)
    if sigma <= 0.5:
        print(f"WARNING: {sigma=} but it should be > 0.5 to provide algorithm stability!")

    for k in range(1, nt):
        A = np.zeros(nx-3)
        B = np.zeros(nx-2)
        C = np.zeros(nx-3)
        D = np.zeros(nx-2)

        B[0] = 2 * a / h + h / tau - c * h
        C[0] = -2 * a / h
        D[0] = h / tau * u[k-1][0] - phi1(t[k]) * 2 * a

        for i in range(1, nx - 3):
            A[i-1] = sigma
            B[i] =  c * tau - 2 * sigma - 1
            C[i] = sigma
            D[i] = -u[k-1][i + 1]

        A[nx-4] = sigma
        B[nx-3] = c * tau - 2 * sigma - 1
        D[nx-3] = -u[k-1][nx-2] - sigma * phi2(t[k])

        u[k,1:-1] = tdma(A, B, C, D)
        u[k][0] = (2 * a / h * u[k][1] + h / tau * u[k-1][0] - phi1(t[k]) * 2 * a) / (2 * a / h + h / tau - c * h)
        u[k][-1] = phi2(t[k])

    return u

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

    h = 0.01
    tau = 0.005

    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    # u = implicit_parabolic_21_2p1o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi)
    # u = implicit_parabolic_21_3p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi)
    u = implicit_parabolic_21_2p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi)

    n = 300

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(x, u[n], 'g', label='result')
    ax.plot(x, u_ref(x, t_begin + tau * n), 'r', label='reference')
    ax.legend()

    ax.set(xlabel='x', ylabel='u',
        title='title')
    ax.grid()

    fig.savefig("test.png")
    plt.show()
