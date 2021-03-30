import numpy as np
from implicit import tdma

def crank_nicolson_parabolic_21_2p1o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi(x)

    sigma = a * tau / (h ** 2)
    theta = 0.5

    a_coef = lambda k, i: (1 - theta) * (a * (u[k][i-1] - 2 * u[k][i] + u[k][i+1]) / (h*h) + c * u[k][i])

    for k in range(1, nt):
        A = np.zeros(nx-3)
        B = np.zeros(nx-2)
        C = np.zeros(nx-3)
        D = np.zeros(nx-2)

        B[0] = theta * c * tau - theta * sigma - 1
        C[0] = theta * sigma
        D[0] = -u[k-1][1] - tau * a_coef(k-1, 1) + theta * sigma * h * phi1(t[k])

        for i in range(1, nx - 3):
            A[i-1] = theta * sigma
            B[i] = theta * c * tau - 2 * theta * sigma - 1
            C[i] = theta * sigma
            D[i] = -u[k-1][i + 1] - a_coef(k-1, i) * tau

        A[nx-4] = theta * sigma
        B[nx-3] = theta * c * tau - 2 * theta * sigma - 1
        D[nx-3] = -u[k-1][nx-2] - theta * sigma * phi2(t[k]) - a_coef(k-1, nx-2) * tau

        u[k,1:-1] = tdma(A, B, C, D)
        u[k][0] = u[k][1] - h * phi1(t[k])
        u[k][-1] = phi2(t[k])

    return u 

def crank_nicolson_parabolic_21_3p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi(x)

    sigma = a * tau / (h ** 2)
    theta = 0.5

    a_coef = lambda k, i: (1 - theta) * (a * (u[k][i-1] - 2 * u[k][i] + u[k][i+1]) / (h*h) + c * u[k][i])

    for k in range(1, nt):
        A = np.zeros(nx-3)
        B = np.zeros(nx-2)
        C = np.zeros(nx-3)
        D = np.zeros(nx-2)

        B[0] = theta * c * tau - 2 / 3 * theta * sigma - 1
        C[0] = 2 / 3 * theta * sigma
        D[0] = -u[k-1][1] - tau * a_coef(k-1, 1) + 2 / 3 * theta * sigma * h * phi1(t[k])

        for i in range(1, nx - 3):
            A[i-1] = theta * sigma
            B[i] = theta * c * tau - 2 * theta * sigma - 1
            C[i] = theta * sigma
            D[i] = -u[k-1][i + 1] - a_coef(k-1, i) * tau

        A[nx-4] = theta * sigma
        B[nx-3] = theta * c * tau - 2 * theta * sigma - 1
        D[nx-3] = -u[k-1][nx-2] - theta * sigma * phi2(t[k]) - a_coef(k-1, nx-2) * tau

        u[k,1:-1] = tdma(A, B, C, D)
        u[k][0] = (2 * h * phi1(t[k]) - 4 * u[k][1] + u[k][2]) / -3
        u[k][-1] = phi2(t[k])

    return u 

def crank_nicolson_parabolic_21_2p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi):
    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    u = np.full((nt, nx), 0.0)
    u[0] = psi(x)

    sigma = a * tau / (h ** 2)
    theta = 0.5

    a_coef = -theta * a / (h ** 2)
    b_coef = 1 / tau + 2 * theta * a / (h ** 2)
    c_coef = -theta * a / (h ** 2)

    A = np.full(nx-3, a_coef)
    B = np.full(nx-2, b_coef)
    C = np.full(nx-3, c_coef)
    D = np.zeros(nx-2)

    denum = 1 + h**2 / (2 * a) * (1 / tau - c)
    B[0] = a_coef / denum + b_coef

    for k in range(0, nt-1):
        D[0] = (1 - theta) * a / (h**2) * (u[k][2] - 2 * u[k][1] + u[k][0]) + 1 / tau * u[k][1] - a_coef * (h**2 / (2 * a * tau) * u[k][0] - phi1(t[k+1]) * h) / denum
        for j in range(1, nx - 3):
            D[j] = (1 - theta) * a / (h**2) * (u[k][j+1] - 2 * u[k][j] + u[k][j+1]) + 1 / tau * u[k][j]
            
        D[nx-3] = (1 - theta) * a / (h**2) * (u[k][nx-1] - 2 * u[k][nx-2] + u[k][nx-3]) + 1 / tau * u[k][nx-2] - c_coef * phi2(t[k+1])

        u[k+1,1:-1] = tdma(A, B, C, D)
        u[k+1][0] = (u[k+1][1] - phi1(t[k+1]) * h + h**2 / (2 * a * tau) * u[k][0]) / denum
        u[k+1][-1] = phi2(t[k])

        print(f'{A=}')
        print(f'{B=}')
        print(f'{C=}')
        print(f'{D=}')

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

    h = 0.1
    tau = 0.1

    nx = round((x_end - x_begin) / h) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    t = np.linspace(t_begin, t_end, nt)

    # u = crank_nicolson_parabolic_21_2p1o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi)
    # u = crank_nicolson_parabolic_21_3p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi)
    u = crank_nicolson_parabolic_21_2p2o(a, c, x_begin, x_end, t_begin, t_end, h, tau, phi1, phi2, psi)

    n = 1

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