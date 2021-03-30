import numpy as np
import matplotlib.pyplot as plt

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

def alternating_direction_method(a, x_begin, x_end, y_begin, y_end, t_begin, t_end, phi1, phi2, phi3, phi4, psi, f, h1, h2, tau):
    nx = round((x_end - x_begin) / h1) + 1
    ny = round((y_end - y_begin) / h2) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    y = np.linspace(y_begin, y_end, ny)
    t = np.linspace(t_begin, t_end, nt)

    xgrid, ygrid = np.meshgrid(x, y)

    u = np.full((nt, nx, ny), 0.0)
    u[0] = psi(xgrid, ygrid).T

    a1_coef = -a / (h1 ** 2)
    b1_coef = 2 * (1 / tau + a / (h1 ** 2))
    c1_coef = -a / (h1 ** 2)

    A1 = np.full(nx-3, a1_coef)
    B1 = np.full(nx-2, b1_coef)
    C1 = np.full(nx-3, c1_coef)
    D1 = np.zeros(nx-2)

    A1[nx-4] = a1_coef + c1_coef * (2 * h1) / (2 * h1 * (2 * h1 - 3))
    B1[nx-3] = b1_coef - c1_coef * (4 * h1) / (h1 * (2 * h1 - 3))

    a2_coef = -a / (h2 ** 2)
    b2_coef = 2 * (1 / tau + a / (h2 ** 2))
    c2_coef = -a / (h2 ** 2)

    A2 = np.full(ny-3, a2_coef)
    B2 = np.full(ny-2, b2_coef)
    C2 = np.full(ny-3, c2_coef)
    D2 = np.zeros(ny-2)

    A2[ny-4] = a2_coef + c2_coef * (2 * h2) / (2 * h2 * (2 * h2 - 3))
    B2[ny-3] = b2_coef - c2_coef * (4 * h2) / (h2 * (2 * h2 - 3))

    u_ = np.full((nx, ny), 0.0)
    for k in range(nt-1):
        for j in range(1, ny-1):
            D1[0] = a / (h2 ** 2) * (u[k][1][j+1] - 2 * u[k][1][j] + u[k][1][j-1]) + 2 / tau * u[k][1][j] + f(x[1], y[j], t[k] + 0.5 * tau) -a1_coef * phi1(y[j], t[k] + 0.5 * tau)
            for i in range(2, nx-2):
                D1[i-1] = a / (h2 ** 2) * (u[k][i][j+1] - 2 * u[k][i][j] + u[k][i][j-1]) + 2 / tau * u[k][i][j] + f(x[i], y[j], t[k] + 0.5 * tau)

            D1[nx-3] = a / (h2 ** 2) * (u[k][nx-2][j+1] - 2 * u[k][nx-2][j] + u[k][nx-2][j-1]) + 2 / tau * u[k][nx-2][j] + f(x[nx-2], y[j], t[k] + 0.5 * tau) - c1_coef * (2 * h1) / (2 * h1 - 3) * phi2(y[j], t[k] + 0.5 * tau)

            u_[1:-1, j] = tdma(A1, B1, C1, D1)
            u_[0][j] = phi1(y[j], t[k] + 0.5 * tau)
            u_[-1][j] = (2 * h1) / (2 * h1 - 3) * (phi2(y[j], t[k] + 0.5 * tau) - 2 / h1 * u_[-2][j] + 1 / (2 * h1) * u_[-3][j])

        for i in range(nx):
            u_[i][0] = phi3(x[i], t[k] + 0.5 * tau)
            u_[i][-1] = (2 * h2) / (2 * h2 - 3) * (phi4(x[i], t[k] + 0.5 * tau) - 2 / h2 * u_[i][-2] + 1 / (2 * h2) * u_[i][-3])
        
        for i in range(1, nx-1):
            D2[0] = a / (h1 ** 2) * (u_[i+1][1] - 2 * u_[i][1] + u_[i-1][1]) + f(x[i], y[1], t[k] + 0.5 * tau) + 2 / tau * u_[i][1] - a2_coef * phi3(x[i], t[k+1])
            for j in range(2, ny-2):
                D2[j-1] = a / (h1 ** 2) * (u_[i+1][j] - 2 * u_[i][j] + u_[i-1][j]) + f(x[i], y[j], t[k] + 0.5 * tau) + 2 / tau * u_[i][j]

            D2[ny-3] = a / (h1 ** 2) * (u_[i+1][ny-2] - 2 * u_[i][ny-2] + u_[i-1][ny-2]) + f(x[i], y[ny-2], t[k] + 0.5 * tau) + 2 / tau * u_[i][ny-2] - c2_coef * (2 * h2) / (2 * h2 - 3) * phi4(x[i], t[k+1])

            u[k+1][i][1:-1] = tdma(A2, B2, C2, D2)
            u[k+1][i][0] = phi3(x[i], t[k+1])
            u[k+1][i][-1] = (2 * h2) / (2 * h2 - 3) * (phi4(x[i], t[k+1]) - 2 / h2 * u[k+1][i][-2] + 1 / (2 * h2) * u[k+1][i][-3])
        
        for j in range(ny):
            u[k+1][0][j] = phi1(y[j], t[k+1])
            u[k+1][-1][j] = (2 * h1) / (2 * h1 - 3) * (phi2(y[j], t[k+1]) - 2 / h1 * u[k+1][-2][j] + 1 / (2 * h1) * u[k+1][-3][j])

    return u

def fractional_step_method(a, x_begin, x_end, y_begin, y_end, t_begin, t_end, phi1, phi2, phi3, phi4, psi, f, h1, h2, tau):
    nx = round((x_end - x_begin) / h1) + 1
    ny = round((y_end - y_begin) / h2) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    y = np.linspace(y_begin, y_end, ny)
    t = np.linspace(t_begin, t_end, nt)

    xgrid, ygrid = np.meshgrid(x, y)

    u = np.full((nt, nx, ny), 0.0)
    u[0] = psi(xgrid, ygrid).T

    a1_coef = -a / (h1 ** 2)
    b1_coef = 1 / tau + 2 * a / (h1 ** 2)
    c1_coef = -a / (h1 ** 2)

    A1 = np.full(nx-3, a1_coef)
    B1 = np.full(nx-2, b1_coef)
    C1 = np.full(nx-3, c1_coef)
    D1 = np.zeros(nx-2)

    A1[nx-4] = a1_coef + c1_coef * (2 * h1) / (2 * h1 * (2 * h1 - 3))
    B1[nx-3] = b1_coef - c1_coef * (4 * h1) / (h1 * (2 * h1 - 3))

    a2_coef = -a / (h2 ** 2)
    b2_coef = 1 / tau + 2 * a / (h2 ** 2)
    c2_coef = -a / (h2 ** 2)

    A2 = np.full(ny-3, a2_coef)
    B2 = np.full(ny-2, b2_coef)
    C2 = np.full(ny-3, c2_coef)
    D2 = np.zeros(ny-2)

    A2[ny-4] = a2_coef + c2_coef * (2 * h2) / (2 * h2 * (2 * h2 - 3))
    B2[ny-3] = b2_coef - c2_coef * (4 * h2) / (h2 * (2 * h2 - 3))

    u_ = np.full((nx, ny), 0.0)
    for k in range(nt-1):
        for j in range(1, ny-1):
            D1[0] = u[k][1][j] / tau + 0.5 * f(x[1], y[j], t[k]) - a1_coef * phi1(y[j], t[k] + 0.5 * tau)
            for i in range(2, nx-2):
                D1[i-1] = u[k][i][j] / tau + 0.5 * f(x[i], y[j], t[k])

            D1[nx-3] =  u[k][nx-2][j] / tau + 0.5 * f(x[nx-2], y[j], t[k]) - c1_coef * (2 * h1) / (2 * h1 - 3) * phi2(y[j], t[k] + 0.5 * tau)

            u_[1:-1, j] = tdma(A1, B1, C1, D1)
            u_[0][j] = phi1(y[j], t[k] + 0.5 * tau)
            u_[-1][j] = (2 * h1) / (2 * h1 - 3) * (phi2(y[j], t[k] + 0.5 * tau) - 2 / h1 * u_[-2][j] + 1 / (2 * h1) * u_[-3][j])

        for i in range(nx):
            u_[i][0] = phi3(x[i], t[k] + 0.5 * tau)
            u_[i][-1] = (2 * h2) / (2 * h2 - 3) * (phi4(x[i], t[k] + 0.5 * tau) - 2 / h2 * u_[i][-2] + 1 / (2 * h2) * u_[i][-3])
        
        for i in range(1, nx-1):
            D2[0] = u_[i][1] / tau + 0.5 * f(x[i], y[1], t[k+1]) - a2_coef * phi3(x[i], t[k+1])
            for j in range(2, ny-2):
                D2[j-1] = u_[i][j] / tau + 0.5 * f(x[i], y[j], t[k+1])

            D2[ny-3] = u_[i][ny-2] / tau + 0.5 * f(x[i], y[ny-2], t[k+1]) - a2_coef * phi3(x[i], t[k+1]) - c2_coef * (2 * h2) / (2 * h2 - 3) * phi4(x[i], t[k+1])

            u[k+1][i][1:-1] = tdma(A2, B2, C2, D2)
            u[k+1][i][0] = phi3(x[i], t[k+1])
            u[k+1][i][-1] = (2 * h2) / (2 * h2 - 3) * (phi4(x[i], t[k+1]) - 2 / h2 * u[k+1][i][-2] + 1 / (2 * h2) * u[k+1][i][-3])
        
        for j in range(ny):
            u[k+1][0][j] = phi1(y[j], t[k+1])
            u[k+1][-1][j] = (2 * h1) / (2 * h1 - 3) * (phi2(y[j], t[k+1]) - 2 / h1 * u[k+1][-2][j] + 1 / (2 * h1) * u[k+1][-3][j])

    return u

if __name__ == '__main__':
    def f(x, y, t):
        return - x * y * np.sin(t)

    def phi1(y, t):
        return 0

    def phi2(y, t):
        return 0

    def phi3(x, t):
        return 0

    def phi4(x, t):
        return 0
    
    def psi(x, y):
        return x * y

    def u_ref(x, y, t):
        return x * y * np.cos(t)

    x_begin = 0
    x_end = 1

    y_begin = 0
    y_end = 1

    t_begin = 0
    t_end = 1

    # h1 = 0.2
    # h2 = 0.1
    # tau = 0.1
    
    h1 = 0.01
    h2 = 0.01
    tau = 0.005

    nx = round((x_end - x_begin) / h1) + 1
    ny = round((y_end - y_begin) / h2) + 1
    nt = round((t_end - t_begin) / tau) + 1

    x = np.linspace(x_begin, x_end, nx)
    y = np.linspace(y_begin, y_end, ny)
    t = np.linspace(t_begin, t_end, nt)

    a = 1

    # u = alternating_direction_method(a, x_begin, x_end, y_begin, y_end, t_begin, t_end, phi1, phi2, phi3, phi4, psi, f, h1, h2, tau)
    u = fractional_step_method(a, x_begin, x_end, y_begin, y_end, t_begin, t_end, phi1, phi2, phi3, phi4, psi, f, h1, h2, tau)

    n1 = 100
    n2 = 200
    n = 100

    print(f'{x.shape=}')
    print(f'{y.shape=}')

    xgrid, ygrid = np.meshgrid(x, y)

    print(f'{xgrid.shape=}')
    print(f'{ygrid.shape=}')

    u_ref_vals_x = u_ref(x, y[n2], t[n])
    u_ref_vals_y = u_ref(x[n1], y, t[n])
    
    fig, ax = plt.subplots()
    ax.plot(x, u[n, :, n2], 'g', label='result')
    ax.plot(x, u_ref_vals_x, 'r', label='reference')
    ax.legend()

    ax.set(xlabel='x', ylabel='u',
        title='u(x, y_fix, t_fix)')
    ax.grid()

    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(y, u[n, n1], 'g', label='result')
    ax.plot(y, u_ref_vals_y, 'r', label='reference')
    ax.legend()

    ax.set(xlabel='y', ylabel='u',
        title='u(x_fix, y, t_fix)')
    ax.grid()

    plt.show()

    # fig, ax = plt.subplots()
    # im = ax.imshow(u[n])
    # fig.colorbar(im)
    # plt.show()

    # fig, ax = plt.subplots()
    # im = ax.imshow(u_ref(xgrid, ygrid, t_begin + tau * n))
    # fig.colorbar(im)
    # plt.show()

    # fig, ax = plt.subplots()
    # im = ax.imshow(np.abs(u_ref(xgrid, ygrid, t_begin + tau * n).T - u[n]))
    # fig.colorbar(im)
    # plt.show()

    # ax.set(xlabel='x', ylabel='u',
    #     title='title')
    # ax.grid()

    max_err = np.abs(u[n] - u_ref(xgrid, ygrid, t_begin + tau * n).T).max()
    print(f'{max_err=}')
    # max_err_ = np.abs(u_[n] - u_ref(x, t[n])).max()
    # print(f'{max_err_=}')

    plt.show()