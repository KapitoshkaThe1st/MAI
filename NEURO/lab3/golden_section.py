_phi = 1.6180339887498948482
_gss_eps = 1e-5

def gss(left_bound, right_bound, f):
    a = left_bound
    b = right_bound

    while abs(a - b) > _gss_eps:
        x1 = b - (b-a) / _phi
        x2 = a + (b-a) / _phi

        if f(x1) >= f(x2):
            a = x1
        else:
            b = x2
    
    return (a + b) / 2

import math as m

def f(x):
    return (x - 1.23456) ** 2 + 1 

if __name__ == '__main__':
    print(gss(-2, 2, f))
    print(gss(-3 / 2 * m.pi,  m.pi / 2, m.sin))

