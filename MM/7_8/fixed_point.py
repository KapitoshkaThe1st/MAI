x_0 = 0.4 + 0.4j
# c = 0.4 + 0.08j
c = 0.1 + 0.2j

def f(x):
    return x*x + c

n = 30
x = x_0

print(f'z_{0}; {x.real:.7f}; {x.imag:.7f}; {abs(x):.7f}')
for i in range(n):
    x = f(x)
    print(f'z_{i}; {x.real:.7f}; {x.imag:.7f}; {abs(x):.7f}')
