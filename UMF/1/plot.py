from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from cycler import cycler

def input_list(n):
    return [float(input()) for _ in range(n)]

n = int(input())

xvals = input_list(n)
print(xvals)

k = int(input())

colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
          for i in range(1000)]
plt.rc('axes', prop_cycle=(cycler('color', colors)))

plt.title("Зависимость температуры u от координаты x при различных t")
for i in range(k):
    t = float(input())

    yvals = input_list(n)
    print(yvals, end='')
    plt.plot(xvals, yvals, label=f"t={t}")

plt.legend()
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.tight_layout()
plt.show()
