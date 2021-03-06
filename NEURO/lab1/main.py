import sys
import random

import math as m
import numpy as np

from perceptron import Perceptron

def random_near(coord, max_radius):
    r = random.random() * max_radius
    phi = random.random() * (2 * m.pi)

    x = r * m.cos(phi)
    y = r * m.sin(phi)

    return x, y

import matplotlib.pyplot as plt

def dist(p1, p2):
    return m.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def main():
    x = [(-2.8, 1.4), (-0.2, -3.5), (2.8, -4), (-2.1, -2.7), (0.3, -4.1), (-1, -4)]
    y = [0, 1, 1, 0, 1, 0]

    class_a_x = []; class_a_y = []
    class_b_x = []; class_b_y = []

    for i, p in enumerate(x):
        if y[i] == 0:
            class_a_x.append(p[0])
            class_a_y.append(p[1])
        else:
            class_b_x.append(p[0])
            class_b_y.append(p[1])

    print(class_a_x)
    print(class_a_y)

    colors_a = (1,0,0)
    colors_b = (0,0,1)

    area = m.pi*3

    plt.scatter(class_a_x, class_a_y, s=area, color=colors_a, alpha=0.5)
    plt.scatter(class_b_x, class_b_y, s=area, color=colors_b, alpha=0.5)

    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')

    classifier = Perceptron()
    classifier.fit(np.array(x), np.array(y), n_epochs=200)

    for p, ref in zip(x, y):
        if(classifier.predict(p) != ref):
            print(f'loser! WRONG CLASSIFICATION: {p}')
            break

    weights = classifier.get_weights()
    k = -weights[0] / weights[1]
    b = -weights[2] / weights[1]

    print(f'{k=}')
    print(f'{b=}')

    f = lambda x_: k * x_ + b

    x_graph = np.linspace(-5, 5, 2)
    y_graph = np.array(list(map(f, x_graph)))
    plt.plot(x_graph, y_graph, 'g-', linewidth=2, markersize=12)
    plt.show()

if __name__ == "__main__":
    main()