
import numpy as np
from rbf import rbf

from utils import *

import matplotlib.pyplot as plt

def rotate(point, alpha):
    c = np.cos(alpha)
    s = np.sin(alpha)

    rotation_matrix = np.array([[c, s], [-s, c]])
    return rotation_matrix @ point

def ellipse(a, b, alpha, x0, y0, n):
    t = np.linspace(0, 2*np.pi, n)

    c = np.cos(alpha)
    s = np.sin(alpha)

    points_x =  a * np.cos(t)
    points_y = b * np.sin(t)
    
    points_x_rotated = points_x * c + points_y * -s
    points_y_rotated = points_x * s + points_y * c

    points_x_rotated_shifted = points_x_rotated + x0
    points_y_rotated_shifted = points_y_rotated + y0

    return points_x_rotated_shifted, points_y_rotated_shifted

class1_n_points = 60
class2_n_points = 100
class3_n_points = 120

h = 0.025
n_elipse_points = int(2 * np.pi / h) + 1

# вариант 8
class1_points_x, class1_points_y = ellipse(0.2, 0.2, 0, -0.25, 0.25, n_elipse_points)
class2_points_x, class2_points_y = ellipse(0.7, 0.5, -np.pi/3, 0, 0, n_elipse_points)
class3_points_x, class3_points_y = ellipse(1, 1, 0, 0, 0, n_elipse_points)

class1_select = np.random.choice(n_elipse_points, class1_n_points, replace=False)
class2_select = np.random.choice(n_elipse_points, class2_n_points, replace=False)
class3_select = np.random.choice(n_elipse_points, class3_n_points, replace=False)

fig, ax = plt.subplots()

ax.scatter(class1_points_x[class1_select], class1_points_y[class1_select], c='red', s=2)
ax.scatter(class2_points_x[class2_select], class2_points_y[class2_select], c='green', s=2)
ax.scatter(class3_points_x[class3_select], class3_points_y[class3_select], c='blue', s=2)
ax.grid()

class1_X = np.column_stack([np.take(class1_points_x, class1_select), np.take(class1_points_y, class1_select)])
class2_X = np.column_stack([np.take(class2_points_x, class2_select), np.take(class2_points_y, class2_select)])
class3_X = np.column_stack([np.take(class3_points_x, class3_select), np.take(class3_points_y, class3_select)])

class1_Y = np.tile(np.array([1, 0, 0]), (class1_n_points, 1))
class2_Y = np.tile(np.array([0, 1, 0]), (class2_n_points, 1))
class3_Y = np.tile(np.array([0, 0, 1]), (class3_n_points, 1))

n_samples = class1_n_points + class2_n_points + class3_n_points
permutation = np.random.permutation(n_samples)

X = np.take(np.vstack([class1_X, class2_X, class3_X]), permutation, axis=0)
Y = np.take(np.vstack([class1_Y, class2_Y, class3_Y]), permutation, axis=0)

x_train, y_train, x_test, y_test, x_valid, y_valid = split_dataset(X, Y, 0.2, 0)

classifier = rbf()
spread = 0.3
classifier.fit(x_train, y_train, spread=spread)

y_pred = np.where(classifier.predict(x_train) > 0.5, 1, 0)
correctly_classified = row_wise_equal(y_pred, y_train)
print(f'{correctly_classified}/{x_train.shape[0]} ({correctly_classified / x_train.shape[0]}) points of train set are classified correctly!')

y_pred = np.where(classifier.predict(x_test) > 0.5, 1, 0)
correctly_classified = row_wise_equal(y_pred, y_test)
print(f'{correctly_classified}/{x_test.shape[0]} ({correctly_classified / x_test.shape[0]}) points of test set are classified correctly!')

x = np.arange(-1.2, 1.2, h)
y = np.arange(-1.2, 1.2, h)

x_, y_ = np.meshgrid(x, y)

points = np.vstack([x_.flatten(), y_.flatten()]).T

pred = classifier.predict(points)

class_labels = np.argmax(pred, axis=1)

class1_grid_points = points[class_labels == 0]
class2_grid_points = points[class_labels == 1]
class3_grid_points = points[class_labels == 2]

x_class1_grid = class1_grid_points[:,0]
y_class1_grid = class1_grid_points[:,1]

x_class2_grid = class2_grid_points[:,0]
y_class2_grid = class2_grid_points[:,1]

x_class3_grid = class3_grid_points[:,0]
y_class3_grid = class3_grid_points[:,1]

ax.scatter(x_class1_grid, y_class1_grid, color=(1, 0, 0, 0.3), s=3)
ax.scatter(x_class2_grid, y_class2_grid, color=(0, 1, 0, 0.3), s=3)
ax.scatter(x_class3_grid, y_class3_grid, color=(0, 0, 1, 0.3), s=3)

ax.grid()

ax.set_title('Классификация точек плоскости')
plt.savefig('3.png')
plt.show()