import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL

from hopfield_nn import hopfield_nn

def noise(x, noise_portion):
    size = x.shape[0]
    index = np.random.randint(0, size, int(noise_portion * size))
    res = np.copy(x)
    for i in index:
        res[i] = -1 if x[i] == 1 else 1
    
    return res

digits = [0, 1, 2, 3, 4, 6, 9]

images = []

for d in digits:
    img = PIL.Image.open(str(d) + '.bmp').convert('L')
    images.append(np.where(np.asarray(img) > 128, 1, -1))

n_examples = len(digits)

x_train = []
for i in range(n_examples):
    x_train.append(images[i].flatten())
x_train = np.array(x_train)
print(f'{x_train.shape=}')

filter = hopfield_nn()
filter.fit(x_train)

initial_shape = (12, 10)

images_to_process = [2, 4, 1]
epochs = 600

img1 = x_train[images_to_process[0]][np.newaxis]
plt.imshow(img1.reshape(initial_shape), cmap=plt.get_cmap('Greys_r'))
plt.savefig('noise1.png')
plt.show()

result = filter.predict(img1, epochs=epochs).reshape(initial_shape)
plt.imshow(result, cmap=plt.get_cmap('Greys_r'))
plt.savefig('result1.png')
plt.show()

img2 = noise(x_train[images_to_process[1]], 0.2)[np.newaxis]
plt.imshow(img2.reshape(initial_shape), cmap=plt.get_cmap('Greys_r'))
plt.savefig('noise2.png')
plt.show()

result = filter.predict(img2, epochs=epochs).reshape(initial_shape)
plt.imshow(result, cmap=plt.get_cmap('Greys_r'))
plt.savefig('result2.png')
plt.show()

img3 = noise(x_train[images_to_process[2]], 0.3)[np.newaxis]
plt.imshow(img3.reshape(initial_shape), cmap=plt.get_cmap('Greys_r'))
plt.savefig('noise3.png')
plt.show()

result = filter.predict(img3, epochs=epochs).reshape(initial_shape)
plt.imshow(result, cmap=plt.get_cmap('Greys_r'))
plt.savefig('result3.png')
plt.show()


