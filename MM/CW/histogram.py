from PIL import Image
import matplotlib.pyplot as plt 
import sys
import numpy as np

path = sys.argv[1]
img = Image.open(path)
gray = img.convert('L')

brightness = np.histogram(gray.getdata(), bins=256, range=(0, 256))[0]

r, g, b = img.split()

red = np.histogram(r.getdata(), bins=256, range=(0, 256))[0]
green = np.histogram(g.getdata(), bins=256, range=(0, 256))[0]
blue = np.histogram(b.getdata(), bins=256, range=(0, 256))[0]

fig, axs = plt.subplots(4, sharex=True)
fig.suptitle('Гистограммы')

axs[0].set_title('Красный')
axs[1].set_title('Зеленый')
axs[2].set_title('Синий')
axs[3].set_title('Яркость')

axs[0].set_yticks([])
axs[1].set_yticks([])
axs[2].set_yticks([])
axs[3].set_yticks([])

axs[0].set_xticks([])
axs[1].set_xticks([])
axs[2].set_xticks([])
axs[3].set_xticks([])

for i in range(0, 256):
    axs[0].bar(i, red[i], color = 'r')
for i in range(0, 256):
    axs[1].bar(i, green[i], color = 'g')
for i in range(0, 256):
    axs[2].bar(i, blue[i], color = 'b')
for i in range(0, 256):
    axs[3].bar(i, brightness[i], color = (0.3, 0.3, 0.3))

fig.tight_layout()

fig.set_size_inches(3, 8)
fig.savefig(sys.argv[2], dpi=400)