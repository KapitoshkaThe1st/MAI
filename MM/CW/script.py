from PIL import Image, ImageChops, ImageEnhance, ImageOps
import numpy as np
import sys

path1 = sys.argv[1]
path2 = sys.argv[2]

out_path = sys.argv[3]

print(path1)
print(path2)
print(out_path)

img1 = Image.open(path1)
img2 = Image.open(path2)

res = ImageChops.difference(img1, img2)
res = ImageOps.invert(res)

enhancer = ImageEnhance.Contrast(res)
res = enhancer.enhance(4.0)

res.save(out_path)

