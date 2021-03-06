import random
import sys

file = open(sys.argv[2], 'w')

n = int(sys.argv[1])
print(n)

for i in range(n):
    val = random.randrange(-100, 100)
    print(val, end=' ')
    file.write("{:10.10e} ".format(abs(val)))

file.close()
