import sys

start = 183004
n = 650
output = sys.argv[1]

print(list(filter(bool, '100 1 00  00 1110  1'.split(' '))))

count = 0
with open(output, 'w') as out:
    with open('data_full.txt', 'r') as file:
        for line in file:
            l = list(filter(bool, line.split(' ')))
            print(l)
            year = int(l[0])
            month = int(l[1]) 

            number = int(l[0]) * 100 + int(l[1])

            if number >= start:
                out.write(f'{l[4]}\n')
                count += 1

            if count == n:
                break
