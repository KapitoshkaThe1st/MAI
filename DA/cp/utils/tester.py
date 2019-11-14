import sys

inp = open(sys.argv[1])
ref = open(sys.argv[2])

totalMatchCount = 0
totalExcessCount = 0 
totalLackCount = 0 

for line in inp:
    li = line.strip().split(',')
    si = set(li)

    lineRef = ref.readline()
    lr = lineRef.strip().split(',')
    sr = set(lr)

    matchCount = len(si.intersection(sr))
    excessCount = len(si.difference(sr))
    lackCount = len(sr.difference(si))

    totalMatchCount += matchCount
    totalExcessCount += excessCount
    totalLackCount += lackCount

print("Match count: ")
print(totalMatchCount)
print("Excess count: ")
print(totalExcessCount)
print("Lack count: ")
print(totalLackCount)

