import sys

n = int(input())
power = input().split()
power = [int(i) for i in power]
k, d = map(int, input().split())

fmax = [[0 for i in range(k + 1)] for i in range(n)]
fmin = [[0 for i in range(k + 1)] for i in range(n)]

for i in range(n):
    fmax[i][1] = power[i]
    fmin[i][1] = power[i]

for i in range(n):
    for j in range(2, k + 1):
        for m in range(max(0, i - d), i):
            fmax[i][j] = max(fmax[i][j], fmax[m][j - 1] * power[i], fmin[m][j - 1] * power[i])
            fmin[i][j] = min(fmin[i][j], fmax[m][j - 1] * power[i], fmin[m][j - 1] * power[i])




print(max(fmax[:][k]))
