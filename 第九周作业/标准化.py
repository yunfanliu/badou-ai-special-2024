import numpy as np
import matplotlib.pyplot as plt

# 归一化【0-1】
def Normalization1(x):
    for i in range(len(x)):
        x[i] = (float(x[i]) - min(x)) / float(max(x) - min(x))
    return x


# 归一化【-1，1】
def Normalization2(y):
    for i in range(len(y)):
        y[i] = (y[i] - np.mean(y)) / (max(y) - min(y))
    return y


# 标准化
def z_score(z):
    for i in range(len(z)):
        sd = sum([z[i] - np.mean(z) **2 for i in range(len(z))]) / len(z)
        z[i] = (z[i] - np.mean(z)) / sd
    return z

a=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
a1 = a.copy()
a3 = a1.copy()
N1 = Normalization1(a)
N2 = Normalization2(a1)
Z = z_score(a)
print(N1)
print(N2)
print(Z)
