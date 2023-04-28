import numpy as np

a = np.arange(10).reshape(5, 2)
a = np.random.permutation(np.arange(len(a)))
hvis = np.zeros()
print(a)

for vec in a:
    print(vec)
