import ICIP_Compression
import numpy as np

arr = np.zeros((10,10))
val = 0

for i in range(10):
        for j in range(10):
                val = val + 0.1
                arr[i,j] = val

arr2 = ICIP_Compression.arraytest(arr)

print(arr2)