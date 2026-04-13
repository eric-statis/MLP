### 数据导入
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname='./xor_dataset.csv', delimiter= ',')
# print(data)

### 划分数据
index = np.random.permutation(range(len(data)))
data = data[index]
# print(data)
ratio = 0.8
split = int(ratio * len(data))
x_train, y_train = data[:split,:2], data[:split,2]
x_test, y_test = data[split:, :2], data[split:,2]