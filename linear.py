import numpy as np
from layer import Layer
class Linear(Layer):
    def __init__(self, num_in, num_out, use_bias = True):
        self.num_in = num_in
        self.num_out = num_out
        self.use_bias = use_bias

        ### 参数初始化
        self.W = np.random.normal(loc=0, scale=1, size=(self.num_in, self.num_out))
        # self.W = np.random.normal(loc=0, scale=1, size=(self.num_out, self.num_in))
        if self.use_bias:
            self.b = np.zeros(shape=(1, self.num_out))

    def forward(self, x):
        self.x = x
        self.y = self.x @ self.W
        if self.use_bias:
            self.y = self.x @ self.W + self.b
        return self.y
    
    def backward(self, grad):
        return super().backward(grad)

    def update(self, learning_rate):
        return super().update(learning_rate)
    
if __name__ == '__main__':
    data = np.random.normal(size=(10,5))
    lin = Linear(5,10)
    y = lin.forward(data)
    print(y.shape)