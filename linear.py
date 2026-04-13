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
        self.x = x ### (batch_size, num_in)
        self.y = self.x @ self.W
        if self.use_bias:
            self.y = self.x @ self.W + self.b
        return self.y
    
    def backward(self, grad):
        ### 反向传播按照链式法则
        ### grad的维度是(batch_size, num_out)
        ### 梯度要对batch_size求平均值
        ### grad_W的维度和W的维度相同时(num_in, num_out)
        self.grad_W = self.x.T @ grad / grad.shape[0]
        if self.use_bias:
            self.grad_b = np.mean(grad, axis=0, keepdims=True)
        grad = grad @ self.W.T
        return grad ### 传给前一层的梯度



    def update(self, learning_rate):
        self.W -= learning_rate * self.grad_W
        if self.use_bias:
            self.b -= learning_rate * self.grad_b
        ### 静默函数

    
if __name__ == '__main__':
    data = np.random.normal(size=(10,5))
    lin = Linear(5,10)
    y = lin.forward(data)
    grad = lin.backward(np.random.normal(size=(10,10)))
    print(grad)
    print(lin.grad_b)
    print(y.shape)