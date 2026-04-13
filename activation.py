import numpy as np
from layer import Layer

class Identity(Layer):
    def forward(self, x):
        return x
    
    def backward(self, grad):
        return grad 
    
class Sigmoid(Layer):
    def forward(self, x):
        self.y = 1 / (1 + np.exp( - x))
        return self.y
    
    def backward(self, grad):
        return grad * self.y * (1 - self.y)
    
class Tanh(Layer):
    def forward(self, x):
        self.y = np.tanh(x)
        return self.y
    
    def backward(self, grad):
        return grad * (1 - self.y**2)
    
# class ReLU(Layer):
#     def forward(self, x):
#         self.x = x
#         self.y = x if x > 0 else 0
#         return self.y
    
#     def backward(self, grad):
#         return grad if self.x > 0 else 0

class ReLU(Layer):
    def forward(self, x):
        self.x = x
        self.y = x * (x > 0)
        return self.y
    
    def backward(self, grad):
        return grad * (self.x > 0)
