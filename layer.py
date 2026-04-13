import numpy as np
class Layer:
    ### 前向传播
    def forward(self,x):
        raise NotImplementedError
    
    ### 反向传播函数
    def backward(self,grad):
        raise NotImplementedError
    
    ### 参数更新
    def update(self, learning_rate):
        raise NotImplementedError
    
