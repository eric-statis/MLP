from activation import Identity, Sigmoid, Tanh, ReLU
from linear import Linear
activation_dict = {'identity': Identity,
                   'sigmoid': Sigmoid,
                   'tanh': Tanh,
                   'relu': ReLU}

class MLP:
    def __init__(self, layer_size: list, use_bias = True, activation = 'relu', out_activation = 'identity'):
        num_in = layer_size[0]
        self.layers = []
        for num_out in layer_size[1:-1]:
            self.layers.append(Linear(num_in, num_out, use_bias))
            self.layers.append(activation_dict[activation]())
            num_in = num_out
        ###
        self.layers.append(Linear(num_in, layer_size[-1], use_bias=use_bias))
        self.layers.append(activation_dict[out_activation]())

    def forward(self, x):
        out = x
        for k in self.layers:
            out = k.forward(out)
        return out
    
    def backward(self, grad):
        for k in reversed(self.layers):
            grad = k.backward(grad)
        
    def update(self, learning_rate):
        for k in self.layers:
            k.update(learning_rate)

