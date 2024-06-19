import numpy as np

X = [[1,2,3,2.5],
     [1.2,2.4,3.5,2.5],
     [0.5,0.2,3.1,2],
     ]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self, n_inputs):
        self.output = np.dot(n_inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)