# nanograd - Pure Python Autograd Implementation

This project implements an autograd system and a simple neural network framework using nothing but plain Python (plus a visualization library for graph rendering). Made mainly to understand how autograd systems work. This does not support tensors on scalara nodes. Inspiration taken from Andrej Karpathy's tiny grad.

## Features

- Automatic differentiation (autograd) system
- Scalar operations with gradient tracking
- Basic neural network components (Neuron, Layer, MLP)
- Support for common activation functions (ReLU, tanh)

## Autograd Implementation

The core of this project is the `Scalar` class, which implements automatic differentiation:

```python
import math as Math

class Scalar:
    def __init__(self, data, grad = 0, op ='', deps = (), label = ''):
        self.data = data
        self.grad = grad
        self._op = op
        self._deps = set(deps)
        self.label = label
        self._backprop = lambda: None
    
    # ... (methods for addition, multiplication, power, exp, log, etc.)

    def backprop(self):
        # ... (implementation of backpropagation)
```

The `Scalar` class supports various operations like addition, multiplication, power, exponential, logarithm, and activation functions. It also implements the backpropagation algorithm for gradient computation.

## Neural Network Components

Built on top of the `Scalar` class, the project includes basic neural network components:

```python
import random

class Neuron:
    def __init__(self, nin):
        self.weights = [Scalar(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Scalar(0)
    
    def __call__(self, x):
        out = sum((w * x for w, x in zip(self.weights, x)), self.bias)
        return out.tanh()
    
    def params(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, nin, n):
        self.neurons = [Neuron(nin) for _ in range(n)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def params(self):
        return [p for n in self.neurons for p in n.params()]

class MLP:
    def __init__(self, nin, nouts):
        structure = [nin] + nouts
        self.layers = [Layer(structure[i], structure[i + 1]) for i in range(0, len(structure) - 1)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def params(self):
        return [p for layer in self.layers for p in layer.params()]
```

These components allow you to create and train simple neural networks.

## Usage Example

Here's an example of how to use the autograd system:

```python
a = Scalar(3); a.label = 'a'
b = Scalar(2, label = 'b')
c = a + b; c.label = 'c'
d = c / 2; d.label = 'd'
e = d.tanh(); e.label = 'e'
f = 20 * e; f.label = 'f'
g = f.relu(); g.label = 'g'
h = g.ln(); h.label = 'h'
i = h.exp(); i.label = 'i'
j = i ** 2; j.label = 'j'
k = Scalar(2, label = 'k')
l = j - k; l.label = 'l'
m = l.log10(); m.label = 'm'
m.backprop()

# Visualize the computation graph
dot = draw_dot(m)
dot
```

And here's an example of training a simple neural network:

```python
# Create a multi-layer perceptron
myNet = MLP(3, [4, 4, 1])

# Prepare training data
x1 = [[1, 2, 3], [6, 3, 5], [21, 11, 13], [5, 3, 7], [12, 2, 1], [3 , 3, 1]]
y1 = [1, -1, -1, 1, 1, -1]
lr = 0.01

# Training loop
for i in range(1000):
    loss = 0
    for n in myNet.params():
        n.grad = 0

    for x, y in zip(x1, y1):
        y_pred = myNet(x)
        loss += (y_pred - y) ** 2
    
    loss.backprop()    
    
    for n in myNet.params():
        n.data = n.data - lr * n.grad

# Test the trained network
for x, y in zip(x1, y1):
    y_pred = myNet(x)
    print(y_pred.data, y)

# Visualize the computation graph
dot = draw_dot(loss)
dot
```

## Visualization

The project uses an external visualization library graphviz to render computation graphs. The `draw_dot()` function is used to create these visualizations.

## Requirements

- Python 3.x
- graphviz - for visualization
