### Simple neural network for Python 3.6
Vectorized Python implementation of backpropagation with NumPy.
Use layers with different *Activations*:
 * Sigmoid
 * Hyperbolic Tangent (Tanh)
 
 You can add preprocessing layers into your pipeline.
 Preprocessing:
  * Normalization, which is
![normalization](files/CodeCogsEqn.gif?raw=true "Title")
  * MinMaxScaling
  
Loss function is MSE (Mean squared error), You can easily add you own loss. Just inheret it from `LossFunction` class
```python
from tezromach.network import NeuralNet
from tezromach.layers import Linear, Sigmoid, Tanh, Normalization

net = NeuralNet([
    Normalization(), # Preprocessing

    Linear(input_size=2, output_size=4),
    Tanh(),
    Linear(input_size=4, output_size=1),
    Sigmoid()
])

net.fit(inputs, targets, verbose_print=10, num_epochs=200, learning_rate=0.1, epsilon=0.002)

predictions = net.predict(inputs_for_test)
```
