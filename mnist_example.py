from tezromach.layers import Normalization, Linear, Tanh, Sigmoid
from tezromach.network import NeuralNet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

encoder = preprocessing.OneHotEncoder(sparse=False)

inputs = np.genfromtxt("mnist_data/inputs.csv", delimiter=',')
targets = np.genfromtxt("mnist_data/targets.csv", delimiter=',')
targets = encoder.fit_transform(np.array(targets).reshape(-1, 1))

inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.15)
print("inputs shape =", inputs.shape)
print("targets shape =", targets.shape)

net = NeuralNet([
    Normalization(),

    Linear(input_size=64, output_size=100),
    Sigmoid(),
    Linear(input_size=100, output_size=80),
    Sigmoid(),
    Linear(input_size=80, output_size=10),
    Sigmoid()
])

print("start fit NeuralNet")
net.fit(inputs_train, targets_train, verbose_print=10, num_epochs=200, learning_rate=0.1)
print("done fit NeuralNet")
print(net)

net.save_model("model_mnist")

predicted = net.predict(inputs_train)
predicted = np.argmax(predicted, axis=1)
predicted_targets_train = np.argmax(targets_train, axis=1)

print("accuracy metrics (train): ", metrics.accuracy_score(predicted, predicted_targets_train))

predicted = net.predict(inputs_test)
predicted = np.argmax(predicted, axis=1)
predicted_targets_test = np.argmax(targets_test, axis=1)

print("accuracy metrics (test): ", metrics.accuracy_score(predicted, predicted_targets_test))
