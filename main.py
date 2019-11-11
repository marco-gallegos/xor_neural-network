import numpy as np
import prints.help as pr
from xor.xor import *
# np.random.seed(0)

# Input datasets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

epochs = 400000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Random weights and bias initialization
hidden_weights = np.random.uniform(low=0, high=1, size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(low=0, high=1, size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(low=0, high=1, size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(low=0, high=1, size=(1, outputLayerNeurons))

# pr.initial(hidden_weights, hidden_bias, output_weights, output_bias)

# Training algorithm
for _ in range(epochs):
    # Determine print progress
    printProgress = ((_%1000) == 0)

    # Forward Propagation
    predicted_output, hidden_layer_output, output_layer_activation, hidden_layer_activation = \
        executeNetwork(inputs=inputs, hidden_bias=hidden_bias, hidden_weights=hidden_weights,
                       output_bias=output_bias, output_weights=output_weights)

    # Backpropagation
    error = expected_output - predicted_output
    print(error)
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    print(d_predicted_output)
    exit()

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    if printProgress:
        pr.progress(predicted_output, error, _)

pr.final(predicted_output, error, epochs, hidden_weights, hidden_bias, output_weights, output_bias)
