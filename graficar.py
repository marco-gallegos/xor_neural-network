import matplotlib as plt
from xor.archivos import *
from xor.xor import *

hidden_weights, hidden_bias, output_weights, output_bias = load_training_data()

inputs = np.array([[0, 0]])

predicted_output, hidden_layer_output, output_layer_activation, hidden_layer_activation = \
    executeNetwork(inputs=inputs, hidden_bias=hidden_bias, hidden_weights=hidden_weights, output_bias=output_bias, output_weights=output_weights)

print(f"clasificado en {predicted_output}")