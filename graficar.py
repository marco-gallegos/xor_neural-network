import matplotlib.pyplot as plt
from xor.archivos import *
from xor.xor import *

def es_viable(valor):
    tolerancia = 0.01
    valor_clasificacion = 1 - tolerancia
    if valor >= valor_clasificacion:
        return True
    return False


hidden_weights, hidden_bias, output_weights, output_bias = load_training_data()

inputs = np.array([[1, 0]])

rango = 1

verdaderos = np.array([])
falsos = np.array([])

for i in np.arange(0, rango, 0.001):
    for j in np.arange(0, rango, 0.001):
        predicted_output, hidden_layer_output, output_layer_activation, hidden_layer_activation = \
            executeNetwork(inputs=np.array([[i,j]]), hidden_bias=hidden_bias, hidden_weights=hidden_weights, output_bias=output_bias,
                           output_weights=output_weights)
        predicted_output = predicted_output[0][0]
        clasificacion = es_viable(predicted_output)
        if clasificacion:
            verdaderos = np.append(verdaderos, [i,j])
            plt.plot(i, j, color='green', marker='o')
        else:
            falsos = np.append(falsos, [i,j])
            plt.plot(i, j, color='red', marker='*')

print(len(verdaderos), len(falsos))
plt.show()