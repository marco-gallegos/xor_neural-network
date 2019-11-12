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
avance = 0.01


def calcular():
    verdaderos = []
    falsos = []
    for i in np.arange(0, rango, avance):
        print(f"valor en i {i}")
        for j in np.arange(0, rango, avance):
            predicted_output, hidden_layer_output, output_layer_activation, hidden_layer_activation = \
                executeNetwork(inputs=np.array([[i,j]]), hidden_bias=hidden_bias, hidden_weights=hidden_weights, output_bias=output_bias,
                               output_weights=output_weights)
            predicted_output = predicted_output[0][0]
            clasificacion = es_viable(predicted_output)
            if clasificacion:
                verdaderos.append(np.array([i,j]))
                plt.plot(i, j, color='green', marker='o')
            else:
                falsos.append(np.array([i,j]))
                plt.plot(i, j, color='red', marker='*')
    verdaderos = np.array(verdaderos)
    falsos = np.array(falsos)
    np.save(file="verdaderos", arr=verdaderos)
    np.save(file="falsos", arr=falsos)
    print(falsos)
    plt.show()

def mostrar():
    verdaderos = np.load("verdaderos.npy")
    falsos = np.load("falsos.npy")

    for punto in falsos:
        plt.plot(punto[0], punto[1], color='red', marker='*')
    for punto in verdaderos:
        plt.plot(punto[0], punto[1], color='green', marker='o')

    plt.show()

if __name__ == "__main__":
    opc = int(input("1 calcular, 2 mostrar : "))

    if opc == 1:
        calcular()
    elif opc == 2:
        mostrar()