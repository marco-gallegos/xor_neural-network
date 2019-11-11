import numpy as np

line_length = 30


def initial(hidden_weights, hidden_bias, output_weights, output_bias):
    print("Pesos Escondidos Iniciales: ")
    print(*hidden_weights)
    print("Bias Bscondidos Iniciales: ")
    print(*hidden_bias)
    print("Pesos de salida iniciales: ")
    print(*output_weights)
    print("Initial output biases: ")
    print(*output_bias)

    print("-" * line_length)

    print("procesando ...")


def progress(predicted_output, error, iteracion):
    print(f"Iteracion {iteracion}\n", "salidas: \n", predicted_output, "\n")
    print(f"error:\n {error}")
    print("+" * line_length)


def final(predicted_output, error, epochs, hidden_weights, hidden_bias, output_weights, output_bias):
    print("-" * line_length, "\n\n")
    print(f"Final", "salidas: \n", predicted_output, "\n")
    print(f"error: {error}")

    print("-" * line_length, "\n\n")

    print("Pesos De Salida Final: ")
    print(*hidden_weights)
    print("Bias Escondido Final: ")
    print(*hidden_bias)
    print("Pesos De Salida Final: ")
    print(*output_weights)
    print("Bias De Salida Final: ")
    print(*output_bias)

    print(f"\nSalida de la red neuronal despues de {epochs} iteraciones: ")
    print(*predicted_output)
