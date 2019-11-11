import numpy as np


def load_training_data():
    try:
        hidden_weights = np.load("hidden_weights.npy")
        hidden_bias = np.load("hidden_bias.npy")
        output_weights = np.load("output_weights.npy")
        output_bias = np.load("output_bias.npy")
        return hidden_weights, hidden_bias, output_weights, output_bias
    except:
        return False, False, False, False


def initialize_training_data(inputLayerNeurons:int, hiddenLayerNeurons:int, outputLayerNeurons:int):
    try:
        hidden_weights = np.load("hidden_weights.npy")
    except:
        hidden_weights = np.random.uniform(low=0, high=1, size=(inputLayerNeurons, hiddenLayerNeurons))

    try:
        hidden_bias = np.load("hidden_bias.npy")
    except:
        hidden_bias = np.random.uniform(low=0, high=1, size=(1, hiddenLayerNeurons))

    try:
        output_weights = np.load("output_weights.npy")
    except:
        output_weights = np.random.uniform(low=0, high=1, size=(hiddenLayerNeurons, outputLayerNeurons))

    try:
        output_bias = np.load("output_bias.npy")
    except:
        output_bias = np.random.uniform(low=0, high=1, size=(1, outputLayerNeurons))

    return hidden_weights, hidden_bias, output_weights, output_bias


def savetrainingdata(hidden_weights, hidden_bias, output_weights, output_bias):
    try:
        np.save("hidden_weights",hidden_weights)
        np.save("hidden_bias",hidden_bias)
        np.save("output_weights", output_weights)
        np.save("output_bias", output_bias)
        return True
    except:
        return False
