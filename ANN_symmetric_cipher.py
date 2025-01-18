import numpy as np
import json

def load_architecture(filename):
    with open(filename, 'r') as f:
        architecture = json.load(f)
    return architecture['layers']

def forward_pass(layers, x):
    activations = [x]
    for layer in layers:
        W = np.array(layer['weights'])
        b = np.array(layer['biases'])
        h = np.dot(W, activations[-1]) + b
        activations.append(h)
    return activations

def reverse_pass(layers, y, y_min, y_max):
    h2_reconstructed = np.dot(np.linalg.pinv(np.array(layers[-1]['weights'])), (y * (y_max - y_min) + y_min - np.array(layers[-1]['biases'])))
    h1_reconstructed = np.dot(np.linalg.pinv(np.array(layers[-2]['weights'])), (h2_reconstructed - np.array(layers[-2]['biases'])))
    x_reconstructed = np.dot(np.linalg.pinv(np.array(layers[0]['weights'])), (h1_reconstructed - np.array(layers[0]['biases'])))
    return (x_reconstructed > 0.5).astype(int)

if __name__ == "__main__":
    layers = load_architecture('nn_architecture.json')

    x = np.array([[0], [1], [1]])

    print("Original message (Input):", x.T)

    activations = forward_pass(layers, x)
    y = activations[-1]

    y_min = np.min(y)
    y_max = np.max(y)
    y_normalized = (y - y_min) / (y_max - y_min)

    print("Encrypted message (Normalized Output):", y_normalized.T)

    x_reconstructed = reverse_pass(layers, y_normalized, y_min, y_max)
    
    print("Decrypted message (Reconstructed):", x_reconstructed.T)
