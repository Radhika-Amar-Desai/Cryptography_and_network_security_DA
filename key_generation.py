import json
import random
import numpy as np


def generate_matrix_with_target_determinant(dim, target_dets=[1, 2, 5]):
    """Generate a random square matrix with a determinant that matches one of the target values or is a multiple of 10."""
    while True:
        A = np.random.randint(0, 1, (dim, dim))  # Matrix with integer entries between 1 and 9
        det = int(np.linalg.det(A))  # Compute determinant
        if det in target_dets:
            return A.tolist()  # Return the matrix as a list for JSON serialization


def generate_layer(dim=16, target_dets=[1, 2, 5]):
    """Generate a layer with weights and random binary biases."""
    weights = generate_matrix_with_target_determinant(dim, target_dets)
    biases = [[random.randint(0, 1)] for _ in range(dim)]  # Random binary biases
    return {"weights": weights, "biases": biases}


def generate_model_json(layers_count, dim=16, target_dets=[1, 2, 5]):
    """Generate a JSON object with the specified number of layers."""
    model = {"layers": [generate_layer(dim, target_dets) for _ in range(layers_count)]}
    return model


def write_to_json_file(data, filename="model.json"):
    """Write the JSON object to a file."""
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=2)
    print(f"Model saved to {filename}")


if __name__ == "__main__":
    # Get the number of layers from the user
    try:
        layers_count = int(input("Enter the number of layers: "))
        if layers_count <= 0:
            print("Number of layers should be greater than 0.")
        else:
            # Generate the model and write to a file
            model_json = generate_model_json(layers_count)
            write_to_json_file(model_json)
    except ValueError:
        print("Please enter a valid integer for the number of layers.")
