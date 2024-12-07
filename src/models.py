import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.WH = np.random.rand(hidden_size, input_size) * 0.01
        self.WO = np.random.rand(output_size, hidden_size) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        # Hidden layer
        self.hidden_inputs = np.dot(self.WH, inputs)
        self.hidden_outputs = self.sigmoid(self.hidden_inputs)

        # Output layer
        self.final_inputs = np.dot(self.WO, self.hidden_outputs)
        self.final_outputs = self.sigmoid(self.final_inputs)
        return self.final_outputs
