import numpy as np

def train_model(model, train_data, train_labels, learning_rate=0.01, epochs=100):
    for epoch in range(epochs):
        for inputs, target in zip(train_data, train_labels):
            # One-hot encoding for target
            target_one_hot = np.zeros(model.WO.shape[0])
            target_one_hot[target] = 1

            # Forward pass
            outputs = model.feedforward(inputs)

            # Calculate error
            error_output = target_one_hot - outputs

            # Backpropagation
            error_hidden = model.sigmoid_derivative(model.hidden_outputs) * np.dot(model.WO.T, error_output)

            # Update weights
            model.WO += learning_rate * np.outer(error_output, model.hidden_outputs)
            model.WH += learning_rate * np.outer(error_hidden, inputs)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Training...")

    print("Training Complete!")
