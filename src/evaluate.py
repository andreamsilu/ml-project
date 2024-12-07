import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_model(model, test_data, test_labels):
    predictions = []
    for inputs in test_data:
        outputs = model.feedforward(inputs)
        predictions.append(np.argmax(outputs))  # Get predicted class

    accuracy = accuracy_score(test_labels, predictions)
    conf_matrix = confusion_matrix(test_labels, predictions)
    return accuracy, conf_matrix
