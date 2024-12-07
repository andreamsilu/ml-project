from src.preprocessing import load_data, preprocess_data
from src.models import MLP
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # Load and preprocess data
    train_data, train_labels = load_data("data/dataSet1.csv")
    test_data, test_labels = load_data("data/dataSet2.csv")
    train_data, test_data = preprocess_data(train_data, test_data)

    # Initialize the model
    model = MLP(input_size=64, hidden_size=32, output_size=10)

    # Train the model
    train_model(model, train_data, train_labels, learning_rate=0.01, epochs=100)

    # Evaluate the model
    accuracy, confusion_matrix = evaluate_model(model, test_data, test_labels)

    # Save and print results
    print(f"Model Accuracy: {accuracy:.2f}")
    with open("reports/results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Confusion Matrix:\n{confusion_matrix}")

if __name__ == "__main__":
    main()
