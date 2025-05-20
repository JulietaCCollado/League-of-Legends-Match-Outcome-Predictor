from src.data_loader import load_data
from src.model import LogisticRegressionModel
from src.train import train_model
import torch
import os

# Load the data
X_train, X_test, y_train, y_test, feature_names = load_data("data/league_of_legends_data_large.csv")

# Initialize the model
input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)

# Train the model
train_acc, test_acc = train_model(model, X_train, y_train, X_test, y_test)
print(f"Train Accuracy: {train_acc * 100:.2f}%, Test Accuracy: {test_acc * 100:.2f}%")


os.makedirs("models", exist_ok=True)  # Ensure models/ directory exists
torch.save(model.state_dict(), "models/logistic_model.pth")
