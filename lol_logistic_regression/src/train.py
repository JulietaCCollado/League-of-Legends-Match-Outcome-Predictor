import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import accuracy  

def train_model(model, X_train, y_train, X_test, y_test, lr=0.01, weight_decay=0.0, epochs=1000):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        outputs = model(X_train).view(-1)  

        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train).view(-1)
        test_outputs = model(X_test).view(-1)

        y_train_flat = y_train.view(-1).float()
        y_test_flat = y_test.view(-1).float()

        train_acc = accuracy(model(X_train), y_train)
        test_acc = accuracy(model(X_test), y_test)

    print(f"Debug: train_acc={train_acc}, test_acc={test_acc}")
    print(train_outputs.shape, y_train_flat.shape)
    print(test_outputs.shape, y_test_flat.shape)
   




    return train_acc, test_acc
