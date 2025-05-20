from src.model import LogisticRegressionModel
from src.train import train_model

def tune_learning_rates(X_train, y_train, X_test, y_test, input_dim):
    lrs = [0.01, 0.05, 0.1]
    best_acc = 0
    best_lr = None

    for lr in lrs:
        print(f"\nTraining with learning rate: {lr}")
        model = LogisticRegressionModel(input_dim)
        _, test_acc = train_model(model, X_train, y_train, X_test, y_test,lr=lr, epochs=100)
        if test_acc > best_acc:
            best_acc = test_acc
            best_lr = lr

    print(f"\nBest learning rate: {best_lr} with test accuracy: {best_acc:.4f}")
    