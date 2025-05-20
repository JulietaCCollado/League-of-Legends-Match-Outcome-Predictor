import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv("data\\league_of_legends_data_large.csv")
    X = df.drop("win", axis=1)
    y = df["win"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32),
            torch.tensor(y_test.values, dtype=torch.float32),
            X.columns.tolist())


