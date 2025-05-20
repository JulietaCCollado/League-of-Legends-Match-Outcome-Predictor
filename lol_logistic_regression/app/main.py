import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import pandas as pd
from src.model import LogisticRegressionModel

st.title("League of Legends Match Outcome Predictor")

uploaded_file = st.file_uploader("Upload a CSV file with match statistics", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Define the exact features the model was trained on
    selected_features = [
        "kills", "deaths", "assists",
        "gold_earned", "cs", "wards_placed",
        "wards_killed", "damage_dealt"
    ]

    # Check if any expected features are missing
    missing = [col for col in selected_features if col not in df.columns]
    if missing:
        st.error(f"Missing required columns in CSV: {missing}")
    else:
        
        inputs = torch.tensor(df[selected_features].values, dtype=torch.float32)

        
        model = LogisticRegressionModel(len(selected_features))
        model.load_state_dict(torch.load("models/logistic_model.pth"))
        model.eval()

        
        with torch.no_grad():
            probs = model(inputs).squeeze().numpy()
            preds = (probs >= 0.5).astype(int)

       
        df["Predicted Outcome"] = preds
        st.success("Prediction completed!")
        st.write(df)
