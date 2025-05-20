# League of Legends Match Outcome Predictor

This project is a logistic regression-based machine learning model that predicts the outcome of League of Legends matches based on in-game statistics. Built with **PyTorch** and deployed vis **Streamlit**, it allows users to upload match data and instantly get predictions.

---
## Model Details

-**Algorithm:** Logistic Regression
-**Framework:** PyTorch
-**Input Features (8):**
 - `kills`  
  - `deaths`  
  - `assists`  
  - `gold_earned`  
  - `cs` (creep score)  
  - `wards_placed`  
  - `wards_killed`  
  - `damage_dealt`

-**Output:** Binary prediction of match outcome (`0 = Loss`, `1 = Win`)

---

## How to Run the app

### 1. Clone the repository

```bash
git clone https://github.com/JulietaCCollado/League-of-Legends-Match-Outcome-Predictor.git
cd lol_logistic_regression
```
### 2. Install dependencies
```
pip install -r requirements.txt

```
### 3. Launch the Streamlit app

```
streamlit run app/main.py

```
### 4. Upload the CSV file
 [Download sample CSV file](league_of_legends_data_large.csv) 
that contains the 8 required features and upload it. The app will return predictions for each match.

## Model Training (Optional)

Training was done in `notebooks/model_training.ipynb` using a dataset of historical LoL matches. The trained model is saved to `models/logistic_model.pth`

### Author
` Julieta Collado` - AI Enginerring Student-<br>
This project is part of a final project for a Machine Learning Course. <br>Thanks for trying it out ♥ ♥ ♥ <br>
05/2025
