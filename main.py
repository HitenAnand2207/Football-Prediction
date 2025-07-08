# main.py

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from utils.features import create_features
from model.predictor import train_model

# Load CSV
df = pd.read_csv("data/matches.csv")

# Parse dates correctly
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Drop rows with missing results
df = df.dropna(subset=["FTHG", "FTAG", "FTR"])

# Encode teams
le = LabelEncoder()
df["HomeTeam_enc"] = le.fit_transform(df["HomeTeam"])
df["AwayTeam_enc"] = le.transform(df["AwayTeam"])

# Encode match result
df["FTR_encoded"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})

# Feature engineering
df, _ = create_features(df)  # Fix here: discard second item from tuple

# Train model
model = train_model(df)

# Save model and encoder
joblib.dump(model, "model/match_predictor.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("✅ Training complete.")
