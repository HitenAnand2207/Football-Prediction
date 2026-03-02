# main.py

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.features import create_features
from model.predictor import train_model
from utils.analytics import evaluate_model_performance, get_feature_importance

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

# Prepare features for model evaluation
features = [
    "HomeTeam_enc",
    "AwayTeam_enc",
    "goal_diff",
    "home_team_form",
    "away_team_form",
]
X = df[features]
y = df["FTR_encoded"]

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("🔄 Training model...")
model = train_model(df)

# Evaluate model
print("\n📊 Evaluating model performance...")
performance = evaluate_model_performance(model, X_test, y_test)

print(f"\n✅ Model Accuracy: {performance['accuracy']:.2%}")
print("\n📈 Classification Report:")
print(
    f"  Home Win - Precision: {performance['classification_report']['Home Win']['precision']:.2%}, "
    f"Recall: {performance['classification_report']['Home Win']['recall']:.2%}"
)
print(
    f"  Draw     - Precision: {performance['classification_report']['Draw']['precision']:.2%}, "
    f"Recall: {performance['classification_report']['Draw']['recall']:.2%}"
)
print(
    f"  Away Win - Precision: {performance['classification_report']['Away Win']['precision']:.2%}, "
    f"Recall: {performance['classification_report']['Away Win']['recall']:.2%}"
)

print("\n🔍 Feature Importance:")
feature_importance = get_feature_importance(model, features)
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\n🎯 Confusion Matrix:")
print(performance["confusion_matrix"])
print("  (Rows: Actual, Columns: Predicted)")
print("  Order: Home Win, Draw, Away Win")

# Save model and encoder
joblib.dump(model, "model/match_predictor.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("\n✅ Training complete. Model saved!")
