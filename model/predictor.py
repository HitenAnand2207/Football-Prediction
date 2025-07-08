from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

def train_model(df):
    features = ["HomeTeam_enc", "AwayTeam_enc", "goal_diff", "home_team_form", "away_team_form"]
    X = df[features]
    y = df["FTR_encoded"]  # Use the encoded target column: 0=Home, 1=Draw, 2=Away
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2%}")
    return model

def predict_match(home, away, df, model, le):
    h_enc = le.transform([home])[0]
    a_enc = le.transform([away])[0]

    # Safely get the latest form
    h_form = df[df.HomeTeam == home]["home_team_form"].iloc[-1] if not df[df.HomeTeam == home].empty else 0.5
    a_form = df[df.AwayTeam == away]["away_team_form"].iloc[-1] if not df[df.AwayTeam == away].empty else 0.5

    features = pd.DataFrame([{
        "HomeTeam_enc": h_enc,
        "AwayTeam_enc": a_enc,
        "goal_diff": 0,
        "home_team_form": h_form,
        "away_team_form": a_form
    }])

    result = model.predict(features)[0]

    outcome_map = {
        0: "Home Win",
        1: "Draw",
        2: "Away Win"
    }

    return outcome_map.get(result, "Unknown")
