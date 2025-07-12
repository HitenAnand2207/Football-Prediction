import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def train_model(df):
    features = ["HomeTeam_enc", "AwayTeam_enc", "goal_diff", "home_team_form", "away_team_form"]
    X = df[features]
    y = df["FTR_encoded"]  # 0 = Home Win, 1 = Draw, 2 = Away Win

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2%}")

    return model

def predict_match(home, away, df, model, le):
    # Encode team names
    h_enc = le.transform([home])[0]
    a_enc = le.transform([away])[0]

    # Estimate average goal differences from recent history
    home_avg_gd = df[df["HomeTeam"] == home]["goal_diff"].mean()
    away_avg_gd = df[df["AwayTeam"] == away]["goal_diff"].mean()
    goal_diff = (home_avg_gd if pd.notna(home_avg_gd) else 0) - (away_avg_gd if pd.notna(away_avg_gd) else 0)

    # Fetch recent form
    h_form = df[df["HomeTeam"] == home]["home_team_form"].iloc[-1] if not df[df["HomeTeam"] == home].empty else 0.5
    a_form = df[df["AwayTeam"] == away]["away_team_form"].iloc[-1] if not df[df["AwayTeam"] == away].empty else 0.5

    # Create prediction feature row
    features = pd.DataFrame([{
        "HomeTeam_enc": h_enc,
        "AwayTeam_enc": a_enc,
        "goal_diff": goal_diff,
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
