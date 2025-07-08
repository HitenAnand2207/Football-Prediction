import joblib
import pandas as pd

# Load saved model and encoder
model = joblib.load("model/match_predictor.pkl")
le = joblib.load("model/label_encoder.pkl")

# Load CSV and recent features
df = pd.read_csv("data/matches.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["home_team_form"] = df["FTR"].map({"H": 1, "D": 0, "A": 0}).rolling(3, min_periods=1).mean()
df["away_team_form"] = df["FTR"].map({"A": 1, "D": 0, "H": 0}).rolling(3, min_periods=1).mean()

# Example prediction
home = "Arsenal"
away = "Liverpool"

# Make sure both teams exist in label encoder
if home not in le.classes_ or away not in le.classes_:
    print("One of the teams is not in the label encoder.")
else:
    h_enc = le.transform([home])[0]
    a_enc = le.transform([away])[0]
    h_form = df[df.HomeTeam == home]["home_team_form"].iloc[-1]
    a_form = df[df.AwayTeam == away]["away_team_form"].iloc[-1]

    X = pd.DataFrame([{
        "HomeTeam_enc": h_enc,
        "AwayTeam_enc": a_enc,
        "goal_diff": 0,
        "home_team_form": h_form,
        "away_team_form": a_form
    }])

    pred = model.predict(X)[0]
    label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    print(f"🏟️ Predicted result: {label_map[pred]}")
