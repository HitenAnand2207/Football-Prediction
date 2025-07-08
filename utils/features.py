import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_features(df):
    # Encode teams
    le = LabelEncoder()
    df["HomeTeam_enc"] = le.fit_transform(df["HomeTeam"])
    df["AwayTeam_enc"] = le.transform(df["AwayTeam"])

    # Goal difference
    df["goal_diff"] = df["FTHG"] - df["FTAG"]

    # Sort by date for rolling calculation
    df = df.sort_values("Date")

    # Result indicators
    df["home_win"] = df["FTR"].map({"H": 1, "D": 0, "A": 0})
    df["away_win"] = df["FTR"].map({"A": 1, "D": 0, "H": 0})

    # Rolling form (last 3 matches)
    df["home_team_form"] = df.groupby("HomeTeam")["home_win"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df["away_team_form"] = df.groupby("AwayTeam")["away_win"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)

    return df, le
