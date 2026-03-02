import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_model(df):
    features = [
        "HomeTeam_enc",
        "AwayTeam_enc",
        "goal_diff",
        "home_team_form",
        "away_team_form",
    ]
    X = df[features]
    y = df["FTR_encoded"]  # 0 = Home Win, 1 = Draw, 2 = Away Win

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
    goal_diff = (home_avg_gd if pd.notna(home_avg_gd) else 0) - (
        away_avg_gd if pd.notna(away_avg_gd) else 0
    )

    # Fetch recent form
    h_form = (
        df[df["HomeTeam"] == home]["home_team_form"].iloc[-1]
        if not df[df["HomeTeam"] == home].empty
        else 0.5
    )
    a_form = (
        df[df["AwayTeam"] == away]["away_team_form"].iloc[-1]
        if not df[df["AwayTeam"] == away].empty
        else 0.5
    )

    # Create prediction feature row
    features = pd.DataFrame(
        [
            {
                "HomeTeam_enc": h_enc,
                "AwayTeam_enc": a_enc,
                "goal_diff": goal_diff,
                "home_team_form": h_form,
                "away_team_form": a_form,
            }
        ]
    )

    result = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}

    return outcome_map.get(result, "Unknown"), probabilities


def get_team_stats(team, df):
    """Get comprehensive statistics for a team"""
    home_games = df[df["HomeTeam"] == team]
    away_games = df[df["AwayTeam"] == team]

    total_games = len(home_games) + len(away_games)

    if total_games == 0:
        return None

    # Calculate wins, draws, losses
    home_wins = len(home_games[home_games["FTR"] == "H"])
    away_wins = len(away_games[away_games["FTR"] == "A"])
    home_draws = len(home_games[home_games["FTR"] == "D"])
    away_draws = len(away_games[away_games["FTR"] == "D"])

    total_wins = home_wins + away_wins
    total_draws = home_draws + away_draws
    total_losses = total_games - total_wins - total_draws

    # Goals statistics
    goals_scored = home_games["FTHG"].sum() + away_games["FTAG"].sum()
    goals_conceded = home_games["FTAG"].sum() + away_games["FTHG"].sum()

    # Recent form (last 5 games)
    all_games = (
        pd.concat([home_games.assign(is_home=True), away_games.assign(is_home=False)])
        .sort_values("Date")
        .tail(5)
    )

    recent_form = []
    for _, game in all_games.iterrows():
        if game["is_home"]:
            if game["FTR"] == "H":
                recent_form.append("W")
            elif game["FTR"] == "D":
                recent_form.append("D")
            else:
                recent_form.append("L")
        else:
            if game["FTR"] == "A":
                recent_form.append("W")
            elif game["FTR"] == "D":
                recent_form.append("D")
            else:
                recent_form.append("L")

    return {
        "total_games": total_games,
        "wins": total_wins,
        "draws": total_draws,
        "losses": total_losses,
        "win_rate": (total_wins / total_games) * 100,
        "goals_scored": int(goals_scored),
        "goals_conceded": int(goals_conceded),
        "goal_difference": int(goals_scored - goals_conceded),
        "avg_goals_scored": goals_scored / total_games,
        "avg_goals_conceded": goals_conceded / total_games,
        "recent_form": "".join(recent_form),
    }


def get_head_to_head(home, away, df):
    """Get head-to-head statistics between two teams"""
    h2h = df[
        ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
        | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
    ].copy()

    if len(h2h) == 0:
        return None

    home_wins = 0
    away_wins = 0
    draws = 0

    for _, match in h2h.iterrows():
        if match["HomeTeam"] == home:
            if match["FTR"] == "H":
                home_wins += 1
            elif match["FTR"] == "D":
                draws += 1
            else:
                away_wins += 1
        else:
            if match["FTR"] == "H":
                away_wins += 1
            elif match["FTR"] == "D":
                draws += 1
            else:
                home_wins += 1

    return {
        "total_matches": len(h2h),
        "home_wins": home_wins,
        "away_wins": away_wins,
        "draws": draws,
        "last_5": h2h.tail(5)[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]],
    }
