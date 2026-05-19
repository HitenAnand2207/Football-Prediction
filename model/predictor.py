import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from utils.features import build_match_features, get_feature_columns


def train_model(df):
    features = get_feature_columns()
    X = df[features]
    y = df["FTR_encoded"]  # 0 = Home Win, 1 = Draw, 2 = Away Win

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base XGBoost model
    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        min_child_weight=2,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
    )

    # Quick cross-validation to estimate performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(base_model, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    except Exception:
        cv_scores = None

    # Fit and calibrate probabilities for better confidence estimates
    calibrator = CalibratedClassifierCV(base_model, cv=5, method="sigmoid")
    calibrator.fit(X_train, y_train)

    accuracy = calibrator.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2%}")

    return calibrator


def predict_match(home, away, df, model, le):
    features = build_match_features(home, away, df, le)

    result = int(model.predict(features)[0])
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
