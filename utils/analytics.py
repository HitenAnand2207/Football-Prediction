import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils.features import summarize_team_history


def get_confidence_tier(confidence):
    """Classify prediction confidence into simple tiers."""
    if confidence >= 0.60:
        return "High"
    if confidence >= 0.45:
        return "Medium"
    return "Low"


def generate_match_insights(
    home_team,
    away_team,
    probabilities,
    home_stats=None,
    away_stats=None,
    home_snapshot=None,
    away_snapshot=None,
):
    """Generate lightweight match intelligence from probabilities and team stats."""
    labels = ["Home Win", "Draw", "Away Win"]
    top_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
    confidence = float(probabilities[top_idx])
    predicted_outcome = labels[top_idx]

    upset_alert = False
    reason = "No major upset signal detected."
    edge_notes = []

    home_win_rate = (home_stats or {}).get("win_rate")
    away_win_rate = (away_stats or {}).get("win_rate")

    if (
        predicted_outcome == "Home Win"
        and home_win_rate is not None
        and away_win_rate is not None
        and away_win_rate - home_win_rate >= 8
    ):
        upset_alert = True
        reason = (
            f"{away_team} has a notably stronger historical win rate, so a {home_team} win "
            "would be an upset."
        )
    elif (
        predicted_outcome == "Away Win"
        and home_win_rate is not None
        and away_win_rate is not None
        and home_win_rate - away_win_rate >= 8
    ):
        upset_alert = True
        reason = (
            f"{home_team} has a notably stronger historical win rate, so a {away_team} win "
            "would be an upset."
        )

    if home_snapshot and away_snapshot:
        momentum_gap = home_snapshot.get("points_per_match", 0) - away_snapshot.get(
            "points_per_match", 0
        )
        attack_gap = home_snapshot.get("goals_for_form", 0) - away_snapshot.get(
            "goals_against_form", 0
        )
        defense_gap = away_snapshot.get("goals_for_form", 0) - home_snapshot.get(
            "goals_against_form", 0
        )

        if abs(momentum_gap) >= 0.4:
            stronger = home_team if momentum_gap > 0 else away_team
            edge_notes.append(f"Momentum edge leans toward {stronger}")

        if attack_gap >= 0.4:
            edge_notes.append(f"{home_team} is creating better chances recently")

        if defense_gap >= 0.4:
            edge_notes.append(f"{away_team} has been more stable defensively")

        if edge_notes and reason == "No major upset signal detected.":
            reason = "; ".join(edge_notes)

    return {
        "predicted_outcome": predicted_outcome,
        "confidence": confidence,
        "confidence_tier": get_confidence_tier(confidence),
        "upset_alert": upset_alert,
        "upset_reason": reason,
    }


def get_match_edge_summary(home_team, away_team, df, window=5):
    """Build a compact pre-match edge summary for the UI."""
    home_snapshot = summarize_team_history(home_team, df, venue="home", window=window)
    away_snapshot = summarize_team_history(away_team, df, venue="away", window=window)

    return {
        "home_snapshot": home_snapshot,
        "away_snapshot": away_snapshot,
        "momentum_gap": home_snapshot.get("points_per_match", 0)
        - away_snapshot.get("points_per_match", 0),
        "attack_gap": home_snapshot.get("goals_for_form", 0)
        - away_snapshot.get("goals_against_form", 0),
        "defense_gap": away_snapshot.get("goals_for_form", 0)
        - home_snapshot.get("goals_against_form", 0),
        "shot_gap": home_snapshot.get("shots_on_target_form", 0)
        - away_snapshot.get("shots_on_target_form", 0),
        "discipline_gap": away_snapshot.get("cards_form", 0)
        - home_snapshot.get("cards_form", 0),
    }


def get_team_recent_trend(team, df, last_n=10):
    """Return recent match-by-match trend data for a team."""
    matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    if matches.empty:
        return pd.DataFrame()

    matches = matches.sort_values("Date").tail(last_n)
    rows = []

    for _, match in matches.iterrows():
        is_home = match["HomeTeam"] == team
        opponent = match["AwayTeam"] if is_home else match["HomeTeam"]
        goals_for = int(match["FTHG"] if is_home else match["FTAG"])
        goals_against = int(match["FTAG"] if is_home else match["FTHG"])

        if goals_for > goals_against:
            result = "W"
            points = 3
        elif goals_for == goals_against:
            result = "D"
            points = 1
        else:
            result = "L"
            points = 0

        rows.append(
            {
                "Date": match["Date"],
                "Opponent": opponent,
                "Venue": "Home" if is_home else "Away",
                "Result": result,
                "Points": points,
                "Goals For": goals_for,
                "Goals Against": goals_against,
            }
        )

    trend_df = pd.DataFrame(rows)
    if trend_df.empty:
        return trend_df

    trend_df["Cumulative Points"] = trend_df["Points"].cumsum()
    trend_df["Form (Last 3)"] = trend_df["Points"].rolling(3, min_periods=1).mean()
    return trend_df


def evaluate_model_performance(model, X_test, y_test):
    """Evaluate model performance with detailed metrics"""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Get classification report as dict
    report = classification_report(
        y_test,
        predictions,
        target_names=["Home Win", "Draw", "Away Win"],
        output_dict=True,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def get_feature_importance(model, feature_names):
    """Get feature importance from the model"""
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    return feature_importance_df


def calculate_betting_odds(probabilities):
    """Convert probabilities to betting odds (decimal format)"""
    # Decimal odds = 1 / probability
    odds = {}
    labels = ["Home Win", "Draw", "Away Win"]

    for i, label in enumerate(labels):
        if probabilities[i] > 0:
            odds[label] = round(1 / probabilities[i], 2)
        else:
            odds[label] = 999.0  # Very high odds for near-impossible events

    return odds


def estimate_expected_goals(home_team, away_team, df):
    """Estimate expected goals (xG) for both teams"""
    # Get historical goal-scoring data
    home_games = df[df["HomeTeam"] == home_team]
    away_games = df[df["AwayTeam"] == away_team]

    if len(home_games) == 0 or len(away_games) == 0:
        return None

    # Calculate average goals scored/conceded
    home_avg_scored = home_games["FTHG"].mean()
    home_avg_conceded = home_games["FTAG"].mean()

    away_avg_scored = away_games["FTAG"].mean()
    away_avg_conceded = away_games["FTHG"].mean()

    # Simple xG estimation based on averages
    home_xg = (home_avg_scored + away_avg_conceded) / 2
    away_xg = (away_avg_scored + home_avg_conceded) / 2

    return {
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "total_xg": round(home_xg + away_xg, 2),
    }


def get_league_table(df, teams=None):
    """Generate league table standings from match data"""
    standings = {}

    team_list = teams if teams else df["HomeTeam"].unique()

    for team in team_list:
        home_games = df[df["HomeTeam"] == team]
        away_games = df[df["AwayTeam"] == team]

        # Points calculation
        home_wins = len(home_games[home_games["FTR"] == "H"])
        home_draws = len(home_games[home_games["FTR"] == "D"])
        away_wins = len(away_games[away_games["FTR"] == "A"])
        away_draws = len(away_games[away_games["FTR"] == "D"])

        wins = home_wins + away_wins
        draws = home_draws + away_draws
        played = len(home_games) + len(away_games)
        losses = played - wins - draws

        points = (wins * 3) + draws

        goals_for = home_games["FTHG"].sum() + away_games["FTAG"].sum()
        goals_against = home_games["FTAG"].sum() + away_games["FTHG"].sum()
        goal_diff = goals_for - goals_against

        standings[team] = {
            "played": played,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": int(goals_for),
            "goals_against": int(goals_against),
            "goal_diff": int(goal_diff),
            "points": points,
        }

    # Convert to DataFrame and sort by points
    standings_df = pd.DataFrame.from_dict(standings, orient="index")
    standings_df = standings_df.sort_values(
        ["points", "goal_diff", "goals_for"], ascending=[False, False, False]
    )
    standings_df.insert(0, "position", range(1, len(standings_df) + 1))

    return standings_df


def predict_score(home_team, away_team, df):
    """Predict the most likely score based on xG and historical data"""
    xg_data = estimate_expected_goals(home_team, away_team, df)

    if not xg_data:
        return None

    # Round to nearest integer for most likely score
    home_score = round(xg_data["home_xg"])
    away_score = round(xg_data["away_xg"])

    # Add some variation based on historical variance
    home_games = df[df["HomeTeam"] == home_team]
    if len(home_games) > 0:
        home_std = home_games["FTHG"].std()
        # Generate alternative scores
        alt_home = max(0, int(xg_data["home_xg"] + (home_std * 0.5)))
    else:
        alt_home = home_score

    away_games = df[df["AwayTeam"] == away_team]
    if len(away_games) > 0:
        away_std = away_games["FTAG"].std()
        alt_away = max(0, int(xg_data["away_xg"] + (away_std * 0.5)))
    else:
        alt_away = away_score

    return {
        "most_likely": f"{home_score}-{away_score}",
        "alternative": f"{alt_home}-{alt_away}",
        "xg": xg_data,
    }


def _venue_matches(team, df, venue):
    if venue == "home":
        return df[df["HomeTeam"] == team].copy()
    if venue == "away":
        return df[df["AwayTeam"] == team].copy()
    return df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()


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

    # Shots, corners, and discipline
    shots_for = home_games.get("HS", pd.Series(dtype=float)).sum() + away_games.get(
        "AS", pd.Series(dtype=float)
    ).sum()
    shots_against = home_games.get("AS", pd.Series(dtype=float)).sum() + away_games.get(
        "HS", pd.Series(dtype=float)
    ).sum()
    shots_on_target_for = home_games.get("HST", pd.Series(dtype=float)).sum() + away_games.get(
        "AST", pd.Series(dtype=float)
    ).sum()
    corners_for = home_games.get("HC", pd.Series(dtype=float)).sum() + away_games.get(
        "AC", pd.Series(dtype=float)
    ).sum()
    cards_for = home_games.get("HY", pd.Series(dtype=float)).sum() + away_games.get(
        "AY", pd.Series(dtype=float)
    ).sum() + (home_games.get("HR", pd.Series(dtype=float)).sum() + away_games.get("AR", pd.Series(dtype=float)).sum()) * 2

    # Recent form (last 5 games)
    all_games = _venue_matches(team, df, None).sort_values("Date")
    recent_games = all_games.tail(5)

    recent_form = []
    for _, game in recent_games.iterrows():
        is_home = game["HomeTeam"] == team
        if is_home:
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

    recent_snapshot = summarize_team_history(team, df, venue=None, window=5)
    home_snapshot = summarize_team_history(team, df, venue="home", window=5)
    away_snapshot = summarize_team_history(team, df, venue="away", window=5)

    total_matches = len(all_games)
    clean_sheets = len(all_games[((all_games["HomeTeam"] == team) & (all_games["FTAG"] == 0)) | ((all_games["AwayTeam"] == team) & (all_games["FTHG"] == 0))])
    btts = len(all_games[(all_games["FTHG"] > 0) & (all_games["FTAG"] > 0)])
    over25 = len(all_games[(all_games["FTHG"] + all_games["FTAG"]) >= 3])

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
        "shots_for": float(shots_for),
        "shots_against": float(shots_against),
        "shots_on_target_for": float(shots_on_target_for),
        "shot_accuracy": float(shots_on_target_for / shots_for) if shots_for else 0.0,
        "corners_for": float(corners_for),
        "cards_for": float(cards_for),
        "clean_sheet_rate": (clean_sheets / total_matches) * 100 if total_matches else 0.0,
        "btts_rate": (btts / total_matches) * 100 if total_matches else 0.0,
        "over25_rate": (over25 / total_matches) * 100 if total_matches else 0.0,
        "recent_snapshot": recent_snapshot,
        "home_snapshot": home_snapshot,
        "away_snapshot": away_snapshot,
    }
