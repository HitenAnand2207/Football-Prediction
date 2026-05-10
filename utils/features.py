from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

WINDOW_SIZE = 5

FEATURE_COLUMNS = [
    "HomeTeam_enc",
    "AwayTeam_enc",
    "home_team_form",
    "away_team_form",
    "home_goal_diff_form",
    "away_goal_diff_form",
    "home_goals_for_form",
    "away_goals_for_form",
    "home_goals_against_form",
    "away_goals_against_form",
    "home_shots_for_form",
    "away_shots_for_form",
    "home_shots_on_target_form",
    "away_shots_on_target_form",
    "home_corner_for_form",
    "away_corner_for_form",
    "home_cards_form",
    "away_cards_form",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    "home_btts_rate",
    "away_btts_rate",
    "home_over25_rate",
    "away_over25_rate",
    "momentum_gap",
    "attack_gap",
    "defense_gap",
    "shot_gap",
    "discipline_gap",
]


def get_feature_columns():
    return FEATURE_COLUMNS.copy()


def _safe_value(value, default=0.0):
    if pd.isna(value):
        return default
    return float(value)


def _safe_mean(values, default=0.0):
    cleaned = [_safe_value(value, default=np.nan) for value in values]
    cleaned = [value for value in cleaned if not pd.isna(value)]
    if not cleaned:
        return float(default)
    return float(np.mean(cleaned))


def _history_snapshot(history, window=WINDOW_SIZE):
    recent = history[-window:]
    if not recent:
        return {
            "matches": 0,
            "points_per_match": 0.0,
            "goal_diff_form": 0.0,
            "goals_for_form": 0.0,
            "goals_against_form": 0.0,
            "shots_for_form": 0.0,
            "shots_on_target_form": 0.0,
            "corner_for_form": 0.0,
            "cards_form": 0.0,
            "clean_sheet_rate": 0.0,
            "btts_rate": 0.0,
            "over25_rate": 0.0,
        }

    points_per_match = _safe_mean([item["points"] for item in recent])
    goals_for_form = _safe_mean([item["goals_for"] for item in recent])
    goals_against_form = _safe_mean([item["goals_against"] for item in recent])

    return {
        "matches": len(recent),
        "points_per_match": points_per_match,
        "goal_diff_form": goals_for_form - goals_against_form,
        "goals_for_form": goals_for_form,
        "goals_against_form": goals_against_form,
        "shots_for_form": _safe_mean([item["shots_for"] for item in recent]),
        "shots_on_target_form": _safe_mean([item["shots_on_target_for"] for item in recent]),
        "corner_for_form": _safe_mean([item["corners_for"] for item in recent]),
        "cards_form": _safe_mean([item["cards_for"] for item in recent]),
        "clean_sheet_rate": _safe_mean([item["clean_sheet"] for item in recent]),
        "btts_rate": _safe_mean([item["btts"] for item in recent]),
        "over25_rate": _safe_mean([item["over25"] for item in recent]),
    }


def _fallback_snapshot(primary, fallback):
    if primary.get("matches", 0) > 0:
        return primary
    return fallback


def _record_from_row(row, team, is_home):
    goals_for = row["FTHG"] if is_home else row["FTAG"]
    goals_against = row["FTAG"] if is_home else row["FTHG"]

    if goals_for > goals_against:
        points = 3
    elif goals_for == goals_against:
        points = 1
    else:
        points = 0

    shots_for = row.get("HS", 0) if is_home else row.get("AS", 0)
    shots_against = row.get("AS", 0) if is_home else row.get("HS", 0)
    shots_on_target_for = row.get("HST", 0) if is_home else row.get("AST", 0)
    corners_for = row.get("HC", 0) if is_home else row.get("AC", 0)
    yellow_cards = row.get("HY", 0) if is_home else row.get("AY", 0)
    red_cards = row.get("HR", 0) if is_home else row.get("AR", 0)
    cards_for = _safe_value(yellow_cards) + (_safe_value(red_cards) * 2)

    return {
        "team": team,
        "points": points,
        "goals_for": _safe_value(goals_for),
        "goals_against": _safe_value(goals_against),
        "shots_for": _safe_value(shots_for),
        "shots_against": _safe_value(shots_against),
        "shots_on_target_for": _safe_value(shots_on_target_for),
        "corners_for": _safe_value(corners_for),
        "cards_for": cards_for,
        "clean_sheet": 1.0 if goals_against == 0 else 0.0,
        "btts": 1.0 if goals_for > 0 and goals_against > 0 else 0.0,
        "over25": 1.0 if (goals_for + goals_against) >= 3 else 0.0,
    }


def _build_feature_row(home_snapshot, away_snapshot, home_team_enc, away_team_enc):
    home_points_form = home_snapshot["points_per_match"] / 3.0
    away_points_form = away_snapshot["points_per_match"] / 3.0

    return {
        "HomeTeam_enc": int(home_team_enc),
        "AwayTeam_enc": int(away_team_enc),
        "home_team_form": home_points_form,
        "away_team_form": away_points_form,
        "home_goal_diff_form": home_snapshot["goal_diff_form"],
        "away_goal_diff_form": away_snapshot["goal_diff_form"],
        "home_goals_for_form": home_snapshot["goals_for_form"],
        "away_goals_for_form": away_snapshot["goals_for_form"],
        "home_goals_against_form": home_snapshot["goals_against_form"],
        "away_goals_against_form": away_snapshot["goals_against_form"],
        "home_shots_for_form": home_snapshot["shots_for_form"],
        "away_shots_for_form": away_snapshot["shots_for_form"],
        "home_shots_on_target_form": home_snapshot["shots_on_target_form"],
        "away_shots_on_target_form": away_snapshot["shots_on_target_form"],
        "home_corner_for_form": home_snapshot["corner_for_form"],
        "away_corner_for_form": away_snapshot["corner_for_form"],
        "home_cards_form": home_snapshot["cards_form"],
        "away_cards_form": away_snapshot["cards_form"],
        "home_clean_sheet_rate": home_snapshot["clean_sheet_rate"],
        "away_clean_sheet_rate": away_snapshot["clean_sheet_rate"],
        "home_btts_rate": home_snapshot["btts_rate"],
        "away_btts_rate": away_snapshot["btts_rate"],
        "home_over25_rate": home_snapshot["over25_rate"],
        "away_over25_rate": away_snapshot["over25_rate"],
        "momentum_gap": home_points_form - away_points_form,
        "attack_gap": home_snapshot["goals_for_form"] - away_snapshot["goals_against_form"],
        "defense_gap": away_snapshot["goals_for_form"] - home_snapshot["goals_against_form"],
        "shot_gap": home_snapshot["shots_on_target_form"] - away_snapshot["shots_on_target_form"],
        "discipline_gap": away_snapshot["cards_form"] - home_snapshot["cards_form"],
    }


def summarize_team_history(team, df, venue=None, window=WINDOW_SIZE):
    if df.empty:
        return _history_snapshot([])

    teams = pd.Index(df["HomeTeam"]).union(df["AwayTeam"])
    if team not in teams:
        return _history_snapshot([])

    history = []
    for _, row in df.sort_values("Date").iterrows():
        if row["HomeTeam"] == team:
            if venue in (None, "home"):
                history.append(_record_from_row(row, team, True))
        elif row["AwayTeam"] == team:
            if venue in (None, "away"):
                history.append(_record_from_row(row, team, False))

    return _history_snapshot(history, window=window)


def build_match_features(home_team, away_team, df, le, window=WINDOW_SIZE):
    if home_team not in le.classes_ or away_team not in le.classes_:
        raise ValueError("One or both teams are not present in the label encoder.")

    home_snapshot = _fallback_snapshot(
        summarize_team_history(home_team, df, venue="home", window=window),
        summarize_team_history(home_team, df, venue=None, window=window),
    )
    away_snapshot = _fallback_snapshot(
        summarize_team_history(away_team, df, venue="away", window=window),
        summarize_team_history(away_team, df, venue=None, window=window),
    )

    feature_row = _build_feature_row(
        home_snapshot,
        away_snapshot,
        le.transform([home_team])[0],
        le.transform([away_team])[0],
    )

    return pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)


def create_features(df, label_encoder=None, window=WINDOW_SIZE):
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    if label_encoder is None:
        label_encoder = LabelEncoder()
        teams = pd.Index(df["HomeTeam"]).union(df["AwayTeam"])
        label_encoder.fit(teams)

    home_histories = defaultdict(list)
    away_histories = defaultdict(list)
    overall_histories = defaultdict(list)

    feature_rows = []

    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_overall_snapshot = _history_snapshot(overall_histories[home_team], window=window)
        away_overall_snapshot = _history_snapshot(overall_histories[away_team], window=window)
        home_home_snapshot = _fallback_snapshot(
            _history_snapshot(home_histories[home_team], window=window),
            home_overall_snapshot,
        )
        away_away_snapshot = _fallback_snapshot(
            _history_snapshot(away_histories[away_team], window=window),
            away_overall_snapshot,
        )

        feature_rows.append(
            _build_feature_row(
                home_home_snapshot,
                away_away_snapshot,
                label_encoder.transform([home_team])[0],
                label_encoder.transform([away_team])[0],
            )
        )

        home_record = _record_from_row(row, home_team, True)
        away_record = _record_from_row(row, away_team, False)

        overall_histories[home_team].append(home_record)
        overall_histories[away_team].append(away_record)
        home_histories[home_team].append(home_record)
        away_histories[away_team].append(away_record)

    features_df = pd.DataFrame(feature_rows, index=df.index)
    df = pd.concat([df, features_df], axis=1)

    numeric_columns = [column for column in FEATURE_COLUMNS if column not in {"HomeTeam_enc", "AwayTeam_enc"}]
    df[numeric_columns] = df[numeric_columns].fillna(0.0)

    return df, label_encoder
