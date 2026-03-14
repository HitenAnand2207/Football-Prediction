import joblib
import pandas as pd
from model.predictor import predict_match, get_head_to_head, get_team_stats
from utils.analytics import (
    calculate_betting_odds,
    predict_score,
    generate_match_insights,
)

# Load saved model and encoder
model = joblib.load("model/match_predictor.pkl")
le = joblib.load("model/label_encoder.pkl")

# Load CSV and recent features
df = pd.read_csv("data/matches.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
df["home_team_form"] = (
    df.groupby("HomeTeam")["FTR"]
    .apply(lambda x: x.map({"H": 1, "D": 0, "A": 0}).rolling(3, min_periods=1).mean())
    .reset_index(0, drop=True)
)
df["away_team_form"] = (
    df.groupby("AwayTeam")["FTR"]
    .apply(lambda x: x.map({"A": 1, "D": 0, "H": 0}).rolling(3, min_periods=1).mean())
    .reset_index(0, drop=True)
)

# Example prediction with enhanced features
home = "Arsenal"
away = "Liverpool"

print("=" * 60)
print(f"🏟️  MATCH PREDICTION: {home} vs {away}")
print("=" * 60)

# Make sure both teams exist in label encoder
if home not in le.classes_ or away not in le.classes_:
    print("❌ One of the teams is not in the label encoder.")
else:
    # Basic prediction
    prediction, probabilities = predict_match(home, away, df, model, le)

    print(f"\n🎯 PREDICTED RESULT: {prediction}")
    print(f"   Confidence: {max(probabilities) * 100:.1f}%")

    print(f"\n📊 PROBABILITY BREAKDOWN:")
    print(f"   Home Win ({home}): {probabilities[0] * 100:.1f}%")
    print(f"   Draw:              {probabilities[1] * 100:.1f}%")
    print(f"   Away Win ({away}): {probabilities[2] * 100:.1f}%")

    # Betting odds
    odds = calculate_betting_odds(probabilities)
    print(f"\n💰 BETTING ODDS (Decimal):")
    for outcome, odd in odds.items():
        print(f"   {outcome}: {odd}")

    # Score prediction
    score_pred = predict_score(home, away, df)
    if score_pred:
        print(f"\n⚽ PREDICTED SCORE:")
        print(f"   Most Likely: {score_pred['most_likely']}")
        print(f"   Alternative: {score_pred['alternative']}")
        print(f"   Expected Goals (xG):")
        print(f"      {home}: {score_pred['xg']['home_xg']}")
        print(f"      {away}: {score_pred['xg']['away_xg']}")

    # Head-to-head stats
    h2h = get_head_to_head(home, away, df)
    if h2h:
        print(f"\n📈 HEAD-TO-HEAD (Last {h2h['total_matches']} matches):")
        print(f"   {home} wins: {h2h['home_wins']}")
        print(f"   Draws: {h2h['draws']}")
        print(f"   {away} wins: {h2h['away_wins']}")

    # Team stats
    print(f"\n📊 TEAM STATISTICS:")
    home_stats = get_team_stats(home, df)
    away_stats = get_team_stats(away, df)

    if home_stats:
        print(f"\n   {home}:")
        print(f"      Win Rate: {home_stats['win_rate']:.1f}%")
        print(f"      Goals Scored: {home_stats['goals_scored']}")
        print(f"      Goal Difference: {home_stats['goal_difference']:+d}")
        print(f"      Recent Form: {home_stats['recent_form']}")

    if away_stats:
        print(f"\n   {away}:")
        print(f"      Win Rate: {away_stats['win_rate']:.1f}%")
        print(f"      Goals Scored: {away_stats['goals_scored']}")
        print(f"      Goal Difference: {away_stats['goal_difference']:+d}")
        print(f"      Recent Form: {away_stats['recent_form']}")

    # Match intelligence summary
    insights = generate_match_insights(
        home, away, probabilities, home_stats, away_stats
    )
    print(f"\n🧠 MATCH INTELLIGENCE:")
    print(f"   Confidence Tier: {insights['confidence_tier']}")
    print(f"   Upset Alert: {'Yes' if insights['upset_alert'] else 'No'}")
    print(f"   Note: {insights['upset_reason']}")

print("\n" + "=" * 60)
