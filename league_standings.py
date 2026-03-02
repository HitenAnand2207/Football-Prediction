"""
League Standings Generator
Generates league table from match data
"""

import pandas as pd
from utils.analytics import get_league_table

# Load match data
df = pd.read_csv("data/matches.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.dropna(subset=["FTHG", "FTAG", "FTR"])

print("=" * 80)
print("🏆 LEAGUE STANDINGS")
print("=" * 80)

# Generate league table
standings = get_league_table(df)

# Display the table
print(
    f"\n{'Pos':<4} {'Team':<20} {'P':<4} {'W':<4} {'D':<4} {'L':<4} {'GF':<4} {'GA':<4} {'GD':<6} {'Pts':<4}"
)
print("-" * 80)

for idx, row in standings.iterrows():
    print(
        f"{row['position']:<4} {idx:<20} {row['played']:<4} {row['wins']:<4} "
        f"{row['draws']:<4} {row['losses']:<4} {row['goals_for']:<4} "
        f"{row['goals_against']:<4} {row['goal_diff']:+6} {row['points']:<4}"
    )

print("-" * 80)
print("\n📊 Top 5 Teams by Points:")
top_5 = standings.head(5)
for pos, (team, data) in enumerate(top_5.iterrows(), 1):
    print(f"   {pos}. {team} - {data['points']} points")

print("\n⚽ Top Scorers (Teams):")
top_scorers = standings.nlargest(5, "goals_for")
for pos, (team, data) in enumerate(top_scorers.iterrows(), 1):
    print(f"   {pos}. {team} - {data['goals_for']} goals")

print("\n🛡️ Best Defense (Fewest Goals Conceded):")
best_defense = standings.nsmallest(5, "goals_against")
for pos, (team, data) in enumerate(best_defense.iterrows(), 1):
    print(f"   {pos}. {team} - {data['goals_against']} goals conceded")

print("\n" + "=" * 80)
