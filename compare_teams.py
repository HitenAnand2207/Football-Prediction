"""
Team Comparison Tool
Compare statistics between multiple teams
"""

import pandas as pd
import sys
from model.predictor import get_team_stats

# Load match data
df = pd.read_csv("data/matches.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.dropna(subset=["FTHG", "FTAG", "FTR"])


def compare_teams(*teams):
    """Compare statistics for multiple teams"""

    if len(teams) < 2:
        print("❌ Please provide at least 2 teams to compare")
        return

    print("=" * 100)
    print("⚔️  TEAM COMPARISON")
    print("=" * 100)

    stats_list = []

    for team in teams:
        stats = get_team_stats(team, df)
        if stats:
            stats["team"] = team
            stats_list.append(stats)
        else:
            print(f"⚠️  No data found for {team}")

    if not stats_list:
        print("❌ No valid teams to compare")
        return

    # Print header
    print(
        f"\n{'Team':<20} {'Games':<8} {'Win%':<8} {'Goals':<8} {'GD':<8} {'Form':<10}"
    )
    print("-" * 100)

    # Print each team
    for stats in stats_list:
        print(
            f"{stats['team']:<20} {stats['total_games']:<8} "
            f"{stats['win_rate']:<7.1f}% {stats['goals_scored']:<8} "
            f"{stats['goal_difference']:<+8} {stats['recent_form']:<10}"
        )

    print("-" * 100)

    # Find best in each category
    df_stats = pd.DataFrame(stats_list)

    print("\n🏆 CATEGORY LEADERS:")
    print(
        f"   Best Win Rate: {df_stats.loc[df_stats['win_rate'].idxmax(), 'team']} "
        f"({df_stats['win_rate'].max():.1f}%)"
    )
    print(
        f"   Most Goals: {df_stats.loc[df_stats['goals_scored'].idxmax(), 'team']} "
        f"({df_stats['goals_scored'].max()} goals)"
    )
    print(
        f"   Best Goal Difference: {df_stats.loc[df_stats['goal_difference'].idxmax(), 'team']} "
        f"({df_stats['goal_difference'].max():+d})"
    )
    print(
        f"   Fewest Goals Conceded: {df_stats.loc[df_stats['goals_conceded'].idxmin(), 'team']} "
        f"({df_stats['goals_conceded'].min()} goals)"
    )

    # Recent form comparison
    print("\n📅 RECENT FORM (Last 5 matches):")
    for stats in stats_list:
        wins = stats["recent_form"].count("W")
        draws = stats["recent_form"].count("D")
        losses = stats["recent_form"].count("L")
        print(
            f"   {stats['team']:<20} {stats['recent_form']:<10} "
            f"({wins}W {draws}D {losses}L)"
        )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    # Example: Compare multiple teams
    # You can modify this list or pass teams as command line arguments

    if len(sys.argv) > 1:
        # Use command line arguments
        teams_to_compare = sys.argv[1:]
    else:
        # Default comparison
        teams_to_compare = ["Arsenal", "Liverpool", "Man City", "Chelsea", "Man United"]

    print(f"\n🔍 Comparing {len(teams_to_compare)} teams...\n")
    compare_teams(*teams_to_compare)

    print(
        "\n💡 TIP: Run with custom teams: python compare_teams.py 'Arsenal' 'Liverpool' 'Chelsea'"
    )
