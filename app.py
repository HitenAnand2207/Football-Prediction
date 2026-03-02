import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

from model.predictor import predict_match, get_team_stats, get_head_to_head
from scraper.live_scrapper import get_today_matches
from utils.features import create_features
from utils.analytics import calculate_betting_odds, predict_score, get_league_table

# Page configuration
st.set_page_config(page_title="Football Predictor Pro", page_icon="⚽", layout="wide")

# Load data and apply feature engineering
df = pd.read_csv("data/matches.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df, _ = create_features(df)  # apply home/away form

# Load trained model and encoder
model = joblib.load("model/match_predictor.pkl")
le = joblib.load("model/label_encoder.pkl")

# Streamlit UI
st.title("⚽ Football Predictor Pro - AI-Powered Match Predictions")
st.markdown("---")

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Match Predictor", "📊 Team Analytics", "📺 Live Fixtures", "🏆 League Table"]
)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        home = st.selectbox("🏠 Home Team", sorted(df.HomeTeam.unique()), key="home")

    with col2:
        away = st.selectbox("✈️ Away Team", sorted(df.HomeTeam.unique()), key="away")

    if st.button("🔮 Predict Match", type="primary", use_container_width=True):
        if home == away:
            st.error("⚠️ Please pick different teams!")
        else:
            try:
                prediction, probabilities = predict_match(home, away, df, model, le)

                # Display prediction with confidence
                st.markdown("### 🏆 Prediction Result")

                # Create columns for the result
                col_home, col_draw, col_away = st.columns(3)

                with col_home:
                    st.metric(
                        "Home Win",
                        f"{probabilities[0] * 100:.1f}%",
                        delta="Most Likely"
                        if probabilities[0] == max(probabilities)
                        else None,
                    )

                with col_draw:
                    st.metric(
                        "Draw",
                        f"{probabilities[1] * 100:.1f}%",
                        delta="Most Likely"
                        if probabilities[1] == max(probabilities)
                        else None,
                    )

                with col_away:
                    st.metric(
                        "Away Win",
                        f"{probabilities[2] * 100:.1f}%",
                        delta="Most Likely"
                        if probabilities[2] == max(probabilities)
                        else None,
                    )

                # Confidence bar chart
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=["Home Win", "Draw", "Away Win"],
                            y=probabilities * 100,
                            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                            text=[f"{p * 100:.1f}%" for p in probabilities],
                            textposition="auto",
                        )
                    ]
                )
                fig.update_layout(
                    title="Prediction Confidence",
                    yaxis_title="Probability (%)",
                    showlegend=False,
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.success(
                    f"🎯 **Predicted Outcome:** {prediction} (Confidence: {max(probabilities) * 100:.1f}%)"
                )

                # Head-to-Head Stats
                h2h = get_head_to_head(home, away, df)
                if h2h:
                    st.markdown("### 📈 Head-to-Head History")
                    h2h_col1, h2h_col2, h2h_col3, h2h_col4 = st.columns(4)

                    with h2h_col1:
                        st.metric("Total Matches", h2h["total_matches"])
                    with h2h_col2:
                        st.metric(f"{home} Wins", h2h["home_wins"])
                    with h2h_col3:
                        st.metric("Draws", h2h["draws"])
                    with h2h_col4:
                        st.metric(f"{away} Wins", h2h["away_wins"])

                    # Show last 5 matches
                    with st.expander("📋 Last 5 Meetings"):
                        st.dataframe(h2h["last_5"], use_container_width=True)
                else:
                    st.info("ℹ️ No previous matches found between these teams.")

                # Team Stats Comparison
                st.markdown("### ⚔️ Team Statistics Comparison")
                home_stats = get_team_stats(home, df)
                away_stats = get_team_stats(away, df)

                if home_stats and away_stats:
                    comp_col1, comp_col2 = st.columns(2)

                    with comp_col1:
                        st.markdown(f"**{home} (Home)**")
                        st.write(f"🏆 Win Rate: {home_stats['win_rate']:.1f}%")
                        st.write(f"⚽ Goals Scored: {home_stats['goals_scored']}")
                        st.write(f"🥅 Goals Conceded: {home_stats['goals_conceded']}")
                        st.write(
                            f"📊 Goal Difference: {home_stats['goal_difference']:+d}"
                        )
                        st.write(f"📅 Recent Form: {home_stats['recent_form']}")

                    with comp_col2:
                        st.markdown(f"**{away} (Away)**")
                        st.write(f"🏆 Win Rate: {away_stats['win_rate']:.1f}%")
                        st.write(f"⚽ Goals Scored: {away_stats['goals_scored']}")
                        st.write(f"🥅 Goals Conceded: {away_stats['goals_conceded']}")
                        st.write(
                            f"📊 Goal Difference: {away_stats['goal_difference']:+d}"
                        )
                        st.write(f"📅 Recent Form: {away_stats['recent_form']}")

                # Betting Odds
                st.markdown("### 💰 Betting Odds (Decimal Format)")
                odds = calculate_betting_odds(probabilities)
                odds_col1, odds_col2, odds_col3 = st.columns(3)

                with odds_col1:
                    st.info(f"**Home Win:** {odds['Home Win']}")
                with odds_col2:
                    st.info(f"**Draw:** {odds['Draw']}")
                with odds_col3:
                    st.info(f"**Away Win:** {odds['Away Win']}")

                # Score Prediction
                st.markdown("### ⚽ Predicted Score")
                score_pred = predict_score(home, away, df)
                if score_pred:
                    score_col1, score_col2 = st.columns(2)

                    with score_col1:
                        st.success(f"**Most Likely:** {score_pred['most_likely']}")
                    with score_col2:
                        st.info(f"**Alternative:** {score_pred['alternative']}")

                    st.caption(
                        f"Expected Goals - {home}: {score_pred['xg']['home_xg']} | {away}: {score_pred['xg']['away_xg']}"
                    )

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

with tab2:
    st.subheader("📊 Detailed Team Analytics")

    selected_team = st.selectbox(
        "Select a team to analyze", sorted(df.HomeTeam.unique()), key="analytics"
    )

    team_stats = get_team_stats(selected_team, df)

    if team_stats:
        # Display key metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("Total Games", team_stats["total_games"])
        with metric_col2:
            st.metric("Win Rate", f"{team_stats['win_rate']:.1f}%")
        with metric_col3:
            st.metric("Goals Scored", team_stats["goals_scored"])
        with metric_col4:
            st.metric("Goal Difference", f"{team_stats['goal_difference']:+d}")

        # Win/Draw/Loss pie chart
        fig_wdl = go.Figure(
            data=[
                go.Pie(
                    labels=["Wins", "Draws", "Losses"],
                    values=[
                        team_stats["wins"],
                        team_stats["draws"],
                        team_stats["losses"],
                    ],
                    hole=0.3,
                    marker_colors=["#2ca02c", "#ff7f0e", "#d62728"],
                )
            ]
        )
        fig_wdl.update_layout(title=f"{selected_team} - Match Results Distribution")
        st.plotly_chart(fig_wdl, use_container_width=True)

        # Recent form display
        st.markdown(
            f"### 📅 Recent Form (Last 5 Matches): **{team_stats['recent_form']}**"
        )
        st.caption("W = Win, D = Draw, L = Loss")

        # Goals analysis
        goals_data = pd.DataFrame(
            {
                "Category": ["Scored", "Conceded"],
                "Goals": [team_stats["goals_scored"], team_stats["goals_conceded"]],
            }
        )
        fig_goals = px.bar(
            goals_data,
            x="Category",
            y="Goals",
            title=f"{selected_team} - Goals Analysis",
            color="Category",
            color_discrete_map={"Scored": "#2ca02c", "Conceded": "#d62728"},
        )
        st.plotly_chart(fig_goals, use_container_width=True)
    else:
        st.warning("No data available for this team.")

with tab3:
    st.subheader("📺 Today's Live Fixtures")
    st.caption("Real-time predictions for today's matches")

    try:
        today_matches = get_today_matches()
        if today_matches:
            for h, a in today_matches:
                try:
                    pred, probs = predict_match(h, a, df, model, le)

                    with st.container():
                        match_col1, match_col2, match_col3 = st.columns([2, 1, 2])

                        with match_col1:
                            st.write(f"**{h}**")
                        with match_col2:
                            st.write("🆚")
                        with match_col3:
                            st.write(f"**{a}**")

                        st.write(
                            f"Prediction: **{pred}** (Confidence: {max(probs) * 100:.1f}%)"
                        )
                        st.progress(max(probs))
                        st.markdown("---")
                except:
                    st.write(f"⚠️ {h} vs {a}: Cannot predict (teams not in database)")
                    st.markdown("---")
        else:
            st.info("No matches scheduled for today.")
    except Exception as e:
        st.error(f"Unable to fetch live fixtures: {e}")

with tab4:
    st.subheader("🏆 League Standings")
    st.caption("Generated from historical match data")

    try:
        standings = get_league_table(df)

        # Display summary stats
        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            leader = standings.index[0]
            leader_pts = standings.iloc[0]["points"]
            st.metric("League Leader", leader, f"{leader_pts} pts")

        with summary_col2:
            top_scorer_team = standings.nlargest(1, "goals_for").index[0]
            top_goals = standings.nlargest(1, "goals_for").iloc[0]["goals_for"]
            st.metric("Top Scoring Team", top_scorer_team, f"{int(top_goals)} goals")

        with summary_col3:
            best_def_team = standings.nsmallest(1, "goals_against").index[0]
            fewest_conceded = standings.nsmallest(1, "goals_against").iloc[0][
                "goals_against"
            ]
            st.metric("Best Defense", best_def_team, f"{int(fewest_conceded)} conceded")

        # Display full table
        st.markdown("### 📋 Full League Table")

        # Format the standings for display
        display_standings = standings.copy()
        display_standings.index.name = "Team"
        display_standings = display_standings.reset_index()
        display_standings.columns = [
            "Team",
            "Pos",
            "Played",
            "W",
            "D",
            "L",
            "GF",
            "GA",
            "GD",
            "Pts",
        ]

        # Style the dataframe
        st.dataframe(
            display_standings,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pos": st.column_config.NumberColumn("Pos", format="%d"),
                "Pts": st.column_config.NumberColumn("Pts", format="%d"),
                "GD": st.column_config.NumberColumn("GD", format="%+d"),
            },
        )

        # Visualizations
        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            # Top 10 teams by points
            top_10 = standings.head(10).reset_index()
            fig_points = px.bar(
                top_10,
                x="points",
                y=top_10.index,
                orientation="h",
                title="Top 10 Teams by Points",
                labels={"points": "Points", "index": "Team"},
                color="points",
                color_continuous_scale="Blues",
            )
            fig_points.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_points, use_container_width=True)

        with col_viz2:
            # Goal difference distribution
            fig_gd = px.scatter(
                standings.reset_index(),
                x="goals_for",
                y="goals_against",
                size="points",
                hover_data=["index"],
                title="Goals Scored vs Conceded",
                labels={"goals_for": "Goals Scored", "goals_against": "Goals Conceded"},
                color="goal_diff",
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig_gd, use_container_width=True)

    except Exception as e:
        st.error(f"Unable to generate league table: {e}")

# Footer
st.markdown("---")
st.caption(
    "⚽ Football Predictor Pro | Powered by XGBoost AI | Data-driven predictions"
)
