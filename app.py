import streamlit as st, pandas as pd, joblib
from model.predictor import predict_match
from scraper.live_scrapper import get_today_matches

df = pd.read_csv("data/matches.csv")
model = joblib.load("model/match_predictor.pkl")
le = joblib.load("model/label_encoder.pkl")

st.title("⚽ Football Predictor + Live Stats")

home = st.selectbox("Home Team", sorted(df.HomeTeam.unique()))
away = st.selectbox("Away Team", sorted(df.HomeTeam.unique()))
if st.button("Predict"):
    if home == away:
        st.error("Pick different teams!")
    else:
        st.success(predict_match(home, away, df, model, le))

st.subheader("📺 Today's Live Fixtures")
for h, a in get_today_matches():
    try:
        pred = predict_match(h, a, df, model, le)
        st.write(f"{h} vs {a}: {pred}")
        # Optionally fetch players/stats using match URL
        # stats, scorers = get_player_stats_and_scorers(url)
        # st.write("Scorers:", scorers)
    except:
        st.warning(f"Skips unknown match: {h} vs {a}")
