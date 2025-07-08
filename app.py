import streamlit as st
import pandas as pd
import joblib

from model.predictor import predict_match
from scraper.live_scrapper import get_today_matches
from utils.features import create_features

# Load data and apply feature engineering
df = pd.read_csv("data/matches.csv")
df, _ = create_features(df)  # apply home/away form

# Load trained model and encoder
model = joblib.load("model/match_predictor.pkl")
le = joblib.load("model/label_encoder.pkl")

# Streamlit UI
st.title("⚽ Football Predictor + Live Stats")

home = st.selectbox("Home Team", sorted(df.HomeTeam.unique()))
away = st.selectbox("Away Team", sorted(df.HomeTeam.unique()))

if st.button("Predict"):
    if home == away:
        st.error("Pick different teams!")
    else:
        try:
            prediction = predict_match(home, away, df, model, le)
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Live match predictions
st.subheader("📺 Today's Live Fixtures")
for h, a in get_today_matches():
    try:
        pred = predict_match(h, a, df, model, le)
        st.write(f"{h} vs {a}: {pred}")
    except:
        st.warning(f"Skipped unknown match: {h} vs {a}")
