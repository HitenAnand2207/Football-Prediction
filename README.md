# ⚽ Football Match Prediction

A machine learning-based web application to predict football match outcomes using key statistics like possession, shots on target, cards, recent form, and player data.

## 🚀 Features

- Predicts match results using a trained classification model.
- Streamlit web interface for easy use.
- Takes match statistics as input (shots, possession, fouls, etc.).
- Displays predicted winner instantly.
- Backend powered by a trained Scikit-learn model.

## 🛠️ Tech Stack

**Languages & Frameworks:**
- Python
- Streamlit

**Libraries:**
- Scikit-learn
- Pandas
- NumPy
- Joblib

**Tools:**
- VS Code
- Git & GitHub

## 📊 Input Parameters

- Home Team Goals
- Away Team Goals
- Shots (On Target / Total)
- Possession (%)
- Yellow & Red Cards
- Recent Performance

## 🧠 Model Details

- Model: Random Forest Classifier (or your chosen model)
- Trained on historical match data with preprocessing.
- Accuracy: ~85% (based on test dataset)

## 🖥️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/HitenAnand2207/Football-Prediction.git
cd Football-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
