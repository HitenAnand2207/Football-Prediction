# ⚽ Football Match Prediction Pro

A comprehensive machine learning-based web application to predict football match outcomes using advanced statistics, team form, head-to-head history, and AI-powered analytics.

## 🚀 Features

### 🎯 Match Prediction
- **AI-Powered Predictions** using XGBoost classifier
- **Confidence Scores** with probability breakdown for all outcomes
- **Predicted Score** with expected goals (xG) analysis
- **Betting Odds** calculator (decimal format)
- **Interactive Visualizations** with confidence bars and charts

### 📊 Advanced Analytics
- **Team Statistics Dashboard** with comprehensive metrics:
  - Win/Draw/Loss ratios
  - Goals scored and conceded
  - Goal difference tracking
  - Recent form (last 5 matches)
  - Win rate percentages
- **Head-to-Head History** between teams
- **League Standings** with full table and visualizations
- **Performance Charts** including:
  - Win/Draw/Loss pie charts
  - Goals analysis bar charts
  - Points distribution charts
  - Goals scored vs conceded scatter plots

### 📺 Live Features
- **Today's Fixtures** with real-time predictions
- **Web Scraping** for live match data
- **Confidence Indicators** for each prediction

### 🔍 Model Insights
- **Feature Importance** analysis
- **Model Evaluation Metrics**:
  - Accuracy scores
  - Precision and recall by outcome
  - Confusion matrix
  - Classification reports

## 🛠️ Tech Stack

**Languages & Frameworks:**
- Python 3.x
- Streamlit (Web UI)

**Machine Learning:**
- XGBoost classifier
- Scikit-learn
- Pandas & NumPy

**Visualization:**
- Plotly (interactive charts)
- Matplotlib
- Seaborn

**Web Scraping:**
- BeautifulSoup4
- Requests

**Tools:**
- VS Code
- Git & GitHub
- Joblib (model persistence)

## 📊 Input Parameters

The model uses these features for predictions:
- **Team Encoding** (Home & Away teams)
- **Goal Difference** (historical average)
- **Team Form** (recent 3-match rolling average)
- **Historical Performance** metrics

## 🧠 Model Details

- **Algorithm:** XGBoost Multi-class Classifier
- **Classes:** Home Win, Draw, Away Win
- **Features:** 5 engineered features
- **Evaluation:** mlogloss metric
- **Output:** Probability distribution across all outcomes

## 🖥️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/HitenAnand2207/Football-Prediction.git
cd Football-Prediction

# Install dependencies
pip install -r requirements.txt

# Train the model (first time setup)
python main.py

# Run the Streamlit web app
streamlit run app.py

# Or run CLI predictions
python predict.py

# View league standings
python league_standings.py
```

## 📁 Project Structure

```
Football-prediction/
├── app.py                  # Streamlit web application
├── main.py                 # Model training script with evaluation
├── predict.py              # CLI prediction tool with detailed output
├── league_standings.py     # League table generator
├── requirements.txt        # Python dependencies
├── data/
│   └── matches.csv        # Historical match data
├── model/
│   ├── predictor.py       # Prediction functions & team stats
│   ├── match_predictor.pkl    # Trained model (generated)
│   └── label_encoder.pkl      # Label encoder (generated)
├── scraper/
│   ├── live_scrapper.py   # Live fixtures scraper
│   └── player_scraper.py  # Player data scraper
└── utils/
    ├── features.py        # Feature engineering
    └── analytics.py       # Advanced analytics & metrics
```

## 🎮 Usage

### Web Application
After running `streamlit run app.py`, navigate through the tabs:

1. **🎯 Match Predictor**
   - Select home and away teams
   - Get AI prediction with confidence scores
   - View betting odds and predicted score
   - See head-to-head history
   - Compare team statistics

2. **📊 Team Analytics**
   - Select any team for detailed analysis
   - View performance metrics
   - See win/draw/loss distribution
   - Check recent form and goals analysis

3. **📺 Live Fixtures**
   - View today's scheduled matches
   - Get instant predictions for all fixtures
   - See confidence levels for each match

4. **🏆 League Table**
   - Full standings with all statistics
   - Interactive visualizations
   - Top scorers and best defense
   - Goals analysis charts

### Command Line Tools

**Predict a specific match:**
```python
# Edit predict.py to change teams
python predict.py
```

**Train/retrain the model:**
```python
python main.py
```

**View league standings:**
```python
python league_standings.py
```

## 🆕 New Features Added

✨ **Prediction Confidence** - See probability percentages for all outcomes  
📈 **Head-to-Head Stats** - Historical matchup data between teams  
📊 **Team Analytics Dashboard** - Comprehensive statistics for every team  
💰 **Betting Odds Calculator** - Convert predictions to betting odds  
⚽ **Score Prediction** - Most likely score with xG (expected goals)  
🏆 **League Table** - Full standings with visualizations  
📉 **Model Evaluation** - Detailed performance metrics  
🎨 **Interactive Charts** - Beautiful Plotly visualizations  
🔍 **Feature Importance** - Understand what drives predictions  
📺 **Live Fixtures** - Real-time match predictions  

## 🎯 Model Performance

The model is evaluated using:
- **Accuracy Score** - Overall prediction accuracy
- **Precision & Recall** - Per-class performance metrics
- **Confusion Matrix** - Detailed prediction analysis
- **Feature Importance** - Impact of each feature

Run `python main.py` to see detailed performance metrics during training.

## 🔮 Future Enhancements

- [ ] Real-time match updates
- [ ] Player-level statistics integration
- [ ] Weather and venue factors
- [ ] Injury and suspension tracking
- [ ] Multiple model ensemble predictions
- [ ] Historical prediction tracking
- [ ] API endpoints for external integrations
- [ ] Mobile-responsive design improvements

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
