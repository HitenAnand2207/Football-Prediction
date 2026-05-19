import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from utils.features import create_features, get_feature_columns
from model.predictor import train_model
from utils.analytics import (
    evaluate_model_performance,
    plot_feature_importance_plotly,
    plot_confusion_matrix_plotly,
    plot_calibration_curve_plotly,
    get_feature_importance,
)


def main():
    os.makedirs("reports/plots", exist_ok=True)

    df = pd.read_csv("data/matches.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"]) 
    df["FTR_encoded"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})

    # Create features
    df, le = create_features(df)

    features = get_feature_columns()
    X = df[features]
    y = df["FTR_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model (this may take a minute)...")
    model = train_model(df)

    print("Evaluating model...")
    perf = evaluate_model_performance(model, X_test, y_test)

    # Save plots
    fi_path = "reports/plots/feature_importance.html"
    cm_path = "reports/plots/confusion_matrix.html"

    print(f"Saving feature importance to {fi_path}")
    plot_feature_importance_plotly(model, features, save_path=fi_path)

    print(f"Saving confusion matrix to {cm_path}")
    plot_confusion_matrix_plotly(perf["confusion_matrix"], save_path=cm_path)

    # Calibration plot
    cal_path = "reports/plots/calibration_curve.html"
    print(f"Saving calibration curve to {cal_path}")
    plot_calibration_curve_plotly(model, X_test, y_test, save_path=cal_path)

    # Save classification report and feature importance
    import json

    cr_path = "reports/classification_report.json"
    with open(cr_path, "w", encoding="utf-8") as fh:
        json.dump(perf["classification_report"], fh, indent=2)

    fi_csv = "reports/plots/feature_importance.csv"
    fi_df = get_feature_importance(model, features)
    fi_df.to_csv(fi_csv, index=False)

    # Save model and encoder for later use
    joblib.dump(model, "model/match_predictor.pkl")
    joblib.dump(le, "model/label_encoder.pkl")

    print("Done. Interactive plots are in reports/plots/")


if __name__ == "__main__":
    main()
