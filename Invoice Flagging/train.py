from pathlib import Path

import joblib
from data_preprocessing import (
    apply_labels,
    load_invoice_data,
    scale_features,
    split_data,
)
from model_evaluation import evaluate_classifier, train_random_forest

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

TARGET = "flag_invoice"


def main():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_invoice_data()

    if df is None or df.empty:
        print("No data found in the database. Exiting.")
        return

    # Apply labels
    df = apply_labels(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, model_dir / "scaler.pkl"
    )

    # Train and evaluate the Random Forest model
    rf_model = train_random_forest(X_train_scaled, y_train)
    evaluate_classifier(rf_model, X_test_scaled, y_test, "Random Forest Classifier")

    # Save the trained model
    joblib.dump(rf_model, model_dir / "invoice_flagging_model.pkl")


if __name__ == "__main__":
    main()
