from pathlib import Path

import joblib
from data_preprocessing import load_vendor_invoice_data, prepare_feature, split_data
from model_evaluation import (
    evaluate_model,
    train_decision_tree,
    train_linear_regression,
    train_random_forest,
)


def main():
    db_path = "../data/inventory.db"
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(db_path)

    if df is None or df.empty:
        print("No data found in the database. Exiting.")
        return

    # Prepare features and split
    X, y = prepare_feature(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree Regressor"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest Regressor"))

    # Select best model (lowest MAE)
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regressor": dt_model,
        "Random Forest Regressor": rf_model,
    }[best_model_name]

    # Save best model
    model_path = model_dir / "predict_freight_cost_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"Best model '{best_model_name}' saved to {model_path}")


if __name__ == "__main__":
    main()
