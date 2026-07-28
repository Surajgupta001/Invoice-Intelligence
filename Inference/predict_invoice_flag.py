from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "invoice_flagging_model.pkl"
)
SCALER_PATH = Path(__file__).resolve().parents[1] / "models" / "scaler.pkl"
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


def load_model(model_path: str | Path = MODEL_PATH):
    """
    Load the trained model from the specified path.

    Parameters:
    model_path (str): Path to the saved model file.

    Returns:
    model: Loaded model object.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def load_scaler(scaler_path: str | Path = SCALER_PATH):
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler


def predict_invoice_flag(input_data):
    """
    Predict the invoice flag based on the input data.

    Parameters:
    input_data (dict): Dictionary containing the input features.

    Returns:
    pd.DataFrame: DataFrame containing the input data and predicted invoice flags.
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    missing_features = [
        feature for feature in FEATURES if feature not in input_df.columns
    ]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    scaler = load_scaler()
    scaled_input = scaler.transform(input_df[FEATURES])
    input_df["Predicted_Invoice_Flag"] = model.predict(scaled_input)
    return input_df


if __name__ == "__main__":
    # Example input data
    sample_data = {
        "invoice_quantity": [10, 5, 20],
        "invoice_dollars": [1000, 500, 2000],
        "Freight": [50, 25, 100],
        "total_item_quantity": [15, 10, 30],
        "total_item_dollars": [1500, 1000, 3000],
    }

    prediction = predict_invoice_flag(sample_data)
    print("Predicted Invoice Flags:")
    print(prediction)
