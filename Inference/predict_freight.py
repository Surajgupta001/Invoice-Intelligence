from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "predict_freight_cost_model.pkl"
)


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


def predict_freight_cost(input_data):
    """
    Predict the freight cost based on the input data.

    Parameters:
    input_data (dict): Dictionary containing the input features.

    Returns:
    float: Predicted freight cost.
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df["Predicted_Freight"] = model.predict(input_df).round()
    return input_df["Predicted_Freight"].tolist()


if __name__ == "__main__":
    # Example input data
    sample_data = {"Dollars": [18500, 9000, 3000, 200]}

    prediction = predict_freight_cost(sample_data)
    print("Predicted Freight Costs:")
    print(prediction)
