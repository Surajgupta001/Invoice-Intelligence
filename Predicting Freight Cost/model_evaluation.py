from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=5):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, max_depth=6):
    model = RandomForestRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate the model's performance on the test set.

    Parameters:
    model: Trained model to evaluate.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): True target values for the test set.
    model_name (str): Name of the model for display purposes.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred) * 100

    print(f"{model_name} Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.4f}\n")

    return {"model_name": model_name, "mae": mae, "rmse": rmse, "r2": r2}
