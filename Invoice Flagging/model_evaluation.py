from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    make_scorer,
)


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"],
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_classifier(model, X_test, y_test, model_name):
    """
    Evaluate the classifier's performance on the test set.

    Parameters:
    model: Trained classifier to evaluate.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): True target values for the test set.
    model_name (str): Name of the model for display purposes.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)

    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_score": f1_score(y_test, pred),
    }
