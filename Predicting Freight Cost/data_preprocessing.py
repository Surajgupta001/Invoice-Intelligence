import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd


def load_vendor_invoice_data(db_path: str):
    """
    Load vendor invoice data from the SQLite database.

    Parameters:
    db_path (str): Path to the SQLite database file.

    Returns:
    pd.DataFrame: DataFrame containing the vendor invoice data.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def prepare_feature(df: pd.DataFrame):
    """
    Prepare features and target variable for model training.

    Parameters:
    df (pd.DataFrame): DataFrame containing the vendor invoice data.

    Returns:
    tuple: Features (X) and target variable (y).
    """
    X = df[["Dollars"]]
    y = df["Freight"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: Training and testing sets (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
