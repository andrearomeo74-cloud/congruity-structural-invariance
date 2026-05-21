import os
import json
import pandas as pd


def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    """
    Save object as JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    """
    Load JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_table(df, path):
    """
    Save dataframe as CSV.
    """
    df.to_csv(path, index=False)


def print_table(df):
    """
    Print dataframe without index.
    """
    print(df.to_string(index=False))
