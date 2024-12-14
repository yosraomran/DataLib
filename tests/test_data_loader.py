import pytest
from src.datalib.data_loader import load_csv, save_csv, filter_data
import pandas as pd

def test_load_csv():
    df = load_csv("dataset/dataset_traffic_accident.csv")
    assert not df.empty

def test_save_csv(tmp_path):
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    save_path = tmp_path / "output.csv"
    save_csv(df, save_path)
    assert save_path.exists()

def test_filter_data():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 4, 5]})
    filtered = filter_data(df, "col1", ">1")
    assert len(filtered) == 2
