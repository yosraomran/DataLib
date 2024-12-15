import pytest
import pandas as pd
import os
from io import StringIO
from datalib.data_loader import *

def test_load_data():
    # Créer un fichier CSV temporaire
    csv_data = """col1,col2,col3
    1,2,3
    4,5,6
    7,8,9"""
    
    file_path = "temp_test_file.csv"
    with open(file_path, "w") as f:
        f.write(csv_data)

    # Charger les données avec la fonction
    df = load_data(file_path)

    # Assertions
    assert df is not None
    assert df.shape == (3, 3)
    assert list(df.columns) == ["col1", "col2", "col3"]

    # Nettoyer
    os.remove(file_path)

def test_save_data():
    # Créer un DataFrame de test
    df = pd.DataFrame({"col1": [1, 4, 7], "col2": [2, 5, 8], "col3": [3, 6, 9]})
    file_path = "temp_test_output.csv"

    # Sauvegarder les données avec la fonction
    save_data(df, file_path)

    # Lire les données sauvegardées
    saved_df = pd.read_csv(file_path)

    # Assertions
    assert saved_df.equals(df)

    # Nettoyer
    os.remove(file_path)

def test_filter_data():
    # Créer un DataFrame de test
    df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]})

    # Appliquer le filtre
    filtered_df = filter_data(df, "col1", lambda x: x > 2)

    # Assertions
    assert filtered_df.shape == (2, 2)
    assert (filtered_df["col1"] > 2).all()

def test_handle_missing_values():
    # Créer un DataFrame avec des valeurs manquantes
    df = pd.DataFrame({"col1": [1, None, 3], "col2": [None, 5, 6]})

    # Test avec la stratégie 'mean'
    df_mean = handle_missing_values(df, strategy="mean")
    assert not df_mean.isnull().values.any()
    assert df_mean.loc[1, "col1"] == 2

    # Test avec la stratégie 'median'
    df_median = handle_missing_values(df, strategy="median")
    assert not df_median.isnull().values.any()
    assert df_median.loc[1, "col1"] == 2

    # Test avec la stratégie 'drop'
    df_dropped = handle_missing_values(df, strategy="drop")
    assert df_dropped.shape == (1, 2)

def test_normalize_data():
    # Créer un DataFrame de test
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # Normaliser les données
    normalized_df = normalize_data(df)

    # Assertions
    assert normalized_df.shape == df.shape
    assert pytest.approx(normalized_df["col1"].mean(), 0.01) == 0
    assert pytest.approx(normalized_df["col2"].std(), 0.01) == 1

