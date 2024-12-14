import pandas as pd

def load_csv(file_path):
    """Charge un fichier CSV dans un DataFrame."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

def save_csv(data, file_path):
    """Sauvegarde un DataFrame dans un fichier CSV."""
    try:
        data.to_csv(file_path, index=False)
        print(f"Data saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        raise

def filter_data(data, column_name, condition):
    """Filtre les données selon une condition."""
    try:
        filtered_data = data.query(f"{column_name}{condition}")
        return filtered_data
    except Exception as e:
        print(f"Error filtering data: {e}")
        raise
