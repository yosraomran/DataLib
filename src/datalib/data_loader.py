import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Charge un fichier CSV dans un DataFrame pandas.
    
    Parameters:
    - file_path : str : chemin du fichier CSV à charger.
    
    Returns:
    - pd.DataFrame : données chargées sous forme de DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        return None

def save_data(df, file_path):
    """
    Sauvegarde un DataFrame pandas dans un fichier CSV.
    
    Parameters:
    - df : pd.DataFrame : DataFrame à sauvegarder.
    - file_path : str : chemin du fichier de destination.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Données sauvegardées dans {file_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier : {e}")

def filter_data(df, column_name, condition):
    """
    Applique un filtre sur un DataFrame en fonction d'une condition.
    
    Parameters:
    - df : pd.DataFrame : DataFrame à filtrer.
    - column_name : str : nom de la colonne sur laquelle filtrer.
    - condition : fonction : condition de filtrage (par exemple, une valeur spécifique).
    
    Returns:
    - pd.DataFrame : DataFrame filtré.
    """
    return df[df[column_name].apply(condition)]

def handle_missing_values(df, strategy='mean'):
    """
    Gère les valeurs manquantes dans un DataFrame.
    
    Parameters:
    - df : pd.DataFrame : DataFrame avec des valeurs manquantes.
    - strategy : str : stratégie pour gérer les valeurs manquantes. Options : 'mean', 'median', 'drop'.
    
    Returns:
    - pd.DataFrame : DataFrame avec les valeurs manquantes gérées.
    """
    if strategy == 'mean':
        # Remplir les valeurs manquantes avec la moyenne de chaque colonne
        df_filled = df.fillna(df.mean())
    elif strategy == 'median':
        # Remplir les valeurs manquantes avec la médiane de chaque colonne
        df_filled = df.fillna(df.median())
    elif strategy == 'drop':
        # Supprimer les lignes avec des valeurs manquantes
        df_filled = df.dropna()
    else:
        raise ValueError("Stratégie non supportée. Choisissez parmi 'mean', 'median', 'drop'.")
    
    return df_filled

def normalize_data(df):
    """
    Normalise les données numériques du DataFrame.
    
    Parameters:
    - df : pd.DataFrame : DataFrame contenant les données à normaliser.
    
    Returns:
    - pd.DataFrame : DataFrame avec les données normalisées.
    """
    scaler = StandardScaler()
    
    # Applique la normalisation uniquement aux colonnes numériques
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    
    # Normalisation des données
    df_numeric_normalized = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
    
    # Remplacer les colonnes numériques par les colonnes normalisées
    df[df_numeric.columns] = df_numeric_normalized
    
    return df
