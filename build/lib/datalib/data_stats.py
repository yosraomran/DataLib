import numpy as np
import scipy.stats as stats
import pandas as pd

def compute_mean(df):
    """
    Calcule la moyenne de toutes les colonnes numériques du DataFrame.
    
    Parameters:
    - df : pd.DataFrame : DataFrame contenant les données numériques.
    
    Returns:
    - pd.Series : Moyenne de chaque colonne numérique du DataFrame.
    """
    return df.mean()

def compute_median(df):
    """
    Calcule la médiane de toutes les colonnes numériques du DataFrame.
    
    Parameters:
    - df : pd.DataFrame : DataFrame contenant les données numériques.
    
    Returns:
    - pd.Series : Médiane de chaque colonne numérique du DataFrame.
    """
    return df.median()

def compute_mode(df):
    """
    Calcule le mode de toutes les colonnes du DataFrame.
    
    Parameters:
    - df : pd.DataFrame : DataFrame contenant les données.
    
    Returns:
    - pd.DataFrame : Mode de chaque colonne du DataFrame. Si plusieurs modes existent, 
                      tous les modes seront renvoyés.
    """
    return df.mode()

def compute_std(df):
    """
    Calcule l'écart-type de toutes les colonnes numériques du DataFrame.
    
    Parameters:
    - df : pd.DataFrame : DataFrame contenant les données numériques.
    
    Returns:
    - pd.Series : Écart-type de chaque colonne numérique du DataFrame.
    """
    return df.std()

def compute_correlation_matrix(df):
    """
    Calcule la matrice de corrélation entre toutes les colonnes numériques du DataFrame.
    
    Parameters:
    - df : pd.DataFrame : DataFrame contenant les données numériques.
    
    Returns:
    - pd.DataFrame : Matrice de corrélation entre les colonnes numériques du DataFrame.
    """
    return df.corr()
def t_test(df, column1, column2):
    """
    Effectue un test t pour comparer les moyennes de deux colonnes.
    
    Parameters:
    - df : pd.DataFrame : DataFrame contenant les colonnes.
    - column1, column2 : str : noms des colonnes à comparer.
    
    Returns:
    - tuple : statistique t, p-value
    """
    t_stat, p_value = stats.ttest_ind(df[column1].dropna(), df[column2].dropna())
    return t_stat, p_value

def chi_square_test(df, column1, column2):
    """
    Effectue un test du chi-carré pour deux colonnes catégorielles.
    
    Parameters:
    - df : pd.DataFrame : DataFrame contenant les colonnes.
    - column1, column2 : str : noms des colonnes à tester.
    
    Returns:
    - tuple : statistique chi-carré, p-value
    """
    contingency_table = pd.crosstab(df[column1], df[column2])
    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
    return chi2_stat, p_value