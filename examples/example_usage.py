import pandas as pd
import numpy as np
from datalib.data_loader import save_data,load_data,normalize_data,filter_data,handle_missing_values
from datalib.data_stats import *
from datalib.data_viz import *
from datalib.ml_models import *
# Création d'un DataFrame de test
data = {
    'Age': [25, 30, np.nan, 35, 40],
    'Salary': [50000, 60000, 55000, np.nan, 70000],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR']
}
df = pd.DataFrame(data)

# Afficher le DataFrame initial
print("DataFrame initial:")
print(df)

# Test des fonctions
from sklearn.preprocessing import StandardScaler

# 1. Gestion des valeurs manquantes
df_filled = handle_missing_values(df, strategy='mean')
print("\nDataFrame après gestion des valeurs manquantes (moyenne) :")
print(df_filled)

# 2. Normalisation des données numériques
df_normalized = normalize_data(df_filled.copy())
print("\nDataFrame après normalisation :")
print(df_normalized)

# 3. Filtrage des données (exemple : Age > 30)
df_filtered = filter_data(df_filled, 'Age', lambda x: x > 30)
print("\nDataFrame filtré (Age > 30) :")
print(df_filtered)

# 4. Statistiques descriptives
mean_values = compute_mean(df_filled)
print("\nMoyenne des colonnes numériques :")
print(mean_values)

# 5. Régression linéaire (prédire Salary à partir de Age)
from sklearn.linear_model import LinearRegression

df_clean = df_filled.dropna()  # Nettoyer les valeurs manquantes pour la régression
model, predictions = linear_regression(df_clean, ['Age'], 'Salary')
print("\nPrédictions de régression linéaire (Salary) :")
print(predictions)

# 6. Visualisation : Nuage de points entre Age et Salary
import matplotlib.pyplot as plt

plt.scatter(df_clean['Age'], df_clean['Salary'])
plt.plot(df_clean['Age'], predictions, color='red')  # Ligne de régression
plt.title('Régression linéaire : Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()
