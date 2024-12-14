# usage_example.py

import pandas as pd
from datalib.data_loader import load_data, save_data, filter_data
from datalib.data_stats import compute_mean, compute_median, compute_mode, compute_std, compute_correlation_matrix
from datalib.data_viz import plot_histogram, plot_scatter, plot_correlation_matrix
from datalib.ml_models import train_linear_regression, train_kmeans

# Charger les données
file_path = 'dataset_traffic_accident.csv'
df = load_data(file_path)

# Vérification du DataFrame chargé
print("Données chargées :")
print(df.head())

# Sauvegarder les données dans un autre fichier
output_path = 'output_traffic_accident.csv'
save_data(df, output_path)

# Filtrage des données : par exemple, ne conserver que les lignes où la colonne 'accident_severity' > 2
filtered_df = filter_data(df, 'accident_severity', lambda x: x > 2)
print("\nDonnées filtrées :")
print(filtered_df.head())

# Calculs statistiques : Moyenne, Médiane, Mode, Écart-type pour toutes les colonnes numériques
print("\nStatistiques de base :")
print("Moyenne :\n", compute_mean(df))
print("Médiane :\n", compute_median(df))
print("Mode :\n", compute_mode(df))
print("Écart-type :\n", compute_std(df))

# Calcul de la matrice de corrélation
print("\nMatrice de corrélation :")
correlation_matrix = compute_correlation_matrix(df)
print(correlation_matrix)

# Visualisation des données : Histogramme pour la colonne 'accident_severity'
plot_histogram(df, 'accident_severity')

# Scatter plot entre 'latitude' et 'longitude' pour observer la distribution géographique des accidents
plot_scatter(df, 'latitude', 'longitude')

# Matrice de corrélation sous forme de graphique
plot_correlation_matrix(df)

# Application de modèles de machine learning :
# 1. Régression linéaire : prédire 'accident_severity' en fonction de 'temperature'
X = df[['temperature']]  # Variables explicatives
y = df['accident_severity']  # Variable cible
linear_model = train_linear_regression(X, y)
print("\nCoefficients de la régression linéaire :")
print(linear_model.coef_)

# 2. Clustering avec K-means : regrouper les données en 3 clusters en fonction de 'latitude' et 'longitude'
kmeans_model = train_kmeans(df[['latitude', 'longitude']], n_clusters=3)
print("\nCentres des clusters K-means :")
print(kmeans_model.cluster_centers_)
