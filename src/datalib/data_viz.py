import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(data):
    """Affiche une matrice de corrélation."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def plot_histogram(data, column_name):
    """Affiche un histogramme pour une colonne spécifique."""
    plt.figure(figsize=(8, 6))
    plt.hist(data[column_name], bins=20, alpha=0.7)
    plt.title(f"Histogram of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.show()
