import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar(df, column):
    """
    Génère un graphique en barres pour une colonne.
    
    :param df: Le DataFrame.
    :param column: Le nom de la colonne.
    """
    df[column].value_counts().plot(kind='bar')
    plt.title(f'Bar Chart for {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

def plot_histogram(df, column, bins=10):
    """
    Génère un histogramme pour une colonne.
    
    :param df: Le DataFrame.
    :param column: Le nom de la colonne.
    :param bins: Le nombre de bins pour l'histogramme.
    """
    df[column].hist(bins=bins)
    plt.title(f'Histogram for {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_scatter(df, column1, column2):
    """
    Génère un graphique en nuage de points entre deux colonnes.
    
    :param df: Le DataFrame.
    :param column1: Le nom de la première colonne.
    :param column2: Le nom de la deuxième colonne.
    """
    plt.scatter(df[column1], df[column2])
    plt.title(f'Scatter Plot: {column1} vs {column2}')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()

def plot_correlation_matrix(df):
    """
    Génère une matrice de corrélation sous forme de heatmap.
    
    :param df: Le DataFrame.
    """
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
