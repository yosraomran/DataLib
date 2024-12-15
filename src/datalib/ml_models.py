from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def linear_regression(df, x_columns, y_column):
    """
    Applique une régression linéaire pour prédire une variable cible.
    
    :param df: Le DataFrame.
    :param x_columns: Les colonnes des variables indépendantes.
    :param y_column: La colonne de la variable dépendante.
    :return: Le modèle de régression et les prédictions.
    """
    X = df[x_columns]
    y = df[y_column]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    return model, predictions

def polynomial_regression(df, x_columns, y_column, degree=2):
    """
    Applique une régression polynomiale.
    
    :param df: Le DataFrame.
    :param x_columns: Les colonnes des variables indépendantes.
    :param y_column: La colonne de la variable dépendante.
    :param degree: Le degré du polynôme.
    :return: Le modèle de régression et les prédictions.
    """
    X = df[x_columns]
    y = df[y_column]
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    predictions = model.predict(X_poly)
    return model, predictions

def knn_classification(df, x_columns, y_column, n_neighbors=3):
    """
    Applique un algorithme de classification k-NN.
    
    :param df: Le DataFrame.
    :param x_columns: Les colonnes des variables indépendantes.
    :param y_column: La colonne de la variable cible.
    :param n_neighbors: Le nombre de voisins à considérer.
    :return: Le modèle k-NN et les prédictions.
    """
    X = df[x_columns]
    y = df[y_column]
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)
    predictions = model.predict(X)
    return model, predictions

def decision_tree_classification(df, x_columns, y_column):
    """
    Applique un algorithme d'arbre de décision pour la classification.
    
    :param df: Le DataFrame.
    :param x_columns: Les colonnes des variables indépendantes.
    :param y_column: La colonne de la variable cible.
    :return: Le modèle d'arbre de décision et les prédictions.
    """
    X = df[x_columns]
    y = df[y_column]
    model = DecisionTreeClassifier()
    model.fit(X, y)
    predictions = model.predict(X)
    return model, predictions

def kmeans_clustering(df, x_columns, n_clusters=3):
    """
    Applique l'algorithme de clustering k-means.
    
    :param df: Le DataFrame.
    :param x_columns: Les colonnes des variables indépendantes.
    :param n_clusters: Le nombre de clusters.
    :return: Les labels des clusters et les centres des clusters.
    """
    X = df[x_columns]
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)
    return model.labels_, model.cluster_centers_

def pca_analysis(df, x_columns, n_components=2):
    """
    Applique une analyse en composantes principales (PCA).
    
    :param df: Le DataFrame.
    :param x_columns: Les colonnes des variables indépendantes.
    :param n_components: Le nombre de composants principaux.
    :return: La transformation PCA des données.
    """
    X = df[x_columns]
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)
