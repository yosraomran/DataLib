from datalib.data_loader import load_csv
from datalib.data_stats import compute_mean, compute_std
from datalib.ml_models import train_regression_model
from datalib.data_viz import plot_data

data = load_csv('dataset/dataset_traffic_accident.csv')
print("Mean:", compute_mean(data, column="age"))
print("Standard Deviation:", compute_std(data, column="age"))

X, y = data[["feature1", "feature2"]], data["target"]

# Train a regression model
model = train_regression_model(X, y)

# Visualize the results
plot_data(data, x="feature1", y="target", model=model)

