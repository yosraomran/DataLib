# DataLib
The goal of **DataLib** is to provide an easy-to-install, intuitive Python package for data manipulation, statistics, and machine learning tasks. With a focus on simplicity, it ensures that users can quickly perform data analysis and visualize results. The project is intended for both beginners and advanced users who want a lightweight and powerful data analysis tool.
## Features

### Data Loader
- CSV file loading and saving
- Data filtering
- Missing value handling
- Data normalization

### Data Statistics
- Descriptive statistics
- Correlation matrix
- T-tests
- Chi-square tests

### Data Visualization
- Bar plots
- Histograms
- Scatter plots
- Correlation heatmaps

### Models Machine Learning
- Linear and Polynomial Regression
- Classification Algorithms (KNN, Decision Trees)
- Clustering (K-means)
- Dimensionality Reduction (PCA)

## Project Structure
Source Code: The code is organized in a modular format (src/ or equivalent), allowing for easy maintenance and scalability.
Dependencies: It requires essential libraries like numpy, pandas, matplotlib, and scikit-learn. All dependencies are defined in setup.py, pyproject.toml, or setup.cfg.
Documentation:
A detailed README.md file is included, providing a comprehensive guide on usage and features.
Examples of how to implement and use the library are also provided.
Technical documentation is generated using Sphinx.
Tests:
Unit tests are written for core functions using pytest.
CI/CD workflows are integrated (e.g., using GitHub Actions) to validate changes and maintain code quality(build ans deploy).

## Installation

```bash
pip install datalib
```
## Installation from PyPi
```
pip install -i https://test.pypi.org/simple/ datalib==0.1.1
```
## How to Use
Once installed, you can start using DataLib in your Python scripts. Here are a few examples of what you can do:
```
import datalib as dl

# Load data from CSV
data = dl.load_csv('data.csv')

# Handle missing values
data = dl.handle_missing_values(data)

# Normalize data
normalized_data = dl.normalize(data)

# Generate descriptive statistics
stats = dl.calculate_statistics(data)

# Create a scatter plot
dl.plot_scatter(data)

```
## License

This project is licensed under the MIT License.
