import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

def compute_statistics(data):
    """Calcule des statistiques de base (moyenne, médiane, écart-type)."""
    stats = {
        "mean": data.mean(),
        "median": data.median(),
        "std_dev": data.std()
    }
    return stats

def perform_statistical_tests(data1, data2, test_type="t-test"):
    """Effectue des tests statistiques entre deux ensembles de données."""
    if test_type == "t-test":
        stat, p_value = ttest_ind(data1, data2)
        return {"statistic": stat, "p_value": p_value}
    elif test_type == "chi-squared":
        table = pd.crosstab(data1, data2)
        chi2, p_value, _, _ = chi2_contingency(table)
        return {"chi2": chi2, "p_value": p_value}
    else:
        raise ValueError("Unsupported test type. Choose 't-test' or 'chi-squared'.")
