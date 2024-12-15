import pytest
import pandas as pd
import numpy as np
from datalib.data_stats import compute_mean, compute_median, compute_mode, compute_std, compute_correlation_matrix, t_test, chi_square_test

def test_compute_mean():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = compute_mean(df)
    expected = pd.Series({"A": 2.0, "B": 5.0})
    pd.testing.assert_series_equal(result, expected)

def test_compute_median():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = compute_median(df)
    expected = pd.Series({"A": 2.0, "B": 5.0})
    pd.testing.assert_series_equal(result, expected)

def test_compute_mode():
    df = pd.DataFrame({"A": [1, 2, 2, 3], "B": [4, 5, 6, 6]})
    result = compute_mode(df)
    expected = pd.DataFrame({"A": [2], "B": [6]})
    pd.testing.assert_frame_equal(result, expected)

def test_compute_std():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = compute_std(df)
    expected = pd.Series({"A": np.std([1, 2, 3], ddof=1), "B": np.std([4, 5, 6], ddof=1)})
    pd.testing.assert_series_equal(result, expected)

def test_compute_correlation_matrix():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = compute_correlation_matrix(df)
    expected = df.corr()
    pd.testing.assert_frame_equal(result, expected)

def test_t_test():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    t_stat, p_value = t_test(df, "A", "B")
    assert isinstance(t_stat, float)
    assert isinstance(p_value, float)

def test_chi_square_test():
    df = pd.DataFrame({"A": ["yes", "no", "yes", "no"], "B": ["cat", "dog", "cat", "dog"]})
    chi2_stat, p_value = chi_square_test(df, "A", "B")
    assert isinstance(chi2_stat, float)
    assert isinstance(p_value, float)

