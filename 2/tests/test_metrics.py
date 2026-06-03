"""RED tests for benchmarks/metrics.py — acceptance metrics."""
import math

import numpy as np

from metrics import angular_error_stats, classification_accuracy


def test_zero_error_when_identical():
    s = angular_error_stats([1, 2, 3], [1, 2, 3])
    assert s["mae"] == 0.0
    assert s["rmse"] == 0.0
    assert s["max"] == 0.0


def test_simple_constant_error():
    s = angular_error_stats([10, 10, 10], [0, 0, 0])
    assert abs(s["mae"] - 10.0) < 1e-9
    assert abs(s["rmse"] - 10.0) < 1e-9
    assert abs(s["max"] - 10.0) < 1e-9


def test_angular_error_wraps_around_180():
    # 179 vs -179 should be 2 degrees apart, not 358.
    s = angular_error_stats([179.0], [-179.0])
    assert abs(s["mae"] - 2.0) < 1e-6


def test_rmse_penalises_outliers_more_than_mae():
    s = angular_error_stats([0, 0, 30], [0, 0, 0])
    assert s["rmse"] > s["mae"]                # one big error inflates RMSE


def test_classification_accuracy_all_correct():
    acc, conf = classification_accuracy(["indoor", "outdoor"], ["indoor", "outdoor"])
    assert acc == 1.0


def test_classification_accuracy_half():
    acc, conf = classification_accuracy(
        ["indoor", "indoor"], ["indoor", "outdoor"])
    assert abs(acc - 0.5) < 1e-9
    # confusion key is (predicted, truth): predicted indoor when truth was outdoor
    assert conf[("indoor", "outdoor")] == 1


def test_classification_accuracy_empty_is_zero():
    acc, conf = classification_accuracy([], [])
    assert acc == 0.0
