"""Platt-style probability calibration helpers."""

from __future__ import annotations

import numpy
import pandas
from sklearn.linear_model import LogisticRegression

from src.football_prediction.core import config


def log_probability_features(class_probabilities):
    """Multiclass Platt features from class probabilities."""
    return numpy.log(numpy.clip(numpy.asarray(class_probabilities), 1e-6, 1.0))


def logit_feature(positive_probabilities):
    """Binary Platt feature from positive-class probabilities."""
    clipped = numpy.clip(numpy.asarray(positive_probabilities), 1e-6, 1.0 - 1e-6)
    return numpy.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_multiclass_platt(raw_probabilities, outcome_labels, random_seed=config.RANDOM_SEED):
    calibrator = LogisticRegression(max_iter=1000, random_state=random_seed)
    calibrator.fit(log_probability_features(raw_probabilities), outcome_labels)
    return calibrator


def apply_multiclass_platt(raw_probabilities, calibrator):
    return calibrator.predict_proba(log_probability_features(raw_probabilities))


def fit_binary_platt(raw_positive_probabilities, outcome_indicator, random_seed=config.RANDOM_SEED):
    calibrator = LogisticRegression(max_iter=1000, random_state=random_seed)
    calibrator.fit(logit_feature(raw_positive_probabilities), outcome_indicator)
    return calibrator


def apply_binary_calibration(raw_positive_probabilities, calibrator):
    return calibrator.predict_proba(logit_feature(raw_positive_probabilities))[:, 1]


def reliability_table(predicted_probabilities, outcome_indicator, number_of_bins=10, minimum_bin_size=25):
    """Mean predicted probability vs observed rate by probability bin."""
    predicted_probabilities = numpy.asarray(predicted_probabilities, dtype=float)
    outcome_indicator = numpy.asarray(outcome_indicator, dtype=float)
    bin_edges = numpy.linspace(0.0, 1.0, number_of_bins + 1)
    bin_ids = numpy.clip(numpy.digitize(predicted_probabilities, bin_edges) - 1, 0, number_of_bins - 1)
    bin_records = []
    for bin_id in range(number_of_bins):
        in_bin = bin_ids == bin_id
        if in_bin.sum() >= minimum_bin_size:
            bin_records.append({
                "mean_predicted": float(predicted_probabilities[in_bin].mean()),
                "observed_rate": float(outcome_indicator[in_bin].mean()),
                "matches": int(in_bin.sum()),
            })
    return pandas.DataFrame(bin_records)

