"""Evaluation metrics for count models and market probabilities."""

from __future__ import annotations

import numpy
import pandas
from sklearn.metrics import accuracy_score, log_loss, mean_poisson_deviance


def count_metrics(observed_counts, predicted_means):
    """MAE, RMSE, and Poisson deviance for count predictions."""
    observed_counts = numpy.asarray(observed_counts, dtype=float)
    predicted_means = numpy.clip(numpy.asarray(predicted_means, dtype=float), 1e-9, None)
    return {
        "mae": float(numpy.abs(observed_counts - predicted_means).mean()),
        "rmse": float(numpy.sqrt(((observed_counts - predicted_means) ** 2).mean())),
        "poisson_deviance": float(mean_poisson_deviance(observed_counts, predicted_means)),
    }


def binary_market_metrics(predicted_probabilities, outcome_indicator):
    predicted_probabilities = numpy.asarray(predicted_probabilities, dtype=float)
    outcome_indicator = numpy.asarray(outcome_indicator, dtype=float)
    return {
        "rows": int(len(outcome_indicator)),
        "log_loss": float(log_loss(outcome_indicator, predicted_probabilities, labels=[0, 1])),
        "brier": float(((predicted_probabilities - outcome_indicator) ** 2).mean()),
        "mean_predicted": float(predicted_probabilities.mean()),
        "empirical_rate": float(outcome_indicator.mean()),
    }


def market_eval_table(markets_by_name):
    """One row of binary market metrics per market."""
    return pandas.DataFrame([
        {"market": market_name, **binary_market_metrics(predicted, observed)}
        for market_name, (predicted, observed) in markets_by_name.items()
    ])


def ranked_probability_score(class_probabilities, outcome_indices):
    """Mean RPS for ordered class probabilities."""
    class_probabilities = numpy.asarray(class_probabilities, dtype=float)
    outcome_indices = numpy.asarray(outcome_indices, dtype=int)
    outcome_matrix = numpy.zeros_like(class_probabilities)
    outcome_matrix[numpy.arange(len(outcome_indices)), outcome_indices] = 1.0
    cumulative_gap = (
        numpy.cumsum(class_probabilities, axis=1) - numpy.cumsum(outcome_matrix, axis=1)
    )
    return float(
        (cumulative_gap[:, :-1] ** 2).sum(axis=1).mean()
        / (class_probabilities.shape[1] - 1)
    )


def evaluate_multiclass_probabilities(outcome_labels, class_probabilities, labels=(0, 1, 2)):
    """Accuracy and log loss for multiclass probabilities."""
    class_probabilities = numpy.asarray(class_probabilities)
    return {
        "accuracy": accuracy_score(outcome_labels, class_probabilities.argmax(axis=1)),
        "log_loss": log_loss(outcome_labels, class_probabilities, labels=list(labels)),
    }

