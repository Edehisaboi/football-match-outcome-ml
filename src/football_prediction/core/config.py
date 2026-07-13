"""Shared constants for every model-group notebook."""

from __future__ import annotations

from pathlib import Path

import pandas

# This file lives at src/football_prediction/core/config.py, so the project
# root is three levels above the package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "datasets" / "rich_stats" / "league_matches_stats.csv"
MODELS_DIRECTORY = PROJECT_ROOT / "models"

# Rolling-history windows.
SHORT_WINDOW = 5
LONG_WINDOW = 10
LONG_WINDOW_MINIMUM_MATCHES = 5
XG_WINDOW = 5
XG_WINDOW_MINIMUM_MATCHES = 3
MINIMUM_TEAM_MATCHES = 10
MAXIMUM_REST_DAYS = 30.0

# Elo configuration.
INITIAL_ELO = 1500.0
ELO_K = 20.0
HOME_ADVANTAGE = 65.0

# Temporal splits. Train on the past, tune on validation, and reserve test for
# the final read-out.
VALIDATION_SPLIT_DATE = pandas.Timestamp("2025-07-01")
TEST_SPLIT_DATE = pandas.Timestamp("2026-01-01")

RANDOM_SEED = 42

EUROPEAN_CUP_NAMES = [
    "UEFA Champions League",
    "UEFA Europa League",
    "UEFA Conference League",
]

MATCH_RESULT_LABELS = {
    0: "home_win",
    1: "draw",
    2: "away_win",
}

