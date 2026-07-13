"""Shared target derivation for football match data."""

from __future__ import annotations

import numpy

from src.football_prediction.core import config


def add_match_targets(match_frame):
    """Add modelling targets and shared derived columns."""
    matches = match_frame.copy()

    matches["home_win"] = (matches["home_score_ft"] > matches["away_score_ft"]).astype(int)
    matches["away_win"] = (matches["away_score_ft"] > matches["home_score_ft"]).astype(int)
    matches["draw"] = matches["home_score_ft"].eq(matches["away_score_ft"]).astype(int)
    matches["match_result"] = numpy.where(
        matches["home_win"].eq(1),
        0,
        numpy.where(matches["away_win"].eq(1), 2, 1),
    ).astype(int)

    matches["home_score_2h"] = matches["home_score_ft"] - matches["home_score_ht"]
    matches["away_score_2h"] = matches["away_score_ft"] - matches["away_score_ht"]
    matches["total_goals"] = matches["home_score_ft"] + matches["away_score_ft"]
    matches["first_half_total_goals"] = matches["home_score_ht"] + matches["away_score_ht"]

    matches["home_wins_either_half"] = (
        (matches["home_score_ht"] > matches["away_score_ht"])
        | (matches["home_score_2h"] > matches["away_score_2h"])
    ).astype(int)
    matches["away_wins_either_half"] = (
        (matches["away_score_ht"] > matches["home_score_ht"])
        | (matches["away_score_2h"] > matches["home_score_2h"])
    ).astype(int)

    matches["home_points"] = numpy.where(
        matches["home_win"].eq(1),
        3,
        numpy.where(matches["away_win"].eq(1), 0, 1),
    )
    matches["away_points"] = numpy.where(
        matches["away_win"].eq(1),
        3,
        numpy.where(matches["home_win"].eq(1), 0, 1),
    )

    matches["is_european_cup"] = matches["competition"].isin(
        config.EUROPEAN_CUP_NAMES
    ).astype(int)
    return matches

