"""Dataset loading and filtering."""

from __future__ import annotations

import numpy
import pandas

from src.football_prediction.core import config
from src.football_prediction.data.targets import add_match_targets

REQUIRED_RAW_COLUMNS = [
    "match_id",
    "season",
    "country",
    "competition",
    "date_utc",
    "home_team",
    "away_team",
    "home_score_ft",
    "away_score_ft",
    "home_score_ht",
    "away_score_ht",
    "home_shots_on_target_ft",
    "away_shots_on_target_ft",
    "home_total_shots_ft",
    "away_total_shots_ft",
    "home_corner_kicks_ft",
    "away_corner_kicks_ft",
    "home_ball_possession_ft",
    "away_ball_possession_ft",
    "home_fouls_ft",
    "away_fouls_ft",
]
OPTIONAL_RAW_COLUMNS = [
    "home_expected_goals_ft",
    "away_expected_goals_ft",
]
NON_NUMERIC_RAW_COLUMNS = [
    "match_id",
    "season",
    "country",
    "competition",
    "date_utc",
    "home_team",
    "away_team",
]


def divide_safely(numerator, denominator):
    """Vectorized division that returns 0 when the denominator is 0."""
    denominator_as_array = numpy.asarray(denominator, dtype=float)
    numerator_as_array = numpy.asarray(numerator, dtype=float)
    return numpy.divide(
        numerator_as_array,
        denominator_as_array,
        out=numpy.zeros_like(numerator_as_array, dtype=float),
        where=denominator_as_array != 0,
    )


def remove_matches_with_low_history_teams(
    match_frame,
    minimum_matches=config.MINIMUM_TEAM_MATCHES,
):
    """Iteratively remove teams without enough matches for rolling features."""
    filtered_matches = match_frame.copy()
    filter_iterations = []

    while True:
        team_match_counts = pandas.concat([
            filtered_matches["home_team"],
            filtered_matches["away_team"],
        ]).value_counts()
        eligible_team_names = team_match_counts[team_match_counts >= minimum_matches].index
        next_filtered_matches = filtered_matches[
            filtered_matches["home_team"].isin(eligible_team_names)
            & filtered_matches["away_team"].isin(eligible_team_names)
        ].copy()

        filter_iterations.append({
            "iteration": len(filter_iterations) + 1,
            "matches_before": len(filtered_matches),
            "matches_after": len(next_filtered_matches),
            "teams_before": len(team_match_counts),
            "eligible_teams": len(eligible_team_names),
            "matches_removed": len(filtered_matches) - len(next_filtered_matches),
        })

        if len(next_filtered_matches) == len(filtered_matches):
            break
        filtered_matches = next_filtered_matches

    final_team_match_counts = pandas.concat([
        filtered_matches["home_team"],
        filtered_matches["away_team"],
    ]).value_counts()
    return filtered_matches, final_team_match_counts, pandas.DataFrame(filter_iterations)


def load_matches(
    path=None,
    extra_columns=None,
    minimum_team_matches=config.MINIMUM_TEAM_MATCHES,
):
    """Load the match dataset and add shared targets.

    Extra columns are numeric-coerced but are not required, so newer stats with
    partial coverage can be used by models that tolerate missing values.
    """
    if path is None:
        path = config.DATA_PATH
    extra_columns = [
        column_name
        for column_name in (extra_columns or [])
        if column_name not in REQUIRED_RAW_COLUMNS + OPTIONAL_RAW_COLUMNS
    ]
    raw_columns_used = REQUIRED_RAW_COLUMNS + OPTIONAL_RAW_COLUMNS + extra_columns

    matches = pandas.read_csv(path, usecols=raw_columns_used).copy()
    matches["date_utc"] = (
        pandas.to_datetime(matches["date_utc"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    numeric_raw_columns = [
        column_name
        for column_name in raw_columns_used
        if column_name not in NON_NUMERIC_RAW_COLUMNS
    ]
    for raw_column_name in numeric_raw_columns:
        matches[raw_column_name] = pandas.to_numeric(matches[raw_column_name], errors="coerce")

    rows_loaded = len(matches)
    matches = matches.dropna(subset=REQUIRED_RAW_COLUMNS).copy()
    rows_after_required_columns = len(matches)

    matches, team_match_counts, team_filter_summary = remove_matches_with_low_history_teams(
        matches,
        minimum_matches=minimum_team_matches,
    )
    matches = matches.sort_values(["date_utc", "match_id"]).reset_index(drop=True)
    matches = add_match_targets(matches)

    load_summary = {
        "rows_loaded": rows_loaded,
        "rows_after_required_columns": rows_after_required_columns,
        "rows_after_team_filter": len(matches),
        "rows_removed_for_required_columns": rows_loaded - rows_after_required_columns,
        "rows_removed_for_low_history_teams": rows_after_required_columns - len(matches),
        "remaining_teams": len(team_match_counts),
        "minimum_remaining_team_matches": (
            int(team_match_counts.min()) if len(team_match_counts) else 0
        ),
    }
    return matches, team_match_counts, team_filter_summary, load_summary

