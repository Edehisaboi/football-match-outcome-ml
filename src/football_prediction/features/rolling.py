"""Team-perspective rows, rolling form, and head-to-head features."""

from __future__ import annotations

import pandas

from src.football_prediction.core import config
from src.football_prediction.data.loading import divide_safely

DEFAULT_OWN_METRICS = {
    "goals_scored": "{side}_score_ft",
    "points": "{side}_points",
    "shots_on_target": "{side}_shots_on_target_ft",
    "corners": "{side}_corner_kicks_ft",
    "possession": "{side}_ball_possession_ft",
    "fouls": "{side}_fouls_ft",
    "expected_goals_for": "{side}_expected_goals_ft",
}
DEFAULT_OPPONENT_METRICS = {
    "goals_conceded": "{side}_score_ft",
    "shots_on_target_conceded": "{side}_shots_on_target_ft",
    "expected_goals_against": "{side}_expected_goals_ft",
}
DEFAULT_RATIO_METRICS = {
    "shot_accuracy": ("{side}_shots_on_target_ft", "{side}_total_shots_ft"),
}
DEFAULT_SHORT_ROLLING_METRICS = [
    "points",
    "goals_scored",
    "goals_conceded",
    "shots_on_target",
    "shots_on_target_conceded",
    "shot_accuracy",
    "corners",
    "possession",
    "fouls",
    "goal_difference",
]
DEFAULT_LONG_ROLLING_METRICS = ["points", "goal_difference"]
DEFAULT_NAN_TOLERANT_ROLLING_METRICS = [
    "expected_goals_for",
    "expected_goals_against",
    "finishing_luck",
]
DEFAULT_VENUE_ROLLING_METRICS = ["points"]


def make_team_match_rows(
    match_frame,
    own_metrics=None,
    opponent_metrics=None,
    ratio_metrics=None,
):
    """Convert each match into one home-team row and one away-team row."""
    if own_metrics is None:
        own_metrics = DEFAULT_OWN_METRICS
    if opponent_metrics is None:
        opponent_metrics = DEFAULT_OPPONENT_METRICS
    if ratio_metrics is None:
        ratio_metrics = DEFAULT_RATIO_METRICS

    perspective_frames = []
    for venue_name, own_prefix, opponent_prefix in [("home", "home", "away"), ("away", "away", "home")]:
        perspective_columns = {
            "match_id": match_frame["match_id"],
            "season": match_frame["season"],
            "date_utc": match_frame["date_utc"],
            "team": match_frame[f"{own_prefix}_team"],
            "opponent": match_frame[f"{opponent_prefix}_team"],
            "venue": venue_name,
        }
        for metric_name, column_template in own_metrics.items():
            perspective_columns[metric_name] = match_frame[column_template.format(side=own_prefix)]
        for metric_name, column_template in opponent_metrics.items():
            perspective_columns[metric_name] = match_frame[column_template.format(side=opponent_prefix)]
        for metric_name, (numerator_template, denominator_template) in ratio_metrics.items():
            perspective_columns[metric_name] = divide_safely(
                match_frame[numerator_template.format(side=own_prefix)],
                match_frame[denominator_template.format(side=own_prefix)],
            )
        perspective_frames.append(pandas.DataFrame(perspective_columns))

    team_match_rows = pandas.concat(perspective_frames, ignore_index=True)
    if {"goals_scored", "goals_conceded"} <= set(team_match_rows.columns):
        team_match_rows["goal_difference"] = (
            team_match_rows["goals_scored"] - team_match_rows["goals_conceded"]
        )
    if "points" in team_match_rows.columns:
        team_match_rows["result"] = team_match_rows["points"].map({3: 1.0, 1: 0.5, 0: 0.0})
    if {"goals_scored", "expected_goals_for"} <= set(team_match_rows.columns):
        team_match_rows["finishing_luck"] = (
            team_match_rows["goals_scored"] - team_match_rows["expected_goals_for"]
        )
    return team_match_rows.sort_values(["team", "date_utc", "match_id"]).reset_index(drop=True)


def form_column_names(
    short_metrics=None,
    long_metrics=None,
    nan_tolerant_metrics=None,
    venue_metrics=None,
):
    """Return the columns produced by add_rolling_team_form."""
    if short_metrics is None:
        short_metrics = DEFAULT_SHORT_ROLLING_METRICS
    if long_metrics is None:
        long_metrics = DEFAULT_LONG_ROLLING_METRICS
    if nan_tolerant_metrics is None:
        nan_tolerant_metrics = DEFAULT_NAN_TOLERANT_ROLLING_METRICS
    if venue_metrics is None:
        venue_metrics = DEFAULT_VENUE_ROLLING_METRICS
    return (
        [f"rolling_{metric_name}" for metric_name in short_metrics]
        + [f"long_rolling_{metric_name}" for metric_name in long_metrics]
        + [f"rolling_{metric_name}" for metric_name in nan_tolerant_metrics]
        + [f"rolling_{metric_name}_at_venue" for metric_name in venue_metrics]
        + ["rest_days"]
    )


def add_rolling_team_form(
    team_match_rows,
    short_metrics=None,
    long_metrics=None,
    nan_tolerant_metrics=None,
    venue_metrics=None,
    short_window=config.SHORT_WINDOW,
    long_window=config.LONG_WINDOW,
    long_window_minimum_matches=config.LONG_WINDOW_MINIMUM_MATCHES,
    nan_tolerant_window=config.XG_WINDOW,
    nan_tolerant_minimum_matches=config.XG_WINDOW_MINIMUM_MATCHES,
    maximum_rest_days=config.MAXIMUM_REST_DAYS,
):
    """Build leakage-safe rolling means per team."""
    if short_metrics is None:
        short_metrics = DEFAULT_SHORT_ROLLING_METRICS
    if long_metrics is None:
        long_metrics = DEFAULT_LONG_ROLLING_METRICS
    if nan_tolerant_metrics is None:
        nan_tolerant_metrics = DEFAULT_NAN_TOLERANT_ROLLING_METRICS
    if venue_metrics is None:
        venue_metrics = DEFAULT_VENUE_ROLLING_METRICS

    team_form_parts = []
    for _, team_history in team_match_rows.groupby("team", sort=False):
        ordered_team_history = team_history.sort_values(["date_utc", "match_id"]).copy()
        for metric_name in short_metrics:
            ordered_team_history[f"rolling_{metric_name}"] = (
                ordered_team_history[metric_name].rolling(short_window, closed="left").mean()
            )
        for metric_name in long_metrics:
            ordered_team_history[f"long_rolling_{metric_name}"] = (
                ordered_team_history[metric_name]
                .rolling(long_window, min_periods=long_window_minimum_matches, closed="left")
                .mean()
            )
        for metric_name in nan_tolerant_metrics:
            ordered_team_history[f"rolling_{metric_name}"] = (
                ordered_team_history[metric_name]
                .rolling(nan_tolerant_window, min_periods=nan_tolerant_minimum_matches, closed="left")
                .mean()
            )
        ordered_team_history["rest_days"] = (
            ordered_team_history["date_utc"].diff().dt.days.clip(upper=maximum_rest_days)
        )
        team_form_parts.append(ordered_team_history)

    team_form_with_all_venues = pandas.concat(team_form_parts, ignore_index=True)

    venue_form_parts = []
    for _, venue_history in team_form_with_all_venues.groupby(["team", "venue"], sort=False):
        ordered_venue_history = venue_history.sort_values(["date_utc", "match_id"]).copy()
        for metric_name in venue_metrics:
            ordered_venue_history[f"rolling_{metric_name}_at_venue"] = (
                ordered_venue_history[metric_name].rolling(short_window, closed="left").mean()
            )
        venue_form_parts.append(ordered_venue_history)

    team_form = pandas.concat(venue_form_parts, ignore_index=True)
    return team_form.sort_values(["date_utc", "match_id", "venue"]).reset_index(drop=True)


def merge_side_form(feature_table, rolling_team_form, column_names):
    """Attach home and away rolling form columns to the match table."""
    for side_name in ["home", "away"]:
        side_form = rolling_team_form[rolling_team_form["venue"].eq(side_name)][
            ["match_id"] + list(column_names)
        ].rename(columns={column_name: f"{side_name}_{column_name}" for column_name in column_names})
        feature_table = feature_table.merge(side_form, on="match_id", how="left")
    return feature_table


def add_head_to_head_features(match_frame, team_match_rows):
    """Add prior head-to-head count, result rate, and goal-difference features."""
    ordered_pair_rows = team_match_rows.sort_values(
        ["team", "opponent", "date_utc", "match_id"]
    ).copy()
    pair_history = ordered_pair_rows.groupby(["team", "opponent"], sort=False)

    ordered_pair_rows["h2h_matches_played"] = pair_history.cumcount()
    ordered_pair_rows["h2h_win_rate"] = pair_history["result"].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )
    ordered_pair_rows["h2h_goal_difference"] = pair_history["goal_difference"].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )

    home_perspective_h2h = ordered_pair_rows[[
        "match_id",
        "team",
        "opponent",
        "h2h_matches_played",
        "h2h_win_rate",
        "h2h_goal_difference",
    ]].rename(columns={
        "team": "home_team",
        "opponent": "away_team",
        "h2h_win_rate": "h2h_home_win_rate",
        "h2h_goal_difference": "h2h_home_goal_difference",
    })

    return match_frame.merge(
        home_perspective_h2h,
        on=["match_id", "home_team", "away_team"],
        how="left",
    )

