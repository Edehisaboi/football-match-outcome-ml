"""Team-state store for building future fixture features."""

from __future__ import annotations

import pandas

from src.football_prediction.core import config
from src.football_prediction.features.elo import add_elo_features
from src.football_prediction.features.rolling import make_team_match_rows


def build_team_state_store(
    results_frame,
    own_metrics=None,
    opponent_metrics=None,
    ratio_metrics=None,
    short_window=config.SHORT_WINDOW,
    long_window=config.LONG_WINDOW,
):
    """Snapshot Elo ratings, recent form, and head-to-head history."""
    if results_frame.empty:
        return {
            "team_elo_ratings": {},
            "team_form": {},
            "h2h_history": pandas.DataFrame(),
            "built_through": None,
        }

    history = results_frame.sort_values(["date_utc", "match_id"]).copy()
    _, team_elo_ratings = add_elo_features(history)
    history_team_rows = make_team_match_rows(
        history,
        own_metrics=own_metrics,
        opponent_metrics=opponent_metrics,
        ratio_metrics=ratio_metrics,
    )

    team_form = {}
    for team_name, team_history in history_team_rows.groupby("team"):
        ordered_team_history = team_history.sort_values(["date_utc", "match_id"])
        team_form[team_name] = {
            "recent_matches": ordered_team_history.tail(long_window).copy(),
            "recent_home_matches": ordered_team_history[
                ordered_team_history["venue"].eq("home")
            ].tail(short_window).copy(),
            "recent_away_matches": ordered_team_history[
                ordered_team_history["venue"].eq("away")
            ].tail(short_window).copy(),
            "last_fixture_date": ordered_team_history["date_utc"].max(),
        }

    h2h_columns = ["team", "opponent", "date_utc"]
    for optional_h2h_column in ["result", "goal_difference"]:
        if optional_h2h_column in history_team_rows.columns:
            h2h_columns.append(optional_h2h_column)

    return {
        "team_elo_ratings": team_elo_ratings,
        "team_form": team_form,
        "h2h_history": history_team_rows[h2h_columns].copy(),
        "built_through": history["date_utc"].max(),
    }


def validate_known_teams(home_team, away_team, known_team_names):
    """Raise a clear error for teams unseen in training."""
    unknown_team_names = [
        team_name
        for team_name in [home_team, away_team]
        if team_name not in known_team_names
    ]
    if unknown_team_names:
        raise ValueError(
            "Cannot predict fixtures for team names that were not present in training data: "
            + ", ".join(unknown_team_names)
        )


def state_mean(
    team_state_store,
    team_name,
    history_bucket,
    metric_name,
    window=config.SHORT_WINDOW,
    minimum_matches=None,
    required=True,
):
    """Return a recent-history mean from a saved team-state store."""
    if minimum_matches is None:
        minimum_matches = window
    recent_history = team_state_store["team_form"].get(team_name, {}).get(history_bucket)
    if recent_history is None:
        available_values = pandas.Series(dtype=float)
    else:
        available_values = pandas.to_numeric(
            recent_history[metric_name].tail(window),
            errors="coerce",
        ).dropna()
    if len(available_values) < minimum_matches:
        if required:
            raise ValueError(
                f"Cannot build {metric_name} for {team_name}: fewer than "
                f"{minimum_matches} usable matches in {history_bucket}."
            )
        return float("nan")
    return float(available_values.mean())


def state_rest_days(
    team_state_store,
    team_name,
    fixture_date,
    maximum_rest_days=config.MAXIMUM_REST_DAYS,
):
    """Days since a team's last fixture, capped at the configured maximum."""
    last_fixture_date = team_state_store["team_form"].get(team_name, {}).get("last_fixture_date")
    if last_fixture_date is None or pandas.isna(last_fixture_date):
        raise ValueError(f"Cannot build rest days for {team_name}: no previous fixture found.")
    return min(float((fixture_date - last_fixture_date).days), maximum_rest_days)


def state_head_to_head(team_state_store, home_team, away_team):
    """Return prior home-perspective H2H count, result rate, and goal difference."""
    h2h_history = team_state_store["h2h_history"]
    if h2h_history.empty:
        return 0.0, float("nan"), float("nan")
    previous_meetings = h2h_history[
        h2h_history["team"].eq(home_team) & h2h_history["opponent"].eq(away_team)
    ]
    if previous_meetings.empty:
        return 0.0, float("nan"), float("nan")
    return (
        float(len(previous_meetings)),
        float(previous_meetings["result"].mean()),
        float(previous_meetings["goal_difference"].mean()),
    )

