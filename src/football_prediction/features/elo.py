"""Sequential Elo ratings with a margin-of-victory multiplier."""

from __future__ import annotations

import numpy
import pandas

from src.football_prediction.core import config


def add_elo_features(
    match_frame,
    initial_elo_value=config.INITIAL_ELO,
    elo_k_value=config.ELO_K,
    home_advantage_value=config.HOME_ADVANTAGE,
):
    """Attach pre-match Elo ratings and return final team ratings."""
    team_ratings = {}
    elo_records = []

    for match_record in match_frame.sort_values(["date_utc", "match_id"]).itertuples(index=False):
        home_rating = float(team_ratings.get(match_record.home_team, initial_elo_value))
        away_rating = float(team_ratings.get(match_record.away_team, initial_elo_value))

        expected_home_result = 1.0 / (
            1.0 + 10.0 ** (-((home_rating + home_advantage_value) - away_rating) / 400.0)
        )
        actual_home_result = 1.0 if match_record.home_win else 0.0 if match_record.away_win else 0.5

        goal_margin = abs(int(match_record.home_score_ft) - int(match_record.away_score_ft))
        margin_multiplier = numpy.log(max(goal_margin, 1) + 1) * (
            2.2 / (abs(home_rating - away_rating) * 0.001 + 2.2)
        )
        rating_change = elo_k_value * margin_multiplier * (
            actual_home_result - expected_home_result
        )

        elo_records.append({
            "match_id": match_record.match_id,
            "home_elo": home_rating,
            "away_elo": away_rating,
            "elo_difference": home_rating - away_rating,
        })

        team_ratings[match_record.home_team] = home_rating + rating_change
        team_ratings[match_record.away_team] = away_rating - rating_change

    matches_with_elo = match_frame.merge(pandas.DataFrame(elo_records), on="match_id", how="left")
    matches_with_elo = matches_with_elo.sort_values(["date_utc", "match_id"]).reset_index(drop=True)
    return matches_with_elo, dict(team_ratings)

