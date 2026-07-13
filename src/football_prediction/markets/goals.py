"""Goal score grids and goal-market probabilities."""

from __future__ import annotations

import numpy
from scipy import stats
from scipy.optimize import minimize_scalar


def poisson_pmf_matrix(mu, max_count):
    """Poisson PMF per row, with shape (n_matches, max_count + 1)."""
    mu = numpy.asarray(mu, dtype=float).reshape(-1, 1)
    counts = numpy.arange(max_count + 1).reshape(1, -1)
    return stats.poisson.pmf(counts, mu)


def dixon_coles_tau(home_goals, away_goals, home_mu, away_mu, rho):
    """Dixon-Coles factor for the four low-score cells."""
    home_goals = numpy.asarray(home_goals)
    away_goals = numpy.asarray(away_goals)
    home_mu = numpy.asarray(home_mu, dtype=float)
    away_mu = numpy.asarray(away_mu, dtype=float)

    tau = numpy.ones(numpy.broadcast(home_goals, away_goals, home_mu, away_mu).shape)
    tau = numpy.where((home_goals == 0) & (away_goals == 0), 1.0 - home_mu * away_mu * rho, tau)
    tau = numpy.where((home_goals == 0) & (away_goals == 1), 1.0 + home_mu * rho, tau)
    tau = numpy.where((home_goals == 1) & (away_goals == 0), 1.0 + away_mu * rho, tau)
    tau = numpy.where((home_goals == 1) & (away_goals == 1), 1.0 - rho, tau)
    return tau


def build_score_grid(home_mu, away_mu, max_goals=10, rho=0.0):
    """Joint score-probability grids with optional Dixon-Coles adjustment."""
    home_mu = numpy.asarray(home_mu, dtype=float)
    away_mu = numpy.asarray(away_mu, dtype=float)
    home_pmf = poisson_pmf_matrix(home_mu, max_goals)
    away_pmf = poisson_pmf_matrix(away_mu, max_goals)
    grid = home_pmf[:, :, None] * away_pmf[:, None, :]

    if rho != 0.0:
        for home_goals, away_goals in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            grid[:, home_goals, away_goals] *= dixon_coles_tau(
                home_goals,
                away_goals,
                home_mu,
                away_mu,
                rho,
            )
        grid = numpy.clip(grid, 0.0, None)

    pre_normalization_mass = grid.sum(axis=(1, 2))
    grid = grid / pre_normalization_mass[:, None, None]
    return grid, pre_normalization_mass


def fit_dixon_coles_rho(home_goals, away_goals, home_mu, away_mu, rho_bounds=(-0.2, 0.2)):
    """Fit Dixon-Coles rho while keeping the predicted means fixed."""
    home_goals = numpy.asarray(home_goals, dtype=int)
    away_goals = numpy.asarray(away_goals, dtype=int)
    home_mu = numpy.asarray(home_mu, dtype=float)
    away_mu = numpy.asarray(away_mu, dtype=float)

    def negative_log_likelihood(rho):
        tau = dixon_coles_tau(home_goals, away_goals, home_mu, away_mu, rho)
        if numpy.any(tau <= 0):
            return numpy.inf
        return -numpy.sum(
            numpy.log(tau)
            + stats.poisson.logpmf(home_goals, home_mu)
            + stats.poisson.logpmf(away_goals, away_mu)
        )

    result = minimize_scalar(negative_log_likelihood, bounds=rho_bounds, method="bounded")
    return float(result.x)


def _total_goals_index(grid):
    grid_size = grid.shape[1]
    home_counts = numpy.arange(grid_size).reshape(-1, 1)
    away_counts = numpy.arange(grid_size).reshape(1, -1)
    return home_counts + away_counts


def grid_total_pmf(grid):
    """Distribution of total goals per match."""
    total_index = _total_goals_index(grid)
    maximum_total = 2 * (grid.shape[1] - 1)
    total_pmf = numpy.zeros((grid.shape[0], maximum_total + 1))
    for total_value in range(maximum_total + 1):
        cell_mask = total_index == total_value
        total_pmf[:, total_value] = grid[:, cell_mask].sum(axis=1)
    return total_pmf


def grid_total_over(grid, line):
    """Probability that total goals are above a half-goal line."""
    minimum_count = int(numpy.floor(line)) + 1
    total_index = _total_goals_index(grid)
    return grid[:, total_index >= minimum_count].sum(axis=1)


def grid_team_over(grid, side, line):
    """Probability that one side's goals are above a half-goal line."""
    minimum_count = int(numpy.floor(line)) + 1
    if side == "home":
        return grid[:, minimum_count:, :].sum(axis=(1, 2))
    if side == "away":
        return grid[:, :, minimum_count:].sum(axis=(1, 2))
    raise ValueError(f"side must be 'home' or 'away', got {side!r}")


def grid_btts(grid):
    """Probability both teams score at least once."""
    return grid[:, 1:, 1:].sum(axis=(1, 2))


def grid_1x2(grid):
    """Home-win, draw, away-win probabilities per match."""
    grid_size = grid.shape[1]
    home_counts = numpy.arange(grid_size).reshape(-1, 1)
    away_counts = numpy.arange(grid_size).reshape(1, -1)
    home_win_probability = grid[:, home_counts > away_counts].sum(axis=1)
    draw_probability = grid[:, home_counts == away_counts].sum(axis=1)
    away_win_probability = grid[:, home_counts < away_counts].sum(axis=1)
    return numpy.column_stack([home_win_probability, draw_probability, away_win_probability])


def grid_goal_ranges(grid, range_bins=((0, 1), (2, 3), (4, 5), (6, None))):
    """Probability that total goals fall in each requested range."""
    total_pmf = grid_total_pmf(grid)
    maximum_total = total_pmf.shape[1] - 1
    range_probabilities = {}
    for lower_bound, upper_bound in range_bins:
        if upper_bound is None:
            label = f"{lower_bound}+"
            upper_bound = maximum_total
        else:
            label = f"{lower_bound}-{upper_bound}"
        range_probabilities[label] = total_pmf[:, lower_bound:upper_bound + 1].sum(axis=1)
    return range_probabilities


def grid_exact_scores(grid, top_k=5):
    """Most likely scorelines per match."""
    grid_size = grid.shape[1]
    flat_grid = grid.reshape(grid.shape[0], -1)
    top_cells = numpy.argsort(flat_grid, axis=1)[:, ::-1][:, :top_k]
    scorelines = []
    for match_index in range(flat_grid.shape[0]):
        match_scorelines = []
        for cell_index in top_cells[match_index]:
            home_goals, away_goals = divmod(int(cell_index), grid_size)
            match_scorelines.append((f"{home_goals}-{away_goals}", float(flat_grid[match_index, cell_index])))
        scorelines.append(match_scorelines)
    return scorelines

