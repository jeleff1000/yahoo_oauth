"""
Playoff Simulation Module

Core simulation engine for playoff odds calculations.

This module handles:
- Team statistical modeling (mu_hat, sigma_hat)
- Regular season simulation
- Playoff bracket simulation
- Power ratings calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


# Configuration
N_SIMS = 10000
RNG_SEED = 42
PLAYOFF_SLOTS = 6
BYE_SLOTS = 2
BRACKET_RESEED = False


def get_rng(season: int, week: int, base: int = RNG_SEED) -> np.random.Generator:
    """Get deterministic RNG for season/week reproducibility."""
    mix = (int(base)
           ^ ((int(season) * 0x9E3779B1) & 0xFFFFFFFF)
           ^ ((int(week) * 0x85EBCA77) & 0xFFFFFFFF))
    return np.random.default_rng(mix & 0xFFFFFFFF)


def build_team_models(
    hist: pd.DataFrame,
    season: int,
    week: int,
    half_life: int,
    shrink_k: float,
    sigma_floor: float,
    boundary_penalty: float = 0.05,
    prior_w_cap: float = 2.0
) -> Tuple[Dict, Dict, Dict, float, float]:
    """
    Build statistical models for each team.

    Uses empirical Bayes shrinkage to estimate mu (mean points) and sigma (std dev).

    Args:
        hist: Historical matchup data with recency weights
        season: Current season
        week: Current week
        half_life: Half-life for recency weighting
        shrink_k: Shrinkage parameter for empirical Bayes
        sigma_floor: Minimum standard deviation
        boundary_penalty: Penalty for cross-season data
        prior_w_cap: Cap on prior season weight

    Returns:
        mu_hat: Mean points per team
        sigma_hat: Std dev per team
        samples_by_team: Historical samples for bootstrap
        league_mu: League average
        sigma_floor: Adjusted sigma floor
    """
    h = hist.copy()

    # Calculate recency weights
    hl = max(1, int(half_life))
    lam = np.log(2.0) / hl

    timeline = (h["year"] - season) * 100 + (h["week"] - week)
    weeks_ago = np.maximum(0, (-timeline).astype(float))
    base = np.exp(-lam * weeks_ago)

    prior = (h["year"] < season).astype(float)
    if week is None:
        fade = 1.0
    else:
        fade = min(max((week - 1) / 4.0, 0.0), 1.0)

    penalty = (1 - fade) * boundary_penalty + fade * 1.0
    h["w"] = base * np.where(prior == 1.0, penalty, 1.0)

    # Cap prior season weights
    prior_mask = h["year"] < season
    if prior_mask.any():
        w_prior = (h.loc[prior_mask]
                   .groupby("manager")["w"].transform(lambda s: s / max(1e-12, s.sum())))
        h.loc[prior_mask, "w"] = w_prior * prior_w_cap

    # League-wide statistics
    league_mu = h["team_points"].mean()
    league_sd = h["team_points"].std(ddof=1)

    # Per-manager statistics
    by_mgr = (
        h.groupby("manager")[["team_points", "w"]]
        .apply(lambda g: pd.Series({
            "n": g.shape[0],
            "w_sum": g["w"].sum(),
            "mu_raw": (g["team_points"] * g["w"]).sum() / (g["w"].sum() + 1e-12),
            "sd_raw": g["team_points"].std(ddof=1) if len(g) >= 2 else np.nan,
        }), include_groups=False)
    )

    # Empirical Bayes shrinkage
    k = float(shrink_k)
    weeks_played = max(0, int(week))
    k_eff = k * max(1.0, (4 - min(weeks_played, 4)) * 0.75)  # More shrinkage early season

    w_eb = by_mgr["w_sum"] / (by_mgr["w_sum"] + k_eff)
    mu_hat = (w_eb * by_mgr["mu_raw"] + (1 - w_eb) * league_mu).to_dict()

    # Standard deviation with floor
    sd_fill = by_mgr["sd_raw"].fillna(league_sd * 0.9)
    sigma_floor = max(float(sigma_floor), league_sd * 0.35)
    sigma_hat = sd_fill.clip(lower=sigma_floor).to_dict()

    # Bootstrap samples (for early season, use current season only)
    samples_by_team = {}
    for m, g in h.groupby("manager"):
        g_cur = g[g["year"] == season]
        if week <= 3 and len(g_cur) >= 2:
            samples_by_team[m] = g_cur[["team_points", "w"]].copy()
        else:
            samples_by_team[m] = g[["team_points", "w"]].copy()

    # Ensure all managers have parameters
    for m in h["manager"].unique():
        mu_hat.setdefault(m, league_mu)
        sigma_hat.setdefault(m, sigma_floor)

    return mu_hat, sigma_hat, samples_by_team, league_mu, sigma_floor


def ensure_params_for_future(
    mu_hat: Dict,
    sigma_hat: Dict,
    samples_by_team: Dict,
    df_future: Optional[pd.DataFrame],
    league_mu: float,
    sigma_floor: float
) -> None:
    """Ensure all future managers have model parameters."""
    if df_future is None or df_future.empty:
        return
    future_mgrs = set(df_future["manager"]).union(set(df_future["opponent"]))
    for m in future_mgrs:
        mu_hat.setdefault(m, league_mu)
        sigma_hat.setdefault(m, sigma_floor)
        samples_by_team.setdefault(m, pd.DataFrame({"team_points": [], "w": []}))


def compute_power_ratings(
    mu_hat: Dict,
    samples_by_team: Dict,
    bootstrap_min: int = 4
) -> pd.Series:
    """
    Calculate power ratings (weighted average of recent performance).

    Uses bootstrap sampling when enough data is available, otherwise uses mu_hat.
    """
    out = {}
    for m, samp in samples_by_team.items():
        use_boot = (
            isinstance(samp, pd.DataFrame)
            and "team_points" in samp.columns
            and "w" in samp.columns
            and len(samp) >= bootstrap_min
            and float(samp["w"].sum()) > 0
        )
        if use_boot:
            out[m] = float(np.average(samp["team_points"].to_numpy(),
                                      weights=samp["w"].to_numpy()))
        else:
            out[m] = float(mu_hat.get(m, np.mean(list(mu_hat.values()))))

    # Ensure all managers have ratings
    for m in mu_hat.keys():
        out.setdefault(m, float(mu_hat[m]))

    return pd.Series(out, name="Power_Rating")


def draw_score(
    manager: str,
    rng: np.random.Generator,
    mu_hat: Dict,
    sigma_hat: Dict,
    samples_by_team: Dict,
    use_bootstrap: bool = True,
    bootstrap_min: int = 4
) -> float:
    """
    Draw a random score for a manager.

    Uses bootstrap sampling from historical games if available, otherwise
    samples from normal distribution N(mu_hat, sigma_hat).
    """
    samp = samples_by_team.get(manager)
    if use_bootstrap and isinstance(samp, pd.DataFrame) and len(samp) >= bootstrap_min and float(samp["w"].sum()) > 0:
        p = (samp["w"] / samp["w"].sum()).to_numpy()
        return float(rng.choice(samp["team_points"].to_numpy(), p=p))

    mu = mu_hat.get(manager, np.mean(list(mu_hat.values())))
    sd = sigma_hat.get(manager, np.mean(list(sigma_hat.values())))
    return float(max(0.0, rng.normal(mu, sd)))


def simulate_game(
    team_a: str,
    team_b: str,
    rng: np.random.Generator,
    mu_hat: Dict,
    sigma_hat: Dict,
    samples_by_team: Dict
) -> str:
    """Simulate a single game between two teams."""
    sa = draw_score(team_a, rng, mu_hat, sigma_hat, samples_by_team)
    sb = draw_score(team_b, rng, mu_hat, sigma_hat, samples_by_team)
    if sa > sb:
        return team_a
    if sb > sa:
        return team_b
    return rng.choice([team_a, team_b])  # Coin flip for ties


def vectorized_regular_season_sim(
    reg_to_date: pd.DataFrame,
    future_canon: Optional[pd.DataFrame],
    mu_hat: Dict,
    sigma_hat: Dict,
    season: int,
    week: int,
    n_sims: int = N_SIMS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized simulation of remaining regular season games.

    Returns:
        W_all: (n_sims, n_managers) array of final wins
        PF_all: (n_sims, n_managers) array of final points
        mgr2idx: Dict mapping manager name to index
        idx2mgr: Array mapping index to manager name
    """
    rng = get_rng(season, week)

    # Build manager index
    managers = sorted(set(reg_to_date["manager"].unique()) | set(reg_to_date["opponent"].unique()))
    mgr2idx = {m: i for i, m in enumerate(managers)}
    idx2mgr = np.array(managers)
    n = len(managers)

    # Current wins/points
    from .playoff_helpers import wins_points_to_date
    wins_to_date, pf_to_date = wins_points_to_date(reg_to_date)
    WTD = np.zeros(n, dtype=float)
    PFTD = np.zeros(n, dtype=float)
    for m, v in wins_to_date.items():
        WTD[mgr2idx[m]] = v
    for m, v in pf_to_date.items():
        PFTD[mgr2idx[m]] = v

    # Convert mu/sigma to arrays
    mu_arr = np.array([mu_hat.get(m, np.mean(list(mu_hat.values()))) for m in managers])
    sd_arr = np.array([sigma_hat.get(m, np.mean(list(sigma_hat.values()))) for m in managers])

    # If no future games, return current state
    if future_canon is None or future_canon.empty:
        PF_all = np.tile(PFTD, (n_sims, 1))
        W_all = np.tile(WTD, (n_sims, 1))
        return W_all, PF_all, mgr2idx, idx2mgr

    # Vectorized simulation of future games
    A = future_canon["manager"].map(mgr2idx).to_numpy()
    B = future_canon["opponent"].map(mgr2idx).to_numpy()
    G = A.size

    # Draw scores for all games across all simulations
    SA = rng.normal(mu_arr[A][None, :], sd_arr[A][None, :], size=(n_sims, G))
    SB = rng.normal(mu_arr[B][None, :], sd_arr[B][None, :], size=(n_sims, G))

    # Accumulate points
    PF_add = np.zeros((n_sims, n), dtype=float)
    rows = np.arange(n_sims)[:, None]
    np.add.at(PF_add, (rows, A[None, :]), SA)
    np.add.at(PF_add, (rows, B[None, :]), SB)
    PF_all = PF_add + PFTD

    # Determine winners
    a_wins = (SA > SB).astype(float)
    ties = (SA == SB)
    if ties.any():
        coin = rng.integers(0, 2, size=ties.shape)
        a_wins = np.where(ties, coin, a_wins)
    b_wins = 1.0 - a_wins

    # Accumulate wins
    W_add = np.zeros((n_sims, n), dtype=float)
    np.add.at(W_add, (rows, A[None, :]), a_wins)
    np.add.at(W_add, (rows, B[None, :]), b_wins)
    W_all = W_add + WTD

    return W_all, PF_all, mgr2idx, idx2mgr


def vectorized_bracket_sim(
    W_all: np.ndarray,
    PF_all: np.ndarray,
    mgr2idx: Dict,
    idx2mgr: np.ndarray,
    mu_hat: Dict,
    sigma_hat: Dict,
    season: int,
    week: int,
    n_sims: int = N_SIMS,
    playoff_slots: int = PLAYOFF_SLOTS,
    bye_slots: int = BYE_SLOTS,
    bracket_reseed: bool = BRACKET_RESEED
) -> Dict[str, np.ndarray]:
    """
    Vectorized simulation of playoff bracket.

    Returns dict with arrays for:
    - seeds_idx: (n_sims, n_managers) - seed assignments
    - qf_winners: (n_sims, 2) - quarterfinal winners
    - sf_winners: (n_sims, 2) - semifinal winners
    - champions: (n_sims,) - championship winners
    """
    rng = get_rng(season, week)
    n = len(idx2mgr)

    # Seeding based on W, then PF
    order_w = np.argsort(-W_all, axis=1, kind="stable")
    W_sorted = np.take_along_axis(W_all, order_w, axis=1)
    PF_reord = np.take_along_axis(PF_all, order_w, axis=1)

    seeds_idx = np.empty_like(order_w)
    for s in range(n_sims):
        idx = np.lexsort(np.vstack([-PF_reord[s], -W_sorted[s]]))
        seeds_idx[s] = order_w[s, idx]

    # Extract top 6 seeds for playoffs
    seed1 = seeds_idx[:, 0]
    seed2 = seeds_idx[:, 1]
    qf1_a = seeds_idx[:, 3]  # 4 seed
    qf1_b = seeds_idx[:, 4]  # 5 seed
    qf2_a = seeds_idx[:, 2]  # 3 seed
    qf2_b = seeds_idx[:, 5]  # 6 seed

    # Convert to arrays for mu/sigma
    mu_arr = np.array([mu_hat.get(m, np.mean(list(mu_hat.values()))) for m in idx2mgr])
    sd_arr = np.array([sigma_hat.get(m, np.mean(list(sigma_hat.values()))) for m in idx2mgr])

    # Quarterfinals
    Q1A = rng.normal(mu_arr[qf1_a], sd_arr[qf1_a])
    Q1B = rng.normal(mu_arr[qf1_b], sd_arr[qf1_b])
    ties = (Q1A == Q1B)
    coin = rng.integers(0, 2, size=ties.shape[0])
    q1_w = np.where(Q1A > Q1B, qf1_a,
                    np.where(Q1B > Q1A, qf1_b,
                             np.where(coin == 0, qf1_a, qf1_b)))

    Q2A = rng.normal(mu_arr[qf2_a], sd_arr[qf2_a])
    Q2B = rng.normal(mu_arr[qf2_b], sd_arr[qf2_b])
    ties = (Q2A == Q2B)
    coin = rng.integers(0, 2, size=ties.shape[0])
    q2_w = np.where(Q2A > Q2B, qf2_a,
                    np.where(Q2B > Q2A, qf2_b,
                             np.where(coin == 0, qf2_a, qf2_b)))

    # Determine semifinal matchups
    if not bracket_reseed:
        s1_opp = q1_w
        s2_opp = q2_w
    else:
        # Reseed: best remaining seed plays worst
        inv_seed = np.full((n_sims, n), n, dtype=int)
        row_idx = np.arange(n_sims)[:, None]
        inv_seed[row_idx, seeds_idx] = np.arange(n)[None, :]

        low_is_q1 = inv_seed[row_idx.squeeze(), q1_w] < inv_seed[row_idx.squeeze(), q2_w]
        low = np.where(low_is_q1, q1_w, q2_w)
        high = np.where(low_is_q1, q2_w, q1_w)
        s1_opp, s2_opp = high, low

    # Semifinals
    S1A = rng.normal(mu_arr[seed1], sd_arr[seed1])
    S1B = rng.normal(mu_arr[s1_opp], sd_arr[s1_opp])
    ties = (S1A == S1B)
    coin = rng.integers(0, 2, size=ties.shape[0])
    s1_w = np.where(S1A > S1B, seed1,
                    np.where(S1B > S1A, s1_opp,
                             np.where(coin == 0, seed1, s1_opp)))

    S2A = rng.normal(mu_arr[seed2], sd_arr[seed2])
    S2B = rng.normal(mu_arr[s2_opp], sd_arr[s2_opp])
    ties = (S2A == S2B)
    coin = rng.integers(0, 2, size=ties.shape[0])
    s2_w = np.where(S2A > S2B, seed2,
                    np.where(S2B > S2A, s2_opp,
                             np.where(coin == 0, seed2, s2_opp)))

    # Championship
    FA = rng.normal(mu_arr[s1_w], sd_arr[s1_w])
    FB = rng.normal(mu_arr[s2_w], sd_arr[s2_w])
    ties = (FA == FB)
    coin = rng.integers(0, 2, size=ties.shape[0])
    champ = np.where(FA > FB, s1_w,
                     np.where(FB > FA, s2_w,
                              np.where(coin == 0, s1_w, s2_w)))

    return {
        "seeds_idx": seeds_idx,
        "seed1": seed1,
        "seed2": seed2,
        "qf_winners": np.column_stack([q1_w, q2_w]),
        "sf_winners": np.column_stack([s1_w, s2_w]),
        "champions": champ,
    }


def normalize_power_rating(power_series: pd.Series, inflation_rate: Optional[float]) -> pd.Series:
    """Normalize power rating by inflation rate."""
    if pd.isna(inflation_rate) or inflation_rate == 0 or inflation_rate is None:
        return power_series
    return power_series / float(inflation_rate)
