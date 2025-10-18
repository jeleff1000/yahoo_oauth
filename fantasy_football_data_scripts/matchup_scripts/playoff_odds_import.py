import numpy as np
import pandas as pd
from pathlib import Path
from md.md_utils import df_from_md_or_parquet

# =========================================================
# KMFFL Odds Engine – Parquet version
# =========================================================

# -------------------------
# Config
# -------------------------
PLAYOFF_SLOTS = 6
BYE_SLOTS = 2
HALF_LIFE_WEEKS = 10
SHRINK_K = 6.0
SIGMA_FLOOR_MIN = 10
N_SIMS = 10000
RNG_SEED = 42
BRACKET_RESEED = False

# -------------------------
# File paths (relative to script location)
# -------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "fantasy_football_data"

MATCHUP_PATH = DATA_DIR / "matchup.parquet"
SCHEDULE_PATH = DATA_DIR / "schedule.parquet"
PLAYER_PATH = DATA_DIR / "player.parquet"
DRAFT_PATH = DATA_DIR / "draft.parquet"
TRANSACTION_PATH = DATA_DIR / "transactions.parquet"

# Output paths
OUTPUT_MATCHUP_PARQUET = DATA_DIR / "matchup.parquet"
OUTPUT_MATCHUP_CSV = DATA_DIR / "matchup.csv"

# -------------------------
# Target columns to fill
# -------------------------
TARGET_COLS = [
    "avg_seed", "p_playoffs", "p_bye", "exp_final_wins", "exp_final_pf",
    "p_semis", "p_final", "p_champ",
    "x1_seed", "x2_seed", "x3_seed", "x4_seed", "x5_seed",
    "x6_seed", "x7_seed", "x8_seed", "x9_seed", "x10_seed",
    "power_rating",
    "x0_win", "x1_win", "x2_win", "x3_win", "x4_win", "x5_win", "x6_win", "x7_win",
    "x8_win", "x9_win", "x10_win", "x11_win", "x12_win", "x13_win", "x14_win",
]

# -------------------------
# Round column names
# -------------------------
ROUND_COLS = {
    "qf": "quarterfinal",
    "sf": "semifinal",
    "fn": "championship",
}


# -------------------------
# RNG helper
# -------------------------
def get_rng(season: int, week: int, base: int = RNG_SEED) -> np.random.Generator:
    mix = (int(base)
           ^ ((int(season) * 0x9E3779B1) & 0xFFFFFFFF)
           ^ ((int(week) * 0x85EBCA77) & 0xFFFFFFFF))
    return np.random.default_rng(mix & 0xFFFFFFFF)


# -------------------------
# Helpers: keys & canonical
# -------------------------
def add_match_key(df):
    df = df.copy()
    df["mA"] = df[["manager", "opponent"]].min(axis=1)
    df["mB"] = df[["manager", "opponent"]].max(axis=1)
    df["match_key"] = list(zip(df["year"], df["week"], df["mA"], df["mB"]))
    return df


def canonicalize(df):
    df = df.copy()
    df["mA"] = df[["manager", "opponent"]].min(axis=1)
    df["mB"] = df[["manager", "opponent"]].max(axis=1)
    df["match_key"] = list(zip(df["year"], df["week"], df["mA"], df["mB"]))
    return df[df["manager"] == df["mA"]]


# -------------------------
# Enforce playoff monotonicity
# -------------------------
def _sim_game(a, b, rng, mu_hat, sigma_hat, samples_by_team):
    sa = draw_score(a, rng, mu_hat, sigma_hat, samples_by_team)
    sb = draw_score(b, rng, mu_hat, sigma_hat, samples_by_team)
    if sa > sb: return a
    if sb > sa: return b
    return rng.choice([a, b])


def enforce_playoff_monotonicity(df):
    df = df.copy()
    if "is_playoffs" not in df.columns:
        return df
    for yr in sorted(df["year"].dropna().unique().astype(int)):
        season_mask = df["year"] == yr
        po_weeks = df.loc[season_mask & (df["is_playoffs"] == 1), "week"].dropna()
        if po_weeks.empty:
            continue
        start_po = int(po_weeks.min())
        df.loc[season_mask & (df["week"] >= start_po), "is_playoffs"] = 1
    return df


# -------------------------
# Recency weights
# -------------------------
def recency_weights(df, season, week, half_life, boundary_penalty=0.05):
    hl = max(1, int(half_life))
    lam = np.log(2.0) / hl

    timeline = (df["year"] - season) * 100 + (df["week"] - week)
    weeks_ago = np.maximum(0, (-timeline).astype(float))
    base = np.exp(-lam * weeks_ago)

    prior = (df["year"] < season).astype(float)
    if week is None:
        fade = 1.0
    else:
        fade = min(max((week - 1) / 4.0, 0.0), 1.0)

    penalty = (1 - fade) * boundary_penalty + fade * 1.0
    return base * np.where(prior == 1.0, penalty, 1.0)


# -------------------------
# Win/points snapshot
# -------------------------
def wins_points_to_date(played_raw):
    tmp = add_match_key(played_raw)

    def _win_val(s):
        if len(s) < 2:
            return pd.Series([np.nan] * len(s), index=s.index)
        a, b = s.iloc[0], s.iloc[1]
        if a > b: return pd.Series([1.0, 0.0], index=s.index)
        if b > a: return pd.Series([0.0, 1.0], index=s.index)
        return pd.Series([0.5, 0.5], index=s.index)

    tmp["win_val"] = tmp.groupby("match_key")["team_points"].transform(lambda s: _win_val(s).values)
    wins = tmp.groupby("manager")["win_val"].sum().astype(float)
    pts = tmp.groupby("manager")["team_points"].sum().astype(float)
    return wins, pts


# -------------------------
# Ranking/seeding
# -------------------------
def rank_and_seed(wins, points, playoff_slots, bye_slots, played_raw=None):
    managers = sorted(set(wins.index) | set(points.index))
    w = wins.reindex(managers).fillna(0.0)
    pf = points.reindex(managers).fillna(0.0)

    if played_raw is not None and len(played_raw) > 0:
        mgr_weeks = pd.concat([
            played_raw[["manager", "year", "week"]].rename(columns={"manager": "name"}),
            played_raw[["opponent", "year", "week"]].rename(columns={"opponent": "name"})
        ])
        games_played = (mgr_weeks.drop_duplicates()
                        .groupby("name").size()
                        .reindex(managers).fillna(0).astype(int))
    else:
        games_played = pd.Series(0, index=managers)

    l = (games_played - w).clip(lower=0)
    table = (pd.DataFrame({"manager": managers, "W": w.values, "L": l.values, "PF": pf.values})
             .sort_values(["W", "PF"], ascending=[False, False])
             .reset_index(drop=True))
    table["seed"] = np.arange(1, len(table) + 1)
    table["made_playoffs"] = table["seed"] <= playoff_slots
    table["bye"] = table["seed"] <= bye_slots
    return table[["seed", "manager", "W", "L", "PF", "made_playoffs", "bye"]]


# -------------------------
# Team models
# -------------------------
def build_team_models(hist, season, week, half_life, shrink_k, sigma_floor,
                      boundary_penalty=0.05, prior_w_cap=2.0):
    h = hist.copy()
    h["w"] = recency_weights(h, season, week, half_life, boundary_penalty)

    prior_mask = h["year"] < season
    if prior_mask.any():
        w_prior = (h.loc[prior_mask]
                   .groupby("manager")["w"].transform(lambda s: s / max(1e-12, s.sum())))
        h.loc[prior_mask, "w"] = w_prior * prior_w_cap

    league_mu = h["team_points"].mean()
    league_sd = h["team_points"].std(ddof=1)

    by_mgr = (
        h.groupby("manager")[["team_points", "w"]]
        .apply(lambda g: pd.Series({
            "n": g.shape[0],
            "w_sum": g["w"].sum(),
            "mu_raw": (g["team_points"] * g["w"]).sum() / (g["w"].sum() + 1e-12),
            "sd_raw": g["team_points"].std(ddof=1) if len(g) >= 2 else np.nan,
        }), include_groups=False)
    )

    k = float(shrink_k)
    weeks_played = max(0, int(week))
    k_eff = k * max(1.0, (4 - min(weeks_played, 4)) * 0.75)

    w_eb = by_mgr["w_sum"] / (by_mgr["w_sum"] + k_eff)
    mu_hat = (w_eb * by_mgr["mu_raw"] + (1 - w_eb) * league_mu).to_dict()

    sd_fill = by_mgr["sd_raw"].fillna(league_sd * 0.9)
    sigma_floor = max(float(sigma_floor), league_sd * 0.35)
    sigma_hat = sd_fill.clip(lower=sigma_floor).to_dict()

    samples_by_team = {}
    for m, g in h.groupby("manager"):
        g_cur = g[g["year"] == season]
        if week <= 3 and len(g_cur) >= 2:
            samples_by_team[m] = g_cur[["team_points", "w"]].copy()
        else:
            samples_by_team[m] = g[["team_points", "w"]].copy()

    for m in h["manager"].unique():
        mu_hat.setdefault(m, league_mu)
        sigma_hat.setdefault(m, sigma_floor)

    return mu_hat, sigma_hat, samples_by_team, league_mu, sigma_floor


def ensure_params_for_future(mu_hat, sigma_hat, samples_by_team, df_future, league_mu, sigma_floor):
    if df_future is None or df_future.empty:
        return
    future_mgrs = set(df_future["manager"]).union(set(df_future["opponent"]))
    for m in future_mgrs:
        mu_hat.setdefault(m, league_mu)
        sigma_hat.setdefault(m, sigma_floor)
        samples_by_team.setdefault(m, pd.DataFrame({"team_points": [], "w": []}))


# -------------------------
# Power rating calculator
# -------------------------
def compute_power_ratings(mu_hat, samples_by_team, bootstrap_min=4):
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
    for m in mu_hat.keys():
        out.setdefault(m, float(mu_hat[m]))
    return pd.Series(out, name="Power_Rating")


# -------------------------
# Score draw
# -------------------------
def draw_score(manager, rng, mu_hat, sigma_hat, samples_by_team, use_bootstrap=True, bootstrap_min=4):
    samp = samples_by_team.get(manager)
    if use_bootstrap and isinstance(samp, pd.DataFrame) and len(samp) >= bootstrap_min and float(samp["w"].sum()) > 0:
        p = (samp["w"] / samp["w"].sum()).to_numpy()
        return float(rng.choice(samp["team_points"].to_numpy(), p=p))
    mu = mu_hat.get(manager, np.mean(list(mu_hat.values())))
    sd = sigma_hat.get(manager, np.mean(list(sigma_hat.values())))
    return float(max(0.0, rng.normal(mu, sd)))


# -------------------------
# Schedule helpers
# -------------------------
def schedules_last_regular_week(df_sched, season):
    s = df_sched[(df_sched["year"] == season)]
    if s.empty:
        return None
    po = s.loc[s["is_playoffs"] == 1, "week"].dropna()
    if not po.empty:
        return int(po.min()) - 1
    return int(s["week"].max())


def future_regular_from_schedule(df_sched, season, current_week):
    s = df_sched[(df_sched["year"] == season) & (df_sched["is_playoffs"] == 0)]
    if s.empty:
        return pd.DataFrame(columns=["year", "week", "manager", "opponent"])
    s = s[s["week"] > current_week].copy()
    if s.empty:
        return s[["year", "week", "manager", "opponent"]]
    s = canonicalize(s.rename(columns={"Opponent Week": "opponent_week", "OpponentYear": "opponent_year"}))
    return s[["year", "week", "manager", "opponent"]].drop_duplicates()


# -------------------------
# Vectorized regular and bracket sim
# -------------------------
def _vectorized_regular_and_bracket(reg_to_date, future_canon, mu_hat, sigma_hat, season, week):
    rng = get_rng(season, week)

    managers = sorted(set(reg_to_date["manager"].unique()) | set(reg_to_date["opponent"].unique()))
    mgr2idx = {m: i for i, m in enumerate(managers)}
    idx2mgr = np.array(managers)
    n = len(managers)

    wins_to_date, pf_to_date = wins_points_to_date(reg_to_date)
    WTD = np.zeros(n, dtype=float)
    PFTD = np.zeros(n, dtype=float)
    for m, v in wins_to_date.items(): WTD[mgr2idx[m]] = v
    for m, v in pf_to_date.items():  PFTD[mgr2idx[m]] = v

    mu_arr = np.array([mu_hat.get(m, np.mean(list(mu_hat.values()))) for m in managers])
    sd_arr = np.array([sigma_hat.get(m, np.mean(list(sigma_hat.values()))) for m in managers])

    if future_canon is None or future_canon.empty:
        PF_all = np.tile(PFTD, (N_SIMS, 1))
        W_all = np.tile(WTD, (N_SIMS, 1))
    else:
        A = future_canon["manager"].map(mgr2idx).to_numpy()
        B = future_canon["opponent"].map(mgr2idx).to_numpy()
        G = A.size

        SA = rng.normal(mu_arr[A][None, :], sd_arr[A][None, :], size=(N_SIMS, G))
        SB = rng.normal(mu_arr[B][None, :], sd_arr[B][None, :], size=(N_SIMS, G))

        PF_add = np.zeros((N_SIMS, n), dtype=float)
        rows = np.arange(N_SIMS)[:, None]
        np.add.at(PF_add, (rows, A[None, :]), SA)
        np.add.at(PF_add, (rows, B[None, :]), SB)
        PF_all = PF_add + PFTD

        a_wins = (SA > SB).astype(float)
        ties = (SA == SB)
        if ties.any():
            coin = rng.integers(0, 2, size=ties.shape)
            a_wins = np.where(ties, coin, a_wins)
        b_wins = 1.0 - a_wins

        W_add = np.zeros((N_SIMS, n), dtype=float)
        np.add.at(W_add, (rows, A[None, :]), a_wins)
        np.add.at(W_add, (rows, B[None, :]), b_wins)
        W_all = W_add + WTD

    order_w = np.argsort(-W_all, axis=1, kind="stable")
    W_sorted = np.take_along_axis(W_all, order_w, axis=1)
    PF_reord = np.take_along_axis(PF_all, order_w, axis=1)

    seeds_idx = np.empty_like(order_w)
    for s in range(N_SIMS):
        idx = np.lexsort(np.vstack([-PF_reord[s], -W_sorted[s]]))
        seeds_idx[s] = order_w[s, idx]

    inv_seed = np.full((N_SIMS, n), n, dtype=int)
    row_idx = np.arange(N_SIMS)[:, None]
    inv_seed[row_idx, seeds_idx] = np.arange(n)[None, :]
    seed_numbers = inv_seed + 1

    seed1 = seeds_idx[:, 0]
    seed2 = seeds_idx[:, 1]
    qf1_a = seeds_idx[:, 3]
    qf1_b = seeds_idx[:, 4]
    qf2_a = seeds_idx[:, 2]
    qf2_b = seeds_idx[:, 5]

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

    if not BRACKET_RESEED:
        s1_opp = q1_w
        s2_opp = q2_w
    else:
        low_is_q1 = inv_seed[row_idx, q1_w] < inv_seed[row_idx, q2_w]
        low = np.where(low_is_q1, q1_w, q2_w)
        high = np.where(low_is_q1, q2_w, q1_w)
        s1_opp, s2_opp = high, low

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

    FA = rng.normal(mu_arr[s1_w], sd_arr[s1_w])
    FB = rng.normal(mu_arr[s2_w], sd_arr[s2_w])
    ties = (FA == FB)
    coin = rng.integers(0, 2, size=ties.shape[0])
    champ = np.where(FA > FB, s1_w,
                     np.where(FB > FA, s2_w,
                              np.where(coin == 0, s1_w, s2_w)))

    def pct_counts(idx_arr, nteams):
        arr = np.asarray(idx_arr).reshape(-1).astype(np.int64, copy=False)
        if arr.size == 0:
            return np.zeros(nteams, dtype=float)
        c = np.bincount(arr, minlength=nteams).astype(float)
        return 100.0 * c / arr.size

    p_semis_sim = pct_counts(np.concatenate([seed1, seed2, q1_w, q2_w]), n)
    p_final_sim = pct_counts(np.concatenate([s1_w, s2_w]), n)
    p_champ_sim = pct_counts(champ, n)

    exp_final_wins = W_all.mean(axis=0)
    exp_final_pf = PF_all.mean(axis=0)
    avg_seed = seed_numbers.mean(axis=0)

    W_int = np.rint(W_all).astype(int)
    W_int = np.clip(W_int, 0, 14)
    win_prob_mat = np.zeros((n, 15), dtype=float)
    for k in range(0, 15):
        mask = (W_int == k).astype(np.int32)
        counts_k = mask.sum(axis=0).astype(float)
        win_prob_mat[:, k] = 100.0 * counts_k / float(N_SIMS)

    seed_mat = np.zeros((n, n), dtype=float)
    for pos in range(n):
        teams_at_pos = seeds_idx[:, pos]
        seed_mat[:, pos] = np.bincount(teams_at_pos, minlength=n) / N_SIMS * 100.0
    seed_df = pd.DataFrame(seed_mat, index=idx2mgr, columns=list(range(1, n + 1)))

    s_index = pd.Index(idx2mgr, name="manager")
    series_pack = {
        "Avg_Seed": pd.Series(avg_seed, index=s_index),
        "Exp_Final_Wins": pd.Series(exp_final_wins, index=s_index),
        "Exp_Final_PF": pd.Series(exp_final_pf, index=s_index),
        "P_Semis_SIM": pd.Series(p_semis_sim, index=s_index),
        "P_Final": pd.Series(p_final_sim, index=s_index),
        "P_Champ": pd.Series(p_champ_sim, index=s_index),
    }

    win_cols = [f"x{k}_win" for k in range(15)]
    win_df = pd.DataFrame(win_prob_mat, index=idx2mgr, columns=win_cols)

    bye_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    bye_mat[np.arange(N_SIMS), seed1] = 1
    bye_mat[np.arange(N_SIMS), seed2] = 1
    no_bye_mat = 1 - bye_mat

    qfwin_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    qfwin_mat[np.arange(N_SIMS), q1_w] = 1
    qfwin_mat[np.arange(N_SIMS), q2_w] = 1

    sfwin_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    sfwin_mat[np.arange(N_SIMS), s1_w] = 1
    sfwin_mat[np.arange(N_SIMS), s2_w] = 1

    champ_mat = np.zeros((N_SIMS, n), dtype=np.int8)
    champ_mat[np.arange(N_SIMS), champ] = 1

    no_bye_counts = no_bye_mat.sum(axis=0).astype(float)
    qfwin_and_no_bye = (qfwin_mat & no_bye_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        qf_win_given_no_bye = np.where(no_bye_counts > 0,
                                       100.0 * qfwin_and_no_bye / no_bye_counts,
                                       0.0)
    series_pack["P_QFWin_Given_NoBye_SIM"] = pd.Series(qf_win_given_no_bye, index=s_index)

    bye_counts = bye_mat.sum(axis=0).astype(float)
    sfwin_and_bye = (sfwin_mat & bye_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sf_win_given_bye = np.where(bye_counts > 0,
                                    100.0 * sfwin_and_bye / bye_counts,
                                    0.0)
    series_pack["P_SFWin_Given_Bye_SIM"] = pd.Series(sf_win_given_bye, index=s_index)

    qfwin_counts = qfwin_mat.sum(axis=0).astype(float)
    sfwin_and_qf = (sfwin_mat & qfwin_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sf_win_given_nonbye = np.where(qfwin_counts > 0,
                                       100.0 * sfwin_and_qf / qfwin_counts,
                                       0.0)
    series_pack["P_SFWin_Given_NonBye_SIM"] = pd.Series(sf_win_given_nonbye, index=s_index)

    final_counts = sfwin_mat.sum(axis=0).astype(float)
    champ_and_final = (champ_mat & sfwin_mat).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        win_final_given_final = np.where(final_counts > 0,
                                         100.0 * champ_and_final / final_counts,
                                         0.0)
    series_pack["P_WinFinal_Given_Final_SIM"] = pd.Series(win_final_given_final, index=s_index)

    return series_pack, seed_df, win_df


# -------------------------
# History helpers
# -------------------------
def _gaussian_kernel(d2): return np.exp(-0.5 * d2)


def empirical_kernel_seed_dist(played_raw, week, history_snapshots_df, n_teams,
                               h_W=0.9, h_L=0.9, h_PF=15.0, h_week=0.9,
                               prior_strength=3.0):
    if played_raw.empty or history_snapshots_df.empty:
        return pd.DataFrame()

    wins_w, pts_w = wins_points_to_date(played_raw)
    gp = (pd.concat([
        played_raw[["manager", "year", "week"]].rename(columns={"manager": "name"}),
        played_raw[["opponent", "year", "week"]].rename(columns={"opponent": "name"})
    ])
          .drop_duplicates()
          .groupby("name")["week"].nunique())
    mgrs = sorted(set(wins_w.index) | set(gp.index) | set(pts_w.index))
    if not mgrs:
        return pd.DataFrame()

    W = wins_w.reindex(mgrs).fillna(0.0)
    PF = pts_w.reindex(mgrs).fillna(0.0)
    GP = gp.reindex(mgrs).fillna(0).astype(int)
    L = (GP - W).clip(lower=0).astype(int)
    pf_pct = 100.0 * PF.rank(pct=True)

    H = history_snapshots_df.dropna(subset=["final_seed"]).copy()
    seeds = np.arange(1, n_teams + 1)
    cols = list(seeds)
    out = pd.DataFrame(0.0, index=mgrs, columns=cols)

    alpha0 = np.ones(n_teams) * (prior_strength / n_teams)
    H_seed = H["final_seed"].to_numpy()

    for m in mgrs:
        xW, xL, xPF, xwk = float(W.loc[m]), float(L.loc[m]), float(pf_pct.loc[m]), float(week)
        dW = (H["W"].to_numpy() - xW) / max(1e-6, h_W)
        dL = (H["L"].to_numpy() - xL) / max(1e-6, h_L)
        dPF = (H["PF_pct"].to_numpy() - xPF) / max(1e-6, h_PF)
        dWK = (H["week"].to_numpy() - xwk) / max(1e-6, h_week)
        d2 = dW * dW + dL * dL + dPF * dPF + dWK * dWK
        w = _gaussian_kernel(d2)

        counts = np.zeros(n_teams)
        for k in range(1, n_teams + 1):
            counts[k - 1] = float(w[(H_seed == k)].sum())
        probs = (counts + alpha0) / (counts.sum() + alpha0.sum() + 1e-12)
        out.loc[m, cols] = probs
    return out


def history_snapshots(all_games, playoff_slots=PLAYOFF_SLOTS):
    rows = []
    seasons = sorted(all_games["year"].dropna().unique().astype(int))
    for yr in seasons:
        reg = all_games[(all_games["year"] == yr) & (all_games["is_playoffs"] == 0)]
        if reg.empty:
            continue
        wins_f, pts_f = wins_points_to_date(reg)
        final_table = rank_and_seed(wins_f, pts_f, playoff_slots, BYE_SLOTS, played_raw=reg)
        final_seed_map = dict(zip(final_table["manager"], final_table["seed"]))
        made = set(final_table.loc[final_table["made_playoffs"], "manager"])

        weeks = sorted(reg["week"].dropna().unique().astype(int))
        for w in weeks:
            played = reg[reg["week"] <= w]
            wins_w, pts_w = wins_points_to_date(played)
            gp = (pd.concat([
                played[["manager", "year", "week"]].rename(columns={"manager": "name"}),
                played[["opponent", "year", "week"]].rename(columns={"opponent": "name"})
            ])
                  .drop_duplicates()
                  .groupby("name")["week"].nunique())
            mgrs = sorted(set(wins_w.index) | set(gp.index) | set(pts_w.index))
            if not mgrs:
                continue
            W = wins_w.reindex(mgrs).fillna(0.0)
            PF = pts_w.reindex(mgrs).fillna(0.0)
            GP = gp.reindex(mgrs).fillna(0).astype(int)
            L = (GP - W).clip(lower=0).astype(int)
            pf_pct = 100.0 * PF.rank(pct=True)
            for m in mgrs:
                rows.append({
                    "year": yr,
                    "week": int(w),
                    "manager": m,
                    "W": float(W.loc[m]),
                    "L": float(L.loc[m]),
                    "PF_pct": float(pf_pct.loc[m]),
                    "made_playoffs": 1.0 if m in made else 0.0,
                    "final_seed": final_seed_map.get(m, np.nan)
                })
    return pd.DataFrame(rows)


# -------------------------
# Normalization helpers
# -------------------------
def normalize_seed_matrix_to_100(seed_df_full):
    if seed_df_full is None or seed_df_full.empty:
        return seed_df_full
    seed_df = seed_df_full.copy()
    for col in seed_df.columns:
        col_sum = float(seed_df[col].sum())
        seed_df[col] = seed_df[col] * (100.0 / col_sum) if col_sum > 0 else 0.0
    return seed_df


def p_playoffs_from_seeds(seed_df, slots=PLAYOFF_SLOTS):
    if seed_df is None or seed_df.empty:
        return pd.Series(dtype=float)
    cols = [c for c in seed_df.columns if isinstance(c, int) and 1 <= c <= slots]
    return seed_df[cols].sum(axis=1)


def p_bye_from_seeds(seed_df, bye_slots=BYE_SLOTS):
    if seed_df is None or seed_df.empty:
        return pd.Series(dtype=float)
    cols = [c for c in seed_df.columns if isinstance(c, int) and 1 <= c <= bye_slots]
    return seed_df[cols].sum(axis=1)


def normalize_power_rating(power_series, inflation_rate):
    """Normalize power rating by inflation rate."""
    if pd.isna(inflation_rate) or inflation_rate == 0:
        return power_series
    return power_series / float(inflation_rate)


# -------------------------
# Core calculators
# -------------------------
def calc_regular_week_outputs(df_season, df_sched, season, week, history_df, inflation_rate=None):
    df_to_date = df_season[df_season["week"] <= week].copy()
    reg_to_date = df_to_date[df_to_date["is_playoffs"] == 0].copy()
    played_raw = reg_to_date.copy()

    future_canon = future_regular_from_schedule(df_sched, season, week)
    simulate_future_reg = not future_canon.empty

    sigma_floor_dynamic = SIGMA_FLOOR_MIN if week >= 3 else max(SIGMA_FLOOR_MIN, 14)
    mu_hat, sigma_hat, samples_by_team, league_mu, sigma_floor = build_team_models(
        df_to_date, season, week, HALF_LIFE_WEEKS, SHRINK_K, sigma_floor_dynamic,
        boundary_penalty=0.05, prior_w_cap=2.0
    )
    ensure_params_for_future(mu_hat, sigma_hat, samples_by_team, future_canon, league_mu, sigma_floor)

    power_s = compute_power_ratings(mu_hat, samples_by_team, bootstrap_min=4)

    # Normalize power ratings by inflation rate if provided
    if inflation_rate is not None and pd.notna(inflation_rate) and inflation_rate != 0:
        power_s = normalize_power_rating(power_s, inflation_rate)

    series_pack, seed_dist_sim, win_df = _vectorized_regular_and_bracket(
        reg_to_date, future_canon, mu_hat, sigma_hat, season, week
    )

    n_teams_all = seed_dist_sim.shape[1]
    if simulate_future_reg:
        hist_seed_dist = empirical_kernel_seed_dist(
            played_raw, week, history_df, n_teams_all
        ).reindex(index=seed_dist_sim.index, columns=seed_dist_sim.columns, fill_value=0.0) * 100.0

        weeks_played = int(played_raw["week"].nunique())
        sim_w = 0.25 + 0.75 * min(1.0, max(0.0, (weeks_played - 1) / 4.0))
        blended_seed = sim_w * seed_dist_sim + (1 - sim_w) * hist_seed_dist
        blended_seed_norm = normalize_seed_matrix_to_100(blended_seed)
    else:
        blended_seed_norm = normalize_seed_matrix_to_100(seed_dist_sim)

    odds = pd.concat([
        series_pack["Exp_Final_Wins"].rename("Exp_Final_Wins"),
        series_pack["Exp_Final_PF"].rename("Exp_Final_PF"),
        series_pack["Avg_Seed"].rename("Avg_Seed"),
    ], axis=1)

    odds["P_Playoffs"] = p_playoffs_from_seeds(blended_seed_norm, PLAYOFF_SLOTS)
    odds["P_Bye"] = p_bye_from_seeds(blended_seed_norm, BYE_SLOTS)

    qf_cnd = series_pack["P_QFWin_Given_NoBye_SIM"]
    odds["P_Semis"] = odds["P_Bye"] + (100.0 - odds["P_Bye"]) * (qf_cnd / 100.0)
    odds["P_Semis"] = np.minimum(100.0, np.maximum(odds["P_Semis"], odds["P_Bye"]))

    sf_bye = series_pack["P_SFWin_Given_Bye_SIM"]
    sf_nonbye = series_pack["P_SFWin_Given_NonBye_SIM"]
    semis_nonbye = np.maximum(0.0, odds["P_Semis"] - odds["P_Bye"])
    odds["P_Final"] = (odds["P_Bye"] * sf_bye + semis_nonbye * sf_nonbye) / 100.0

    win_final = series_pack["P_WinFinal_Given_Final_SIM"]
    odds["P_Champ"] = odds["P_Final"] * (win_final / 100.0)

    odds["P_Final"] = np.minimum(odds["P_Semis"], odds["P_Final"])
    odds["P_Champ"] = np.minimum(odds["P_Final"], odds["P_Champ"])
    odds[["P_Semis", "P_Final", "P_Champ"]] = odds[["P_Semis", "P_Final", "P_Champ"]].clip(lower=0.0, upper=100.0)

    if not power_s.empty:
        odds["Power_Rating"] = power_s.reindex(odds.index)

    return odds, blended_seed_norm, win_df


def calc_playoff_week_outputs(df_season, df_sched, season, week, inflation_rate=None):
    rng = get_rng(season, week)
    df_to_date = df_season[df_season["week"] <= week].copy()

    reg = df_to_date[df_to_date["is_playoffs"] == 0].copy()
    wins_to_date, pts_to_date = wins_points_to_date(reg)
    seeds = rank_and_seed(wins_to_date, pts_to_date, PLAYOFF_SLOTS, BYE_SLOTS, played_raw=reg)
    top6 = seeds.loc[seeds["made_playoffs"], "manager"].tolist()

    last_reg_week = schedules_last_regular_week(df_sched, season)
    if last_reg_week is None:
        last_reg_week = reg["week"].max() if not reg.empty else week
    mu_hat, sigma_hat, samples_by_team, league_mu, sigma_floor = build_team_models(
        df_to_date, season, last_reg_week, HALF_LIFE_WEEKS, SHRINK_K, SIGMA_FLOOR_MIN,
        boundary_penalty=0.05, prior_w_cap=2.0
    )

    power_s = compute_power_ratings(mu_hat, samples_by_team, bootstrap_min=4)

    # Normalize power ratings by inflation rate if provided
    if inflation_rate is not None and pd.notna(inflation_rate) and inflation_rate != 0:
        power_s = normalize_power_rating(power_s, inflation_rate)

    def actual_winner_by_round(round_col, teamA, teamB):
        if (round_col not in df_to_date.columns) or (teamA is None) or (teamB is None):
            return None
        g = df_to_date[(df_to_date.get(round_col, 0) == 1) &
                       (df_to_date["manager"].isin([teamA, teamB])) &
                       (df_to_date["opponent"].isin([teamA, teamB]))]
        if "is_consolation" in df_to_date.columns:
            g = g[g["is_consolation"] == 0]
        if g.empty:
            return None
        weeks_a = set(g.loc[g["manager"] == teamA, "week"].dropna().astype(int).tolist())
        weeks_b = set(g.loc[g["manager"] == teamB, "week"].dropna().astype(int).tolist())
        shared_weeks = sorted([w for w in weeks_a.intersection(weeks_b) if w <= int(week)])
        if not shared_weeks:
            return None
        w_star = shared_weeks[-1]
        gw = g[g["week"] == w_star].copy()
        pts = gw.groupby("manager", as_index=True)["team_points"].mean()
        if set(pts.index) != {teamA, teamB}: return None
        if pd.isna(pts.get(teamA)) or pd.isna(pts.get(teamB)): return None
        if pts[teamA] > pts[teamB]: return teamA
        if pts[teamB] > pts[teamA]: return teamB
        return min(teamA, teamB)

    def get_round_participants_with_scores(df_data, round_col):
        if round_col not in df_data.columns:
            return [], {}
        g = df_data[(df_data.get(round_col, 0) == 1)]
        if "is_consolation" in df_data.columns:
            g = g[g["is_consolation"] == 0]
        parts = sorted(g["manager"].dropna().unique().tolist())
        has_pts = g.groupby("manager")["team_points"].apply(lambda s: s.notna().any()).to_dict()
        return parts, has_pts

    if len(top6) != 6:
        return pd.DataFrame(), pd.DataFrame(), seeds
    byes, qtrs = (top6[:BYE_SLOTS], [(top6[3], top6[4]), (top6[2], top6[5])])

    qf_col = ROUND_COLS.get("qf", "quarterfinal")
    sf_col = ROUND_COLS.get("sf", "semifinal")
    fn_col = ROUND_COLS.get("fn", "championship")

    q1_actual = actual_winner_by_round(qf_col, qtrs[0][0], qtrs[0][1])
    q2_actual = actual_winner_by_round(qf_col, qtrs[1][0], qtrs[1][1])

    def semi_opponents(q1_w, q2_w):
        winners = [w for w in [q1_w, q2_w] if w is not None]
        if len(winners) < 2:
            return None, None
        if not BRACKET_RESEED:
            return (top6[0], q1_w), (top6[1], q2_w)
        seeds_map = dict(zip(seeds["manager"], seeds["seed"]))
        winners_sorted = sorted(winners, key=lambda m: seeds_map[m])
        best = winners_sorted[0]
        worst = winners_sorted[-1]
        return (top6[0], worst), (top6[1], best)

    s1_pair, s2_pair = semi_opponents(q1_actual, q2_actual)
    s1_actual = actual_winner_by_round(sf_col, *(s1_pair or (None, None))) if s1_pair else None
    s2_actual = actual_winner_by_round(sf_col, *(s2_pair or (None, None))) if s2_pair else None

    finalists = None
    if (s1_actual is not None) and (s2_actual is not None):
        finalists = (s1_actual, s2_actual)
    else:
        fn_parts, fn_has_pts = get_round_participants_with_scores(df_to_date, fn_col)
        if (len(fn_parts) == 2):
            if any(fn_has_pts.get(p, False) for p in fn_parts):
                known_winners = {w for w in [s1_actual, s2_actual] if w is not None}
                if not known_winners or known_winners.issubset(set(fn_parts)):
                    finalists = (fn_parts[0], fn_parts[1])

    champ_locked = None
    if finalists is not None:
        champ_locked = actual_winner_by_round(fn_col, finalists[0], finalists[1])

    sims, playoff_r2, playoff_r3, champions = [], [], [], []
    for _ in range(N_SIMS):
        q1_w = q1_actual if q1_actual else _sim_game(qtrs[0][0], qtrs[0][1], rng, mu_hat, sigma_hat, samples_by_team)
        q2_w = q2_actual if q2_actual else _sim_game(qtrs[1][0], qtrs[1][1], rng, mu_hat, sigma_hat, samples_by_team)
        semi1, semi2 = semi_opponents(q1_w, q2_w)
        if semi1 is None:
            continue
        s1_w = s1_actual if s1_actual else _sim_game(semi1[0], semi1[1], rng, mu_hat, sigma_hat, samples_by_team)
        s2_w = s2_actual if s2_actual else _sim_game(semi2[0], semi2[1], rng, mu_hat, sigma_hat, samples_by_team)
        if finalists is not None:
            s1_w, s2_w = finalists[0], finalists[1]
        champ = champ_locked if champ_locked else _sim_game(s1_w, s2_w, rng, mu_hat, sigma_hat, samples_by_team)
        playoff_r2.extend([semi1[0], semi2[0], q1_w, q2_w])
        playoff_r3.extend([s1_w, s2_w])
        champions.append(champ)
        sims.append(seeds)

    if not sims:
        return pd.DataFrame(), pd.DataFrame(), seeds

    tall = pd.concat(sims, ignore_index=True)

    odds = (tall.groupby("manager")
            .agg(Exp_Final_Wins=("W", "mean"),
                 Exp_Final_PF=("PF", "mean")))
    odds["Avg_Seed"] = tall.groupby("manager")["seed"].mean()
    idx_mgrs = odds.index.tolist()
    odds["P_Semis"] = [playoff_r2.count(m) / N_SIMS * 100 for m in idx_mgrs]
    odds["P_Final"] = [playoff_r3.count(m) / N_SIMS * 100 for m in idx_mgrs]
    odds["P_Champ"] = [champions.count(m) / N_SIMS * 100 for m in idx_mgrs]

    if finalists is not None:
        finals_set = set(finalists)
        non_finalists = [m for m in idx_mgrs if m not in finals_set]
        if non_finalists:
            odds.loc[non_finalists, ["P_Final", "P_Champ"]] = 0.0
        odds.loc[list(finals_set), "P_Final"] = 100.0
    if 'champ_locked' in locals() and champ_locked:
        odds["P_Champ"] = 0.0
        if champ_locked in odds.index:
            odds.at[champ_locked, "P_Champ"] = 100.0

    team_count = seeds.shape[0]
    seed_dist = (tall.pivot_table(index="manager", columns="seed", values="W",
                                  aggfunc="size", fill_value=0)
                 .div(len(sims)) * 100.0)
    all_cols = list(range(1, team_count + 1))
    seed_dist = seed_dist.reindex(columns=all_cols, fill_value=0.0)
    seed_dist_norm = normalize_seed_matrix_to_100(seed_dist)

    top6_set = set(top6)
    odds["P_Playoffs"] = [100.0 if m in top6_set else 0.0 for m in idx_mgrs]
    odds["P_Bye"] = [100.0 if (m in top6_set and bool(seeds.loc[seeds['manager'] == m, 'bye'].iloc[0])) else 0.0
                     for m in idx_mgrs]

    if "Power_Rating" not in odds.columns:
        power_vals = power_s
        odds["Power_Rating"] = power_vals.reindex(odds.index)

    print(f"[PO diag] season={season} week={week}")

    return odds, seed_dist_norm, seeds


# -------------------------
# Main processing function
# -------------------------
def process_parquet_files():
    print(f"Loading data from: {DATA_DIR}")

    # Load parquet files
    df_matches = df_from_md_or_parquet("matchup", MATCHUP_PATH)
    df_sched = df_from_md_or_parquet("schedule", SCHEDULE_PATH)

    print(f"Loaded {len(df_matches)} matchup records")
    print(f"Loaded {len(df_sched)} schedule records")

    # Clean blanks
    for df in (df_matches, df_sched):
        df.replace("", np.nan, inplace=True)

    # Coerce numeric columns
    numeric_cols_matches = ["week", "year", "is_playoffs", "is_consolation", "inflation_rate"]
    for col in numeric_cols_matches:
        if col in df_matches.columns:
            df_matches[col] = pd.to_numeric(df_matches[col], errors="coerce")

    for col in ["week", "year", "is_playoffs", "is_consolation"]:
        if col in df_sched.columns:
            df_sched[col] = pd.to_numeric(df_sched[col], errors="coerce")

    # Ensure target columns exist
    for col in TARGET_COLS:
        if col not in df_matches.columns:
            df_matches[col] = np.nan

    # Monotonic playoff flags
    df_matches = enforce_playoff_monotonicity(df_matches)
    df_sched = enforce_playoff_monotonicity(df_sched)

    hist_df = history_snapshots(df_matches, PLAYOFF_SLOTS)

    seasons = sorted(df_matches["year"].dropna().unique().astype(int))
    df_all = df_matches.copy()

    for season in seasons:
        print(f"\nProcessing season {season}...")
        df_season = df_matches[df_matches["year"] == season].copy()
        if df_season.empty:
            continue

        # Get inflation rate for this season
        season_inflation = df_season["inflation_rate"].dropna()
        inflation_rate = season_inflation.iloc[0] if not season_inflation.empty else None

        reg_weeks = sorted(df_season[df_season["is_playoffs"] == 0]["week"].dropna().unique().astype(int))
        po_weeks = sorted(df_season[(df_season["is_playoffs"] == 1) & (df_season["is_consolation"] == 0)][
                              "week"].dropna().unique().astype(int))

        # Process regular season weeks
        for w in reg_weeks:
            print(f"  Processing regular season week {w}...")
            mask_week = (df_all["year"] == season) & (df_all["week"] == w) & (df_all["is_consolation"] == 0)
            if not mask_week.any():
                continue

            odds_df, seed_df, win_df = calc_regular_week_outputs(
                df_season, df_sched, season, w, hist_df, inflation_rate
            )

            for idx in df_all[mask_week].index:
                m = df_all.at[idx, "manager"]
                if m not in odds_df.index:
                    continue

                def set_val(col, val):
                    if col in df_all.columns and pd.notna(val):
                        df_all.at[idx, col] = round(float(val), 2) if isinstance(val,
                                                                                 (float, np.floating, int)) else val

                set_val("avg_seed", odds_df.at[m, "Avg_Seed"])
                set_val("exp_final_wins", odds_df.at[m, "Exp_Final_Wins"])
                set_val("exp_final_pf", odds_df.at[m, "Exp_Final_PF"])
                set_val("p_semis", odds_df.at[m, "P_Semis"])
                set_val("p_final", odds_df.at[m, "P_Final"])
                set_val("p_champ", odds_df.at[m, "P_Champ"])
                set_val("p_playoffs", odds_df.at[m, "P_Playoffs"])
                set_val("p_bye", odds_df.at[m, "P_Bye"])

                # Power rating is already normalized in the odds_df
                if "Power_Rating" in odds_df.columns:
                    set_val("power_rating", odds_df.at[m, "Power_Rating"])

                if not seed_df.empty and m in seed_df.index:
                    for k in range(1, 11):
                        if k in seed_df.columns:
                            set_val(f"x{k}_seed", seed_df.loc[m, k])

                if (win_df is not None) and (m in win_df.index):
                    for k in range(0, 15):
                        col = f"x{k}_win"
                        if col in win_df.columns:
                            set_val(col, win_df.at[m, col])

        # Process playoff weeks
        for w in po_weeks:
            print(f"  Processing playoff week {w}...")
            mask_week = (
                    (df_all["year"] == season)
                    & (df_all["week"] == w)
                    & (df_all["is_playoffs"] == 1)
                    & (df_all["is_consolation"] == 0)
            )
            if not mask_week.any():
                continue

            odds_df, seed_df, seeds = calc_playoff_week_outputs(
                df_season, df_sched, season, w, inflation_rate
            )

            for idx in df_all[mask_week].index:
                m = df_all.at[idx, "manager"]
                if m not in odds_df.index:
                    continue

                def set_val(col, val):
                    if col in df_all.columns and pd.notna(val):
                        df_all.at[idx, col] = round(float(val), 2) if isinstance(val,
                                                                                 (float, np.floating, int)) else val

                set_val("avg_seed", odds_df.at[m, "Avg_Seed"])
                set_val("exp_final_wins", odds_df.at[m, "Exp_Final_Wins"])
                set_val("exp_final_pf", odds_df.at[m, "Exp_Final_PF"])
                set_val("p_semis", odds_df.at[m, "P_Semis"])
                set_val("p_final", odds_df.at[m, "P_Final"])
                set_val("p_champ", odds_df.at[m, "P_Champ"])
                set_val("p_playoffs", odds_df.at[m, "P_Playoffs"])
                set_val("p_bye", odds_df.at[m, "P_Bye"])

                # Power rating is already normalized in the odds_df
                if "Power_Rating" in odds_df.columns:
                    set_val("power_rating", odds_df.at[m, "Power_Rating"])

                if not seed_df.empty and m in seed_df.index:
                    for k in range(1, 11):
                        if k in seed_df.columns:
                            set_val(f"x{k}_seed", seed_df.loc[m, k])

    # Save outputs
    print(f"\nSaving results...")
    print(f"  Parquet: {OUTPUT_MATCHUP_PARQUET}")
    print(f"  CSV: {OUTPUT_MATCHUP_CSV}")

    df_all.to_parquet(OUTPUT_MATCHUP_PARQUET, index=False)
    df_all.to_csv(OUTPUT_MATCHUP_CSV, index=False)

    print("\nProcessing complete!")
    print(f"Updated {len(df_all)} records with playoff odds")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    process_parquet_files()