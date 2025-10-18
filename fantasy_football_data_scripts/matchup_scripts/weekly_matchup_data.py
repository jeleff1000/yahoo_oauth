#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import re
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import yahoo_fantasy_api as yfa

# Add parent directory to path for oauth_utils import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from oauth_utils import ensure_oauth_path, create_oauth2
except Exception:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from oauth_utils import ensure_oauth_path, create_oauth2

# =============================================================================
# Safe paths
# =============================================================================
try:
    THIS_FILE = __file__
    BASE_DIR = os.path.dirname(os.path.abspath(THIS_FILE))
except NameError:
    BASE_DIR = os.getcwd()

OUT_DIR   = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'fantasy_football_data', 'matchup_data'))
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# OAuth discovery: centralized helper
# =============================================================================
try:
    oauth_path = ensure_oauth_path()
    oauth = create_oauth2(oauth_path)
except SystemExit:
    raise
except Exception as e:
    raise SystemExit(f"OAuth initialization failed: {e}")

gm = yfa.Game(oauth, 'nfl')

# =============================================================================
# Inputs
# =============================================================================
parser = argparse.ArgumentParser(description="Fetch Yahoo Fantasy Football weekly matchup data")
parser.add_argument('--year', type=int, default=None, help='Season year (0 for all years)')
parser.add_argument('--week', type=int, default=None, help='Week number (0 for all weeks)')
args = parser.parse_args()

# Use CLI args if provided, otherwise prompt
if args.year is not None:
    year_input = args.year
else:
    year_input = int(input("select the year to get data for: ").strip())

if args.week is not None:
    week_input = str(args.week)
else:
    week_input = input("Select the week to get data for, (type 0 for the whole year): ").strip()

week_input = week_input.strip()

# =============================================================================
# GPA scale
# =============================================================================
gpa_scale = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "F": 0.0, "F-": 0.0,
}

# =============================================================================
# Helpers
# =============================================================================
def norm_manager(nickname: str) -> str:
    if not nickname:
        return "N/A"
    s = str(nickname).strip()
    if s == "--hidden--":
        return "Ilan"
    return s

def safe_float(text, default=np.nan):
    try:
        if text is None:
            return default
        s = str(text).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def _first_text(node: ET.Element, paths: list[str]) -> str:
    for p in paths:
        v = node.findtext(p)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""

def _first_float(node: ET.Element, paths: list[str]) -> float:
    for p in paths:
        v = node.findtext(p)
        fv = safe_float(v, np.nan)
        if not (isinstance(fv, float) and np.isnan(fv)):
            return fv
    return np.nan

def get_league_for_year(y: int):
    league_ids = gm.league_ids(year=y)
    if not league_ids:
        raise RuntimeError(f"No leagues found for {y}")
    yearid = league_ids[-1]
    league = gm.to_league(yearid)
    return yearid, league

def load_league_settings(league) -> dict:
    try:
        return league.settings()
    except Exception:
        return {}

def derive_playoff_structure(settings: dict) -> dict:
    start_week = int(settings.get('start_week') or 1)
    end_week = int(settings.get('end_week') or 17)
    playoff_start_week = int(settings.get('playoff_start_week') or max(end_week - 2, start_week))
    matchup_len = int(settings.get('playoff_matchup_length') or 1)
    num_teams = int(settings.get('num_playoff_teams') or 6)

    if num_teams <= 4:
        rounds = 2
    elif num_teams in (6, 8):
        rounds = 3
    else:
        rounds = 3 if num_teams >= 6 else 2

    qf_weeks, sf_weeks, final_weeks = [], [], []
    wk = playoff_start_week
    if rounds == 3:
        qf_weeks = list(range(wk, wk + matchup_len)); wk += matchup_len
        sf_weeks = list(range(wk, wk + matchup_len)); wk += matchup_len
        final_weeks = list(range(wk, wk + matchup_len))
    else:
        sf_weeks = list(range(wk, wk + matchup_len)); wk += matchup_len
        final_weeks = list(range(wk, wk + matchup_len))

    champion_week = final_weeks[-1] if final_weeks else end_week
    return {
        'start_week': start_week, 'end_week': end_week,
        'playoff_start_week': playoff_start_week, 'matchup_len': matchup_len,
        'num_playoff_teams': num_teams, 'rounds': rounds,
        'qf_weeks': qf_weeks, 'sf_weeks': sf_weeks, 'final_weeks': final_weeks,
        'champion_week': champion_week,
    }

def get_weeks_to_fetch(league, year: int, week_input_str: str, settings: dict) -> list[int]:
    s = derive_playoff_structure(settings)
    start_week, end_week = s['start_week'], s['end_week']
    if week_input_str != "0":
        return [int(week_input_str)]
    try:
        cw_attr = getattr(league, 'current_week', None)
        current_week = int(cw_attr() if callable(cw_attr) else cw_attr) if cw_attr is not None else None
    except Exception:
        current_week = None
    if (year == datetime.now().year) and current_week and current_week > start_week:
        last = min(end_week, current_week - 1)
        return list(range(start_week, last + 1))
    return list(range(start_week, end_week + 1))

# =============================================================================
# Core fetch/parsing
# =============================================================================
def parse_matchups_for_week(yearid: str, year_val: int, week: int) -> list[dict]:
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{yearid}/scoreboard;week={week}"
    print("Making API Call:", url)
    resp = oauth.session.get(url); resp.raise_for_status()

    xmlstring = re.sub(r' xmlns=\"[^\"]+\"', '', resp.text, count=1)
    root = ET.fromstring(xmlstring)

    # Skip incomplete weeks (all totals zero)
    try:
        totals = [safe_float(tp.text, 0.0) for tp in root.findall(".//team_points/total")]
        if totals and all((p or 0.0) == 0.0 for p in totals):
            return []
    except Exception:
        pass

    def extract_team(team_node: ET.Element) -> dict:
        nickname = (
            team_node.findtext(".//managers/manager/nickname")
            or team_node.findtext(".//managers/manager/name")
            or team_node.findtext(".//managers/manager/guid")
            or ""
        )
        manager = norm_manager(nickname)
        team_name = team_node.findtext("name") or manager

        points = safe_float(team_node.findtext("team_points/total"), 0.0)
        projected = safe_float(team_node.findtext("team_projected_points/total"), np.nan)

        grade = _first_text(team_node, [
            ".//matchup_grade/grade",
            ".//matchup_grades/matchup_grade/grade",
            "matchup_grade/grade",
            "grade",
        ])
        gpa = gpa_scale.get(grade, np.nan)

        url_ = team_node.findtext("url") or ""
        image_url = team_node.findtext("team_logos/team_logo/url") or ""
        division_id = team_node.findtext("division_id") or ""
        waiver_priority = safe_float(team_node.findtext("waiver_priority"), np.nan)
        faab_balance = safe_float(team_node.findtext("faab_balance"), np.nan)
        number_of_moves = safe_float(team_node.findtext("number_of_moves"), np.nan)
        number_of_trades = safe_float(team_node.findtext("number_of_trades"), np.nan)
        coverage_value = safe_float(team_node.findtext("coverage_value"), np.nan)
        value = safe_float(team_node.findtext("value"), np.nan)
        has_draft_grade = (team_node.findtext("has_draft_grade") or "").strip()

        felo_score = _first_float(team_node, ["felo_score", ".//team_standings/felo_score"])
        felo_tier = _first_text(team_node, ["felo_tier", ".//team_standings/felo_tier"])
        win_probability = safe_float(team_node.findtext("win_probability"), np.nan)

        auction_budget_total = _first_float(team_node, ["draft_results/auction_budget/total", "auction_budget_total"])
        auction_budget_spent = _first_float(team_node, ["draft_results/auction_budget/spent", "auction_budget_spent"])

        return {
            'manager': manager,
            'team_name': team_name,
            'team_points': points,
            'team_projected_points': projected,
            'grade': grade,
            'gpa': gpa,
            'url': url_,
            'image_url': image_url,
            'division_id': division_id,
            'waiver_priority': waiver_priority,
            'faab_balance': faab_balance,
            'number_of_moves': number_of_moves,
            'number_of_trades': number_of_trades,
            'coverage_value': coverage_value,
            'value': value,
            'has_draft_grade': has_draft_grade,
            'auction_budget_total': auction_budget_total,
            'auction_budget_spent': auction_budget_spent,
            'felo_score': felo_score,
            'felo_tier': felo_tier,
            'win_probability': win_probability,
        }

    rows: list[dict] = []
    for matchup in root.findall(".//matchup"):
        wk_node = matchup.find("week")
        if wk_node is None or not wk_node.text:
            continue
        week_num = int(wk_node.text)

        is_playoffs = int(matchup.findtext("is_playoffs", default="0") or "0")
        is_consolation = int(matchup.findtext("is_consolation", default="0") or "0")
        week_start = matchup.findtext("week_start", default="") or ""
        week_end = matchup.findtext("week_end", default="") or ""
        matchup_recap_url = matchup.findtext("matchup_recap_url", default="") or ""
        matchup_recap_title = matchup.findtext("matchup_recap_title", default="") or ""

        teams = matchup.findall(".//teams/team")
        if len(teams) != 2:
            teams = matchup.findall(".//team")
        if len(teams) != 2:
            continue

        t1 = extract_team(teams[0])
        t2 = extract_team(teams[1])

        def build_row(a: dict, b: dict) -> dict:
            a_points = safe_float(a['team_points'], 0.0)
            b_points = safe_float(b['team_points'], 0.0)
            margin = a_points - b_points
            total_matchup_score = a_points + b_points
            close_margin = int(abs(margin) <= 10.0)
            win = 1 if a_points > b_points else 0
            loss = 1 if a_points < b_points else 0

            return {
                'week': int(week_num),
                'year': int(year_val),
                'manager': a['manager'],
                'team_name': a['team_name'],
                'team_points': a_points,
                'opponent': b['manager'],
                'opponent_points': b_points,
                'team_projected_points': a['team_projected_points'],
                'opponent_projected_points': b['team_projected_points'],
                'grade': a['grade'],
                'gpa': a['gpa'],
                'week_start': week_start,
                'week_end': week_end,
                'is_playoffs': is_playoffs,
                'is_consolation': is_consolation,
                'matchup_recap_title': matchup_recap_title,
                'matchup_recap_url': matchup_recap_url,
                'url': a['url'],
                'image_url': a['image_url'],
                'division_id': a['division_id'],
                'waiver_priority': a['waiver_priority'],
                'faab_balance': a['faab_balance'],
                'number_of_moves': a['number_of_moves'],
                'number_of_trades': a['number_of_trades'],
                'coverage_value': a['coverage_value'],
                'value': a['value'],
                'has_draft_grade': a['has_draft_grade'],
                'auction_budget_total': a['auction_budget_total'],
                'auction_budget_spent': a['auction_budget_spent'],
                'felo_score': a['felo_score'],
                'felo_tier': a['felo_tier'],
                'win_probability': a['win_probability'],
                'margin': margin,
                'total_matchup_score': total_matchup_score,
                'close_margin': close_margin,
                'win': win,
                'loss': loss,
            }

        rows.append(build_row(t1, t2))
        rows.append(build_row(t2, t1))

    return rows

# =============================================================================
# Data collection
# =============================================================================
all_rows: list[dict] = []

def process_year(y: int):
    yearid, league = get_league_for_year(y)
    settings = load_league_settings(league)
    weeks = get_weeks_to_fetch(league, y, week_input, settings)
    if not weeks:
        return
    for w in weeks:
        try:
            all_rows.extend(parse_matchups_for_week(yearid, y, w))
        except Exception as e:
            print(f"Skipping year {y} week {w} due to error: {e}", file=sys.stderr)

if year_input == 0:
    y = datetime.now().year
    while True:
        try:
            process_year(y); y -= 1
        except Exception:
            break
else:
    process_year(year_input)

if not all_rows:
    print("No completed weeks found or all weeks skipped due to zero scores. Exiting.")
    sys.exit(0)

# =============================================================================
# Build DataFrame
# =============================================================================
df = pd.DataFrame(all_rows)

# --- Projection deltas (opponent_projected_points already in data)
df['proj_score_error'] = df['team_projected_points'] - df['team_points']
df['abs_proj_score_error'] = df['proj_score_error'].abs()
df['above_proj_score'] = (df['team_points'] > df['team_projected_points']).astype(int)
df['below_proj_score'] = (df['team_points'] < df['team_projected_points']).astype(int)

# --- Expected spread & win vs spread
df['expected_spread'] = df['team_projected_points'] - df['opponent_projected_points']
df['expected_odds'] = (1 / (1 + np.exp(-0.0375 * df['expected_spread']))).round(2)
df['win_vs_spread'] = ((df['margin'] > df['expected_spread']) & df['expected_spread'].notna()).astype(int)
df['lose_vs_spread'] = ((df['margin'] < df['expected_spread']) & df['expected_spread'].notna()).astype(int)

# --- Upset/favorite outcomes
df['underdog_wins'] = ((df['team_projected_points'] < df['opponent_projected_points']) & (df['win'] == 1)).astype(int)
df['favorite_losses'] = ((df['team_projected_points'] > df['opponent_projected_points']) & (df['loss'] == 1)).astype(int)

# --- League-weekly aggregates
weekly_stats = (
    df.groupby(['year', 'week'], as_index=False)['team_points']
      .agg(weekly_mean='mean', weekly_median='median')
)
df = df.merge(weekly_stats, on=['year', 'week'], how='left')

# =============================================================================
# Teams-beat metrics & league/median comparisons & projection recordkeeping
# (No groupby.apply; all vectorized)
# =============================================================================

# Count of strictly lower scores within (year, week). Ties don't count as beaten.
# We can reproduce your old logic via ranks: rank(method='min', ascending=True) - 1
# but use transform so it's vectorized.
rank_within_week = (
    df.groupby(['year', 'week'])['team_points']
      .transform(lambda s: s.rank(method='min', ascending=True) - 1)
)
df['teams_beat_this_week'] = rank_within_week.fillna(0).astype(int)

# Opponent teams beaten this week: number of scores strictly less than each row's opponent_points
# To avoid Python loops, compute per (year, week):
df['opponent_teams_beat_this_week'] = 0  # init
for (y, w), g in df.groupby(['year', 'week'], sort=False):
    # Sort the team's scores (ascending)
    tp_sorted = np.sort(pd.to_numeric(g['team_points'], errors='coerce').to_numpy())
    # For each row's opponent_points, count how many team_points are strictly less
    opp = pd.to_numeric(g['opponent_points'], errors='coerce').to_numpy()
    counts = np.searchsorted(tp_sorted, opp, side='left')
    df.loc[g.index, 'opponent_teams_beat_this_week'] = counts.astype(int)

# Projection recordkeeping
df["proj_wins"]   = (df["team_projected_points"] > df["opponent_projected_points"]).astype(int)
df["proj_losses"] = (df["team_projected_points"] < df["opponent_projected_points"]).astype(int)

# League/median comparisons (aliases + flags)
df["league_weekly_mean"]   = df["weekly_mean"]
df["league_weekly_median"] = df["weekly_median"]
df["above_league_median"]  = (df["team_points"] > df["league_weekly_median"]).astype(int)
df["below_league_median"]  = (df["team_points"] < df["league_weekly_median"]).astype(int)

# =============================================================================
# Head-to-head W/L vs each manager & vs each manager's schedule
# =============================================================================
def _mgr_token(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^a-z0-9_]+', '', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or "na"

mgr_all = sorted(pd.unique(pd.concat([df['manager'], df['opponent']]).dropna()))
mgr_tokens = {m: _mgr_token(m) for m in mgr_all}

# Initialize H2H columns (explicit is clearer; order is deterministic)
wl_vs_mgr_cols: dict[str, pd.Series] = {}
for m in mgr_all:
    tok = mgr_tokens[m]
    w_col = f"w_vs_{tok}"
    l_col = f"l_vs_{tok}"
    df[w_col] = 0
    df[l_col] = 0
    wl_vs_mgr_cols[w_col] = df[w_col]
    wl_vs_mgr_cols[l_col] = df[l_col]

# Fill head-to-head W/L by week using vector ops
for (y, w), g in df.groupby(['year', 'week'], sort=False):
    # manager -> points on that week
    scores = g.groupby('manager', sort=False)['team_points'].first()
    # Compare all pairs (mgr vs opp) once and write back rows for that mgr
    # Build arrays for quicker broadcasting
    mgrs = scores.index.to_list()
    vals = scores.to_numpy()

    # For each manager, count how many opponents they outscored / were outscored by
    # Build comparison matrix vals[:,None] vs vals[None,:]
    gt = (vals[:, None] > vals[None, :]).astype(int)
    lt = (vals[:, None] < vals[None, :]).astype(int)
    wins_by_mgr = gt.sum(axis=1)
    losses_by_mgr = lt.sum(axis=1)

    # Assign into rows of that manager for the columns corresponding to each opponent
    for i, mgr in enumerate(mgrs):
        idx = g.index[g['manager'] == mgr]
        # wins vs each specific opponent for this week
        for j, opp in enumerate(mgrs):
            if mgr == opp:
                continue
            tok = mgr_tokens[opp]
            df.loc[idx, f"w_vs_{tok}"] += gt[i, j]
            df.loc[idx, f"l_vs_{tok}"] += lt[i, j]

# W/L vs each manager's schedule (compare to that manager's opponent_points each week)
wl_vs_sched_cols: dict[str, pd.Series] = {}
# Precompute a map: (manager, year, week) -> that manager's opponent_points
opp_points_map = (
    df.set_index(['manager', 'year', 'week'])['opponent_points']
      .to_dict()
)
for m in mgr_all:
    tok = mgr_tokens[m]
    # Align each row's (year, week) to manager m's opponent_points on that (year, week)
    aligned_sched = np.array([opp_points_map.get((m, y, w), np.nan) for y, w in zip(df['year'], df['week'])])
    df[f"w_vs_{tok}_sched"] = (df['team_points'].to_numpy() > aligned_sched).astype(int)
    df[f"l_vs_{tok}_sched"] = (df['team_points'].to_numpy() < aligned_sched).astype(int)
    wl_vs_sched_cols[f"w_vs_{tok}_sched"] = df[f"w_vs_{tok}_sched"]
    wl_vs_sched_cols[f"l_vs_{tok}_sched"] = df[f"l_vs_{tok}_sched"]

# =============================================================================
# Column order
# =============================================================================
KEEP = [
    'week', 'year', 'manager', 'team_name',
    'team_points', 'team_projected_points', 'opponent', 'opponent_points',
    'opponent_projected_points', 'margin', 'total_matchup_score', 'close_margin',
    'weekly_mean', 'weekly_median',
    # league/median comparisons
    'league_weekly_mean', 'league_weekly_median', 'above_league_median', 'below_league_median',
    # core outcomes
    'win', 'loss',
    # projection recordkeeping
    'proj_wins', 'proj_losses',
    # teams-beat metrics
    'teams_beat_this_week', 'opponent_teams_beat_this_week',
    # projection deltas/spread
    'proj_score_error', 'abs_proj_score_error', 'above_proj_score', 'below_proj_score',
    'expected_spread', 'expected_odds', 'win_vs_spread', 'lose_vs_spread',
    'underdog_wins', 'favorite_losses',
    # playoffs/meta
    'is_playoffs', 'is_consolation',
    'gpa', 'grade', 'matchup_recap_title', 'matchup_recap_url', 'url', 'image_url',
    'division_id', 'week_start', 'week_end',
    'waiver_priority', 'has_draft_grade', 'faab_balance', 'number_of_moves', 'number_of_trades',
    'auction_budget_spent', 'auction_budget_total', 'win_probability', 'coverage_value', 'value',
    'felo_score', 'felo_tier'
]

KEEP.extend(sorted([c for c in df.columns if c.startswith("w_vs_") and not c.endswith("_sched")]))
KEEP.extend(sorted([c for c in df.columns if c.startswith("l_vs_") and not c.endswith("_sched")]))
KEEP.extend(sorted([c for c in df.columns if c.startswith("w_vs_") and c.endswith("_sched")]))
KEEP.extend(sorted([c for c in df.columns if c.startswith("l_vs_") and c.endswith("_sched")]))

for col in KEEP:
    if col not in df.columns:
        df[col] = np.nan
df = df[KEEP].copy()

# =============================================================================
# Save files
# =============================================================================
week_tag = "all" if week_input == "0" else str(int(week_input)).zfill(2)
csv_file  = os.path.join(OUT_DIR, f"matchup_data_week_{week_tag}_year_{year_input}.csv")
parq_file = os.path.join(OUT_DIR, f"matchup_data_week_{week_tag}_year_{year_input}.parquet")

df.to_csv(csv_file, index=False, encoding='utf-8-sig')
try:
    df.to_parquet(parq_file, engine='pyarrow', index=False)
except Exception:
    df.to_parquet(parq_file, engine='fastparquet', index=False)

print("Saved CSV:    ", csv_file)
print("Saved Parquet:", parq_file)
print("Rows:", len(df), "Columns:", len(df.columns))
