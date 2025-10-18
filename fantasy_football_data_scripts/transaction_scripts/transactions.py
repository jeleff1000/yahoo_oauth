#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import pandas as pd
import yahoo_fantasy_api as yfa
import nfl_data_py as nfl

# Add parent directory to path for oauth_utils import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from oauth_utils import ensure_oauth_path, create_oauth2
except Exception:
    # fallback if running from a different CWD
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from oauth_utils import ensure_oauth_path, create_oauth2

# =========================
# Name cleaning & mapping
# =========================
manual_mapping = {
    "hollywood brown": "Marquise Brown",
    "will fuller v": "Will Fuller",
    "jeff wilson": "Jeffery Wilson",
    "willie snead iv": "Willie Snead",
    "charles johnson": "Charles D Johnson",
    "kenneth barber": "Peyton Barber",
    "rodney smith": "Rod Smith",
    "bisi johnson": "Olabisi Johnson",
    "chris herndon": "Christopher Herndon",
    "scotty miller": "Scott Miller",
    "trenton richardson": "Trent Richardson",
}

def clean_name(name: str) -> str:
    name = re.sub(r'[èéêëÈÉÊË]', 'e', name or "")
    pattern = r"[.\-']|(\bjr\b\.?)|(\bsr\b\.?)|(\bII\b)|(\bIII\b)"
    cleaned = re.sub(pattern, '', name, flags=re.IGNORECASE).strip()
    return ' '.join(w.capitalize() for w in cleaned.split())

def apply_manual_mapping(name: str) -> str:
    return manual_mapping.get((name or "").lower(), name)

def norm_manager(nickname: str) -> str:
    if not nickname:
        return "N/A"
    return "Ilan" if nickname == "--hidden--" else nickname

def convert_timestamp(ts: str) -> str:
    # Yahoo timestamps are seconds since epoch
    dt = datetime.fromtimestamp(int(ts))
    return dt.strftime('%b %d %Y %I:%M:%S %p').upper()

# =========================
# Paths
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'fantasy_football_data', 'transaction_data'))
MATCHUP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'fantasy_football_data', 'matchup_data'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, 'transactions.csv')
PARQUET_PATH = os.path.join(OUTPUT_DIR, 'transactions.parquet')

# =========================
# OAuth - Supports both Format 1 and Format 2
# =========================
# Use centralized discovery that sets OAUTH_PATH and returns the resolved path
try:
    oauth_path = ensure_oauth_path()
    oauth = create_oauth2(oauth_path)
except SystemExit as e:
    raise SystemExit(str(e))
except Exception as e:
    raise SystemExit(f"OAuth initialization failed: {e}")

gm = yfa.Game(oauth, 'nfl')

# =========================
# Inputs
# =========================
parser = argparse.ArgumentParser(description="Fetch Yahoo Fantasy Football transaction data")
parser.add_argument('--year', type=int, default=None, help='Season year (0 for all years)')
parser.add_argument('--week', type=int, default=None, help='Week number (0 for all weeks)')
args = parser.parse_args()

# Use CLI args if provided, otherwise prompt
if args.year is not None:
    year_input = args.year
    if year_input == 0:
        most_recent_year = datetime.now().year
        years = list(range(most_recent_year, 2013, -1))
        week_input = 0
    else:
        years = [year_input]
        if args.week is not None:
            week_input = args.week
        else:
            week_input = int(input("Select the week to get data for (0 for all weeks): ").strip())
else:
    year_input = int(input("Select the year to get data for (0 for all years starting from most recent): ").strip())
    if year_input == 0:
        most_recent_year = datetime.now().year
        years = list(range(most_recent_year, 2013, -1))
        week_input = 0
    else:
        years = [year_input]
        week_input = int(input("Select the week to get data for (0 for all weeks): ").strip())

# ==========================================================
# Load matchup windows & build cross-season cumulative_week
# ==========================================================
# We read every matchup CSV in MATCHUP_DIR that your other script produced,
# and extract (year, week, week_start, week_end). Then build a unique,
# chronologically ordered list and assign cumulative_week starting at 15.

def _read_matchup_windows(matchup_dir: str) -> pd.DataFrame:
    files = [f for f in os.listdir(matchup_dir)
             if f.startswith("matchup_data_week_") and f.endswith(".csv")]
    frames: List[pd.DataFrame] = []
    usecols = ['week', 'year', 'week_start', 'week_end', 'cumulative_week']
    for f in files:
        try:
            df = pd.read_csv(os.path.join(matchup_dir, f), encoding='utf-8-sig')
        except Exception:
            continue
        cols = [c for c in usecols if c in df.columns]
        if not cols:
            continue
        frames.append(df[cols].copy())
    if not frames:
        return pd.DataFrame(columns=['year', 'week', 'week_start', 'week_end', 'cumulative_week'])

    raw = pd.concat(frames, ignore_index=True)
    # Keep one row per (year, week) with the most complete week_start/week_end if duplicates exist
    raw = (raw
           .sort_values(['year', 'week'])
           .drop_duplicates(subset=['year', 'week'], keep='first')
           .reset_index(drop=True))

    # Ensure types
    raw['year'] = pd.to_numeric(raw['year'], errors='coerce').astype('Int64')
    raw['week'] = pd.to_numeric(raw['week'], errors='coerce').astype('Int64')

    # Parse week_start/week_end if present
    for col in ('week_start', 'week_end'):
        if col in raw.columns:
            raw[col] = pd.to_datetime(raw[col], errors='coerce')

    # If week_start/week_end missing, try to infer from nfl schedule
    missing_mask = raw['week_start'].isna() | raw['week_end'].isna()
    if missing_mask.any():
        yrs = sorted(set(int(y) for y in raw.loc[missing_mask, 'year'].dropna().unique()))
        if yrs:
            sched = nfl.import_schedules(yrs)
            sched['gameday'] = pd.to_datetime(sched['gameday'], errors='coerce')
            wk_starts = (sched.groupby(['season', 'week'])['gameday']
                              .min().reset_index()
                              .rename(columns={'season': 'year',
                                               'gameday': 'week_start'}))
            wk_ends = (sched.groupby(['season', 'week'])['gameday']
                            .max().reset_index()
                            .rename(columns={'season': 'year',
                                             'gameday': 'week_end'}))
            wk = wk_starts.merge(wk_ends, on=['year', 'week'], how='outer')
            raw = raw.merge(wk, on=['year', 'week'], how='left', suffixes=('', '_sched'))
            # Prefer existing explicit values from matchup files, fall back to schedule
            for col in ('week_start', 'week_end'):
                raw[col] = raw[col].combine_first(raw[f'{col}_sched'])
                if f'{col}_sched' in raw.columns:
                    raw.drop(columns=[f'{col}_sched'], inplace=True)

    # Build canonical cross-season order & cumulative_week starting at 15
    canonical = raw[['year', 'week', 'week_start', 'week_end']].dropna(subset=['year', 'week'])
    canonical = canonical.sort_values(['year', 'week']).reset_index(drop=True)
    if len(canonical) > 0:
        canonical['cumulative_week'] = range(15, 15 + len(canonical))
    else:
        canonical['cumulative_week'] = pd.Series(dtype='Int64')

    return canonical

matchup_windows = _read_matchup_windows(MATCHUP_DIR)

# Helper: map timestamp -> (week, week_start, week_end) using matchup_windows of that year
def map_transaction_to_week(ts: str, year: int) -> Tuple[int, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    # If we have year windows from matchup data, use them; else fall back to nfl schedule
    rows = matchup_windows[matchup_windows['year'] == year].sort_values('week')
    t = datetime.fromtimestamp(int(ts))
    t_pd = pd.to_datetime(t)

    if len(rows) == 0:
        # Fallback to nfl schedule
        sched = nfl.import_schedules([year])
        sched['gameday'] = pd.to_datetime(sched['gameday'], errors='coerce')
        wk = (sched.groupby(['season', 'week'])['gameday']
                    .agg(week_start='min', week_end='max')
                    .reset_index()
                    .rename(columns={'season': 'year'}))
        rows = wk.sort_values('week')

    # If transaction is before first week_start -> week 1
    first_start = rows['week_start'].min()
    if pd.notna(first_start) and t_pd < first_start:
        w1 = int(rows['week'].min())
        r = rows[rows['week'] == w1].iloc[0]
        return w1, r.get('week_start', None), r.get('week_end', None)

    # If between week_start and week_end (inclusive), assign that week
    mask_between = (rows['week_start'].notna()) & (rows['week_end'].notna()) & \
                   (t_pd >= rows['week_start']) & (t_pd <= rows['week_end'])
    if mask_between.any():
        r = rows[mask_between].iloc[0]
        return int(r['week']), r.get('week_start', None), r.get('week_end', None)

    # Otherwise, if after last week_end -> final week of season
    last_row = rows.sort_values('week').iloc[-1]
    return int(last_row['week']), last_row.get('week_start', None), last_row.get('week_end', None)

# Quick accessor for cumulative_week mapping
cum_map: Dict[Tuple[int, int], int] = {
    (int(r.year), int(r.week)): int(r.cumulative_week)
    for r in matchup_windows.itertuples(index=False)
}

# =========================
# Collect transactions
# =========================
all_transactions: List[dict] = []

for year in years:
    league_ids = gm.league_ids(year=year)
    if not league_ids:
        continue
    yearid = league_ids[-1]
    league = gm.to_league(yearid)

    # Team mapping (team key -> nickname/manager)
    teams_url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{yearid}/teams"
    resp = oauth.session.get(teams_url)
    xmlstring = re.sub(r' xmlns="[^"]+"', '', resp.text, count=1)
    _ = ET.fromstring(xmlstring)  # parse check
    mgr_df = pd.read_xml(xmlstring, xpath=".//manager", parser="etree")
    mgr_df["team"] = f"{yearid}.t." + mgr_df["manager_id"].astype(str)
    mgr_df = mgr_df[["team", "nickname"]]
    teamid_to_manager = {row["team"]: norm_manager(row["nickname"]) for _, row in mgr_df.iterrows()}

    # Transactions
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{yearid}/transactions"
    print("Making API Call:", url)
    r = oauth.session.get(url)
    xmlstring = re.sub(r' xmlns="[^"]+"', '', r.text, count=1)
    root = ET.fromstring(xmlstring)

    for tr in root.findall(".//transaction"):
        tx_type = (tr.findtext("type") or "").strip()
        status = (tr.findtext("status") or "").strip()
        ts = tr.findtext("timestamp") or "0"

        # Resolve week via matchup windows (uses your exported week_start/week_end)
        week_val, week_start, week_end = map_transaction_to_week(ts, year)

        # Team roles (mapped to manager names)
        def map_team(tag: str) -> str:
            node = tr.find(tag)
            if node is None or node.text is None:
                return "Unknown"
            return teamid_to_manager.get(node.text.strip(), "Unknown")

        team_keys = {
            "destination_team_key": map_team("destination_team_key"),
            "source_team_key": map_team("source_team_key"),
            "waiver_team_key": map_team("waiver_team_key"),
            "trader_team_key": map_team("trader_team_key"),
            "tradee_team_key": map_team("tradee_team_key"),
        }

        faab_bid = (tr.findtext("faab_bid") or "0").strip()

        pickup = {
            "transaction_key": (tr.findtext("transaction_key") or "").strip(),
            "status": status,
            "timestamp": ts,
            "faab_bid": faab_bid,
            "week": int(week_val),
            "year": int(year),
            "week_start": week_start if pd.isna(week_start) is False else pd.NaT,
            "week_end": week_end if pd.isna(week_end) is False else pd.NaT,
            "players": [],
        }
        pickup.update(team_keys)

        for player in tr.findall("players/player"):
            name = player.findtext("name/full") or ""
            name = apply_manual_mapping(clean_name(name))
            ptype = (player.findtext("transaction_data/type") or "").strip()
            source_type = (player.findtext("transaction_data/source_type") or "").strip()
            dest_el = player.find("transaction_data/destination_type")
            destination = (dest_el.text if dest_el is not None else "Unknown") or "Unknown"

            if destination == "team":
                tkey = player.findtext("transaction_data/destination_team_key") or ""
            elif destination == "waivers":
                tkey = player.findtext("transaction_data/source_team_key") or ""
            else:
                tkey = ""

            nickname = teamid_to_manager.get(tkey, "Unknown") if tkey else "Unknown"

            pickup["players"].append({
                "player_key": (player.findtext("player_key") or "").strip(),
                "name": name,
                "transaction_type": ptype,
                "source_type": source_type,
                "destination": destination,
                "team": tkey,
                "nickname": nickname,
            })

        all_transactions.append(pickup)

# =========================
# Build DataFrame
# =========================
def create_dataframe(pickups: List[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for p in pickups:
        year = int(p["year"])
        week = int(p["week"])
        for pl in p["players"]:
            nickname = pl["nickname"]
            player_name = pl["name"]
            human_ts = convert_timestamp(p["timestamp"])
            rows.append({
                "transaction_id": p["transaction_key"],
                "manager": nickname,
                "player_name": player_name,
                "faab_bid": int(p["faab_bid"]),
                "week": week,
                "year": year,
                "trader_team_key": p["trader_team_key"],
                "tradee_team_key": p["tradee_team_key"],
                "transaction_type": pl["transaction_type"],
                "source_type": pl["source_type"],
                "destination": pl["destination"],
                "status": p["status"],
                "human_readable_timestamp": human_ts,
                "week_start": p["week_start"],
                "week_end": p["week_end"],
            })
    df = pd.DataFrame(rows)
    # Filter to requested week if needed
    return df if week_input == 0 else df[df['week'] == week_input].copy()

tx = create_dataframe(all_transactions)

# Attach cumulative_week based on matchup_windows mapping
if not matchup_windows.empty:
    tx = tx.merge(
        matchup_windows[['year', 'week', 'cumulative_week', 'week_start', 'week_end']],
        on=['year', 'week'],
        how='left',
        suffixes=('', '_mw')
    )
    # Prefer explicitly mapped week_start/week_end from earlier logic if present; otherwise take from merge
    for col in ('week_start', 'week_end'):
        if f"{col}_mw" in tx.columns:
            tx[col] = tx[col].combine_first(tx[f"{col}_mw"])
            tx.drop(columns=[f"{col}_mw"], inplace=True)
else:
    # If no matchup windows available at all, build cumulative_week per season as 1..N but still start earliest season at 15
    if not tx.empty:
        base = (tx[['year', 'week']]
                .drop_duplicates()
                .sort_values(['year', 'week'])
                .reset_index(drop=True))
        base['cumulative_week'] = range(15, 15 + len(base))
        tx = tx.merge(base, on=['year', 'week'], how='left')

# Safety: ensure cumulative_week exists
if 'cumulative_week' not in tx.columns:
    tx['cumulative_week'] = pd.NA

# =========================
# Manager/Player IDs using cumulative_week & year
# =========================
def _mk_week_key(name: str, cw) -> str:
    if pd.isna(cw) or name is None or str(name).strip() == "":
        return ""
    return f"{str(name).replace(' ', '')}{int(cw)}"

def _mk_year_key(name: str, yr) -> str:
    if pd.isna(yr) or name is None or str(name).strip() == "":
        return ""
    return f"{str(name).replace(' ', '')}{int(yr)}"

tx['manager_week'] = [ _mk_week_key(m, cw) for m, cw in zip(tx['manager'], tx['cumulative_week']) ]
tx['manager_year'] = [ _mk_year_key(m, y)   for m, y  in zip(tx['manager'], tx['year']) ]
tx['player_week']  = [ _mk_week_key(p, cw) for p, cw in zip(tx['player_name'], tx['cumulative_week']) ]
tx['player_year']  = [ _mk_year_key(p, y)   for p, y  in zip(tx['player_name'], tx['year']) ]

# =========================
# Final column order
# =========================
column_order = [
    "transaction_id", "manager", "player_name", "faab_bid",
    "week", "year", "cumulative_week",
    "week_start", "week_end",
    "trader_team_key", "tradee_team_key",
    "transaction_type", "source_type", "destination",
    "status", "human_readable_timestamp",
    "manager_week", "manager_year", "player_week", "player_year",
]
# Ensure all exist
for c in column_order:
    if c not in tx.columns:
        tx[c] = pd.NA
tx = tx[column_order].copy()

# =========================
# Save
# =========================
tx.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
tx.to_parquet(PARQUET_PATH, index=False)
print(f"Saved: {CSV_PATH}")
print(f"Saved: {PARQUET_PATH}")
