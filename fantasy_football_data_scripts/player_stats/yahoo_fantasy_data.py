#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import time
import re
import json
import xml.etree.ElementTree as ET
import threading
import random
import sys
from time import monotonic

import pandas as pd
import numpy as np
import requests

# Add parent directory to path for oauth_utils import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oauth_utils import create_oauth2

# Your utils module provides yfa (yahoo_fantasy_api), clean_name, etc.
from imports_and_utils import yfa, clean_name  # type: ignore

# Scoring rules helpers
from pull_scoring_rules import parse_scoring_rules  # NOTE: fetch_league_settings removed

MAX_TEAM_WORKERS: int = 3  # maximum concurrent team roster requests per week
REQUEST_DELAY: float = 0.2  # seconds to sleep before each HTTP request
PAUSE_SECONDS_ON_TIMEOUT: int = 300  # 5 minutes

# -----------------------------------------------------------------------------
# Global rate limiter for HTTP requests (token bucket)
# -----------------------------------------------------------------------------
_REQUESTS_PER_SEC: float = 4.0
_TOKENS: float = _REQUESTS_PER_SEC
_LAST: float = monotonic()
_LOCK: threading.Lock = threading.Lock()


def _rl_acquire() -> None:
    """Acquire one request token, sleeping if necessary to replenish tokens."""
    global _TOKENS, _LAST
    with _LOCK:
        now = monotonic()
        _TOKENS = min(_REQUESTS_PER_SEC, _TOKENS + (now - _LAST) * _REQUESTS_PER_SEC)
        _LAST = now
        if _TOKENS < 1.0:
            need = (1.0 - _TOKENS) / _REQUESTS_PER_SEC
            time.sleep(need)
            _TOKENS = 0.0
        else:
            _TOKENS -= 1.0


def _sleep_with_jitter(base: float) -> None:
    """Sleep for base seconds plus up to 30% random jitter (mitigates thundering herd)."""
    if base > 0:
        time.sleep(base + random.uniform(0, base * 0.3))


# ----------------------------
# Paths (relative; no hardcoding)
# ----------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # ...\fantasy_football_data_downloads
OAUTH_PATH = REPO_ROOT / "oauth" / "Oauth.json"
OUTPUT_DIR = REPO_ROOT / "fantasy_football_data" / "player_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_DIR = OUTPUT_DIR / "yahoo_league_settings"
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Regex helpers
# ----------------------------
LEAGUE_KEY_RE = re.compile(r"^\d+\.l\.\d+$")
XML_NS_RE = re.compile(r' xmlns="[^"]+"')


# ----------------------------
# Custom timeout-ish errors
# ----------------------------
class APITimeoutError(Exception):
    """Signal a recoverable API timeout/denial so caller can resume or save partials."""
    pass


class RecoverableAPIError(APITimeoutError):
    """Alias for transient API failures (rate-limit, 403/429, 'Request denied', etc.)."""
    pass


# Last-name normalization: lowercase, strip hyphens/apostrophes/periods, drop suffixes
_PUNCT_LAST_RE = re.compile(r"[.\-'\u2019]")  # includes smart apostrophe
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "ivs"}


def normalize_last_name_from_full(full_name: str) -> str:
    if not isinstance(full_name, str):
        return ""
    s = full_name.strip()
    if not s:
        return ""
    s = s.lower()
    s = _PUNCT_LAST_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    tokens = s.split(" ")
    while tokens and tokens[-1] in _SUFFIXES:
        tokens.pop()
    if not tokens:
        return ""
    return tokens[-1]


# ----------------------------
# League-key acquisition (NO PROMPTS)
# ----------------------------
def get_league_key(oauth: OAuth2, year: int, league_key_arg: Optional[str]) -> Optional[str]:
    """
    No interactive prompts. Use CLI arg if valid; else discovery with retries.
    On transient failures (e.g., 'Request denied', 403/429), raise RecoverableAPIError
    so the caller treats it like a timeout.
    """
    if league_key_arg:
        lk = league_key_arg.strip()
        if LEAGUE_KEY_RE.match(lk):
            return lk
        print(f"[league_key] Ignoring invalid format (####.l.#####): {lk}")
        return None

    last_exc: Optional[Exception] = None
    for attempt in range(6):
        try:
            gm = yfa.Game(oauth, "nfl")
            keys = gm.league_ids(year=year)
            if keys:
                return keys[-1]
            raise RecoverableAPIError(f"No league ids returned for {year}")
        except Exception as e:
            last_exc = e
            msg = str(e)
            transient = (
                "Request denied" in msg
                or "Forbidden" in msg
                or "Forbidden access" in msg
                or "Too Many Requests" in msg
            )
            if attempt < 5 and transient:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise RecoverableAPIError(f"league_ids discovery failed for {year}: {e}") from e

    raise RecoverableAPIError(f"league_ids discovery exhausted retries for {year}: {last_exc}")


# ----------------------------
# HTTP/XML helpers
# ----------------------------
def fetch_url(url: str, oauth: OAuth2, max_retries: int = 6, backoff: float = 0.5) -> ET.Element:
    """
    GET Yahoo XML with retries; strip default namespace; return parsed root.
    Raises APITimeoutError/RecoverableAPIError on transient failures so the caller can resume.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            _rl_acquire()
            _sleep_with_jitter(REQUEST_DELAY)
            r = oauth.session.get(url, timeout=30)

            try:
                r.raise_for_status()
            except requests.HTTPError as he:
                code = getattr(he.response, "status_code", None)
                if code in (429, 403, 502, 503, 504):
                    if attempt == max_retries - 1:
                        raise RecoverableAPIError(f"HTTP {code} on {url}") from he
                    time.sleep(backoff * (2 ** attempt))
                    continue
                raise

            text = (r.text or "").strip()
            if not text:
                if attempt == max_retries - 1:
                    raise RecoverableAPIError(f"Empty XML response from {url}")
                time.sleep(backoff * (2 ** attempt))
                continue
            if "Request denied" in text:
                if attempt == max_retries - 1:
                    raise RecoverableAPIError(f"Request denied at {url}")
                time.sleep(backoff * (2 ** attempt))
                continue

            xmlstring = XML_NS_RE.sub("", text, count=1)
            return ET.fromstring(xmlstring)

        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise APITimeoutError(f"Timeout fetching {url}") from e
            time.sleep(backoff * (2 ** attempt))
        except (requests.RequestException, ET.ParseError, ValueError) as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise RecoverableAPIError(f"Transient error fetching {url}: {e}") from e
            time.sleep(backoff * (2 ** attempt))

    if isinstance(last_exc, requests.exceptions.Timeout):
        raise APITimeoutError(f"Timeout fetching {url}") from last_exc
    raise RecoverableAPIError(f"Unknown fetch_url failure for {url}: {last_exc}")


def fetch_team_week_xml(league_key: str, week: int, team_id: int, oauth: OAuth2) -> Optional[ET.Element]:
    """
    One team's roster+stats for a given week (XML root) or ``None`` if unavailable.
    """
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/team/{league_key}.t.{team_id}/roster;week={week}/players/stats"
    root = fetch_url(url, oauth)
    if root.find("team/roster/players/player") is not None:
        return root
    return None


# ----------------------------
# Week-state helpers (prefer week_end date; fallback to points>0)
# ----------------------------
def week_is_completed(oauth: OAuth2, league_key: str, week: int) -> bool:
    """
    A week is considered complete if:
      1) scoreboard.week_end (YYYY-MM-DD) exists and is <= today (UTC), OR
      2) any team_points/total > 0 (fallback).
    """
    try:
        root = fetch_url(
            f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/scoreboard;week={week}",
            oauth
        )
    except (APITimeoutError, RecoverableAPIError):
        raise
    except Exception:
        return False

    wk_end_node = root.find("league/scoreboard/week_end")
    if wk_end_node is not None and (wk_end_node.text or "").strip():
        try:
            wk_end = datetime.strptime(wk_end_node.text.strip(), "%Y-%m-%d").date()
            today = datetime.now(timezone.utc).date()
            if wk_end <= today:
                return True
        except Exception:
            pass

    for m in root.findall("league/scoreboard/matchups/matchup"):
        for t in m.findall("teams/team"):
            pts = t.find("team_points/total")
            try:
                if float((pts.text if pts is not None else "0") or "0") > 0:
                    return True
            except ValueError:
                pass
    return False


def latest_completed_week_by_week_end(oauth: OAuth2, league_key: str, max_possible: int = 18) -> int:
    last = 0
    for wk in range(1, max_possible + 1):
        try:
            if week_is_completed(oauth, league_key, wk):
                last = wk
            else:
                break
        except (APITimeoutError, RecoverableAPIError):
            raise
        except Exception:
            break
    return last


# ----------------------------
# Scoring + Roster extraction & save
# ----------------------------
def get_dst_scoring_from_rules(rules: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Derive DST-related scoring from the already-parsed full rules.
    No network calls here.
    """
    name_map = {
        "Sack": "Sack",
        "Int": "Interception",
        "Interception": "Interception",
        "Fum Rec": "Fumble Recovery",
        "Fumble Recovery": "Fumble Recovery",
        "TD": "Touchdown",
        "Touchdown": "Touchdown",
        "Safe": "Safety",
        "Safety": "Safety",
        "Ret TD": "Kickoff and Punt Return Touchdowns",
        "Kickoff and Punt Return Touchdowns": "Kickoff and Punt Return Touchdowns",
        "Pts Allow 0": "PA_0",
        "Pts Allow 1-6": "PA_1_6",
        "Pts Allow 7-13": "PA_7_13",
        "Pts Allow 14-20": "PA_14_20",
        "Pts Allow 21-27": "PA_21_27",
        "Pts Allow 28-34": "PA_28_34",
        "Pts Allow 35+": "PA_35_plus",
        "Yds Allow Neg": "Yds Allow Neg",
        "Yds Allow 0-99": "Yds Allow 0-99",
        "Yds Allow 100-199": "Yds Allow 100-199",
        "Yds Allow 200-299": "Yds Allow 200-299",
        "Yds Allow 300-399": "Yds Allow 300-399",
        "Yds Allow 400-499": "Yds Allow 400-499",
        "Yds Allow 500+": "Yds Allow 500+",
    }
    scoring = {v: 0.0 for v in name_map.values()}
    for rule in rules:
        key = name_map.get(str(rule.get("name", "")).strip())
        if key:
            try:
                scoring[key] = float(rule.get("points", 0.0))
            except Exception:
                pass
    return scoring


def save_dst_settings_json_from_rules(
    year: int,
    league_key: str,
    rules: List[Dict[str, Any]],
    start_week: int,
    end_week: int,
) -> Path:
    scoring = get_dst_scoring_from_rules(rules)
    payload = {
        "year": int(year),
        "league_key": league_key,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "start_week": int(start_week),
        "end_week": int(end_week),
        "dst_scoring": scoring,
    }
    safe_lk = league_key.replace(".", "_")
    out_path = SETTINGS_DIR / f"yahoo_dst_scoring_{year}_{safe_lk}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[settings] Saved DST settings: {out_path}")
    return out_path


def save_full_scoring_json(year: int, league_key: str, rules: List[Dict[str, Any]]) -> Path:
    """Persist the full parsed scoring rules for later use (e.g., in merge script)."""
    safe_lk = league_key.replace(".", "_")
    out_path = SETTINGS_DIR / f"yahoo_full_scoring_{year}_{safe_lk}.json"
    payload = {
        "year": int(year),
        "league_key": league_key,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "full_scoring": rules,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[settings] Saved FULL scoring rules: {out_path}")
    return out_path


# ---------- NEW: roster positions parser + saver ----------
def parse_roster_positions(settings_root: ET.Element) -> List[Dict[str, Any]]:
    """
    Parse Yahoo roster positions and slot counts from league/settings.
    Returns a list of {position, count, position_type?, is_flex?}.
    """
    out: List[Dict[str, Any]] = []
    for rp in settings_root.findall("league/settings/roster_positions/roster_position"):
        pos = (rp.findtext("position") or "").strip()
        cnt_txt = (rp.findtext("count") or "0").strip()
        ptype = (rp.findtext("position_type") or "").strip()
        is_flex_txt = (rp.findtext("is_flex") or "").strip()

        try:
            cnt = int(cnt_txt)
        except Exception:
            cnt = 0

        is_flex = False
        if is_flex_txt != "":
            # some leagues include 0/1; others omit the node entirely
            try:
                is_flex = bool(int(is_flex_txt))
            except Exception:
                is_flex = is_flex_txt.lower() in {"true", "yes"}

        if pos:
            out.append(
                {
                    "position": pos,               # e.g., QB, RB, WR, TE, W/R/T, K, DEF, BN, IR, etc.
                    "count": cnt,                  # slot count
                    "position_type": ptype or None,
                    "is_flex": is_flex,
                }
            )
    return out


def save_roster_json(year: int, league_key: str, roster: List[Dict[str, Any]]) -> Path:
    """
    Save roster positions/slot counts to a JSON file for use elsewhere in your pipeline.
    """
    safe_lk = league_key.replace(".", "_")
    out_path = SETTINGS_DIR / f"yahoo_roster_{year}_{safe_lk}.json"
    payload = {
        "year": int(year),
        "league_key": league_key,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "roster_positions": roster,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[settings] Saved roster positions: {out_path}")
    return out_path
# ---------------------------------------------------------


# ----------------------------
# NEW: Lineup position and optimal player logic
# ----------------------------
def assign_lineup_positions_and_optimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates three columns:
    1. lineup_position: Rank each position by points (WR1, WR2, RB1, etc.)
    2. optimal_player: Binary flag if player should have started (0 or 1)
    3. optimal_lineup_position: What slot they'd be in optimal lineup (QB, WR1, FLEX, BN1, etc.)
    """
    df = df.copy()

    def process_team_week(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()

        # Step 1: Assign lineup_position (simple ranking by position+points)
        g['lineup_position'] = ''
        for pos in g['yahoo_position'].dropna().unique():
            pos_mask = g['yahoo_position'] == pos
            sorted_indices = g.loc[pos_mask].sort_values('points', ascending=False).index
            for rank, idx in enumerate(sorted_indices, start=1):
                g.at[idx, 'lineup_position'] = f"{pos}{rank}"

        # Step 2: Determine optimal starters from actual week lineup slots (unchanged)
        fantasy_pos = g['fantasy_position'].fillna('').astype(str).str.upper()

        starters_only = g[~fantasy_pos.isin(['BN', 'BENCH']) & ~fantasy_pos.str.startswith('IR')]

        slot_counts: Dict[str, int] = {}
        flex_slots = 0

        for fp in starters_only['fantasy_position']:
            fp_clean = str(fp).strip().upper()
            if fp_clean in ['W/R/T', 'FLEX']:
                flex_slots += 1
            elif fp_clean and fp_clean not in ['BN', 'BENCH']:
                slot_counts[fp_clean] = slot_counts.get(fp_clean, 0) + 1

        # Initialize columns
        g['optimal_player'] = 0
        g['optimal_lineup_position'] = ''
        available_players = g.copy()
        selected_indices: List[int] = []

        # Fill dedicated position slots first (QB, RB, WR, TE, K, DEF)
        position_counters: Dict[str, int] = {}
        for pos, count in slot_counts.items():
            pos_players = available_players[
                (available_players['yahoo_position'] == pos) &
                (~available_players.index.isin(selected_indices))
            ].sort_values('points', ascending=False)

            for idx in pos_players.head(count).index:
                selected_indices.append(idx)
                g.at[idx, 'optimal_player'] = 1
                position_counters[pos] = position_counters.get(pos, 0) + 1
                g.at[idx, 'optimal_lineup_position'] = f"{pos}{position_counters[pos]}"

        # Fill FLEX slots with best remaining RB/WR/TE
        flex_counter = 0
        if flex_slots > 0:
            flex_eligible = available_players[
                (available_players['yahoo_position'].isin(['RB', 'WR', 'TE'])) &
                (~available_players.index.isin(selected_indices))
            ].sort_values('points', ascending=False)

            for idx in flex_eligible.head(flex_slots).index:
                selected_indices.append(idx)
                g.at[idx, 'optimal_player'] = 1
                flex_counter += 1
                g.at[idx, 'optimal_lineup_position'] = f"W/R/T{flex_counter}"

        # Assign bench/IR positions to non-starters
        bench_counter = 0
        ir_counter = 0
        for idx in g.index:
            if idx not in selected_indices:
                actual_fp = str(g.at[idx, 'fantasy_position']).upper().strip()
                if actual_fp.startswith('IR'):
                    ir_counter += 1
                    g.at[idx, 'optimal_lineup_position'] = f"IR{ir_counter}"
                else:
                    bench_counter += 1
                    g.at[idx, 'optimal_lineup_position'] = f"BN{bench_counter}"

        return g

    # Process each manager/week/year group
    result_parts: List[pd.DataFrame] = []
    for _, group in df.groupby(['manager', 'week', 'year'], dropna=False, sort=False):
        result_parts.append(process_team_week(group))

    return pd.concat(result_parts, ignore_index=True)


# ----------------------------
# Core
# ----------------------------
def yahoo_fantasy_data(year: int, week: int, league_key: Optional[str] = None) -> pd.DataFrame:
    if not OAUTH_PATH.exists():
        raise FileNotFoundError(f"OAuth file not found at: {OAUTH_PATH}")

    oauth = OAuth2(None, None, from_file=str(OAUTH_PATH))
    if not oauth.token_is_valid():
        oauth.refresh_access_token()

    # --- league key (treat discovery denials as timeout-ish)
    try:
        league_key = get_league_key(oauth, year, league_key)
    except RecoverableAPIError as e:
        raise APITimeoutError(str(e)) from e

    if not league_key:
        raise APITimeoutError(f"No league_key found for {year}")

    # --- league settings (single parse -> stat_map, week bounds, save DST + FULL + ROSTER)
    try:
        if not oauth.token_is_valid():
            print(f"[yahoo_fantasy_data] Refreshing token for {year}")
            oauth.refresh_access_token()

        settings_root = fetch_url(
            f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/settings",
            oauth
        )

        # Parse ONCE
        rules: List[Dict[str, Any]] = parse_scoring_rules(settings_root)

        # Build stat map from settings_root
        stat_map: Dict[str, str] = {
            s.find("stat_id").text: s.find("display_name").text
            for s in settings_root.findall("league/settings/stat_categories/stats/stat")
            if s.find("stat_id") is not None and s.find("display_name") is not None
        }

        # League bounds
        def _ival(path: str, default: int) -> int:
            node = settings_root.find(path)
            try:
                return int(node.text) if (node is not None and node.text) else default
            except Exception:
                return default

        start_week_cfg = _ival("league/start_week", 1)
        end_week_cfg = _ival("league/end_week", 18)
        if start_week_cfg < 1:
            start_week_cfg = 1
        if end_week_cfg < start_week_cfg:
            end_week_cfg = start_week_cfg

        # Save DST + FULL rules
        save_dst_settings_json_from_rules(year, league_key, rules, start_week_cfg, end_week_cfg)
        save_full_scoring_json(year, league_key, rules)

        # ---------- NEW: Parse and save roster positions ----------
        roster_positions = parse_roster_positions(settings_root)
        save_roster_json(year, league_key, roster_positions)
        # ----------------------------------------------------------

    except (APITimeoutError, RecoverableAPIError):
        raise
    except Exception as e:
        print(f"[yahoo_fantasy_data] Failed to get/parse league settings for {year}: {e}")
        return pd.DataFrame()

    current_year = datetime.now().year
    if week == 0:
        if year == current_year:
            latest = latest_completed_week_by_week_end(oauth, league_key, max_possible=end_week_cfg)
            if latest == 0:
                print("[yahoo_fantasy_data] No completed weeks yet.")
                return pd.DataFrame()
            weeks = range(max(1, start_week_cfg), latest + 1)
        else:
            weeks = range(max(1, start_week_cfg), end_week_cfg + 1)
    else:
        if year == current_year:
            latest = latest_completed_week_by_week_end(oauth, league_key, max_possible=end_week_cfg)
            if week > latest:
                print(
                    f"[yahoo_fantasy_data] Future/unplayed week requested ({week} > latest {latest}). Returning empty.")
                return pd.DataFrame()
        if not (start_week_cfg <= week <= end_week_cfg):
            print(
                f"[yahoo_fantasy_data] Week {week} out of league bounds [{start_week_cfg}-{end_week_cfg}]. Returning empty.")
            return pd.DataFrame()
        weeks = [week]

    # team count (default 10 if not present)
    team_count = 10
    try:
        tnode = settings_root.find("league/num_teams")
        if tnode is not None and tnode.text:
            team_count = int(tnode.text)
    except Exception:
        pass

    def opponent_map_for_week(wk: int) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        sb = fetch_url(
            f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/scoreboard;week={wk}",
            oauth
        )
        for m in sb.findall("league/scoreboard/matchups/matchup"):
            t1 = m.find("teams/team[1]/managers/manager/nickname")
            t2 = m.find("teams/team[2]/managers/manager/nickname")
            if t1 is not None and t2 is not None:
                n1 = (t1.text or "").strip()
                n2 = (t2.text or "").strip()
                if n1 and n2:
                    mapping[n1] = n2
                    mapping[n2] = n1
        return mapping

    rows: List[Dict[str, Any]] = []

    for wk in weeks:
        try:
            opp_map = opponent_map_for_week(wk)

            with ThreadPoolExecutor(max_workers=min(team_count, MAX_TEAM_WORKERS)) as ex:
                futures = [ex.submit(fetch_team_week_xml, league_key, wk, tid, oauth) for tid in
                           range(1, team_count + 1)]
                for fut in futures:
                    try:
                        root = fut.result()
                    except (APITimeoutError, RecoverableAPIError):
                        raise
                    except requests.HTTPError as he:
                        if he.response is not None and he.response.status_code == 400:
                            raise
                        raise

                    if not root:
                        continue

                    nick_node = root.find("team/managers/manager/nickname")
                    manager = (nick_node.text or "").strip() if nick_node is not None else ""
                    opponent = opp_map.get(manager, "")

                    for pl in root.findall("team/roster/players/player"):
                        def _txt(node, default=""):
                            return node.text if node is not None else default

                        name = clean_name(_txt(pl.find("name/full"), ""))
                        player_last = normalize_last_name_from_full(name)

                        primary_pos = _txt(pl.find("primary_position"))
                        position = _txt(pl.find("selected_position/position"))
                        nfl_team = _txt(pl.find("editorial_team_abbr"))
                        player_id = _txt(pl.find("player_id"))
                        bye = _txt(pl.find("bye_weeks/week"), None)

                        pts_node = pl.find("player_points/total")
                        try:
                            points = round(float(_txt(pts_node, "0") or "0"), 2)
                        except ValueError:
                            points = 0.0

                        stat_vals: Dict[str, Any] = {}
                        for s in pl.findall("player_stats/stats/stat"):
                            sid = _txt(s.find("stat_id"))
                            if sid in stat_map:
                                stat_vals[stat_map[sid]] = _txt(s.find("value"), "0")

                        row = {
                            "year": int(year),
                            "week": int(wk),
                            "manager": manager,
                            "opponent": opponent,
                            "player": name,
                            "player_last_name": player_last,
                            "yahoo_player_id": player_id,
                            "nfl_team": nfl_team,
                            "yahoo_position": primary_pos,
                            "fantasy_position": position,
                            "bye": bye,
                            "points": points,
                        }
                        row.update(stat_vals)
                        rows.append(row)

        except (APITimeoutError, RecoverableAPIError):
            raise
        except requests.HTTPError as he:
            if he.response is not None and he.response.status_code == 400:
                print(f"[yahoo_fantasy_data] Week {wk} not available for league (HTTP 400). Skipping week.")
                continue
            raise

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # normalize dtypes
    for col in ("year", "week"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # tidy DEF names
    def _fix_def(mask_team: str, label: str) -> None:
        mask = (df["yahoo_position"] == "DEF") & (df["nfl_team"] == mask_team)
        df.loc[mask, ["player", "player_last_name"]] = [label, normalize_last_name_from_full(label)]
        df.loc[df["manager"] == "--hidden--", "manager"] = "Ilan"
    for abbr, name_ in [("LAC", "Chargers"), ("LAR", "Rams"), ("NYJ", "Jets"), ("NYG", "Giants")]:
        _fix_def(abbr, name_)

    # avoid 'targets' collision if present
    if "targets" in df.columns:
        df.rename(columns={"targets": "targetz"}, inplace=True)

    # --- column normalization: lowercase + underscores ---
    df.columns = [re.sub(r"\s+", "_", c).lower() for c in df.columns]

    # --- requested Yahoo-side defensive renames ---
    df.rename(
        columns={
            "sack": "def_sack",
            "td": "def_td",
            "safe": "def_safeties",
            "tfl": "def_tackles_for_loss",
        },
        inplace=True,
    )

    # Add 'started' column - 1 if player was in starting lineup, 0 if on bench/IR/etc.
    def is_started(fantasy_position):
        if pd.isna(fantasy_position):
            return 0
        pos_str = str(fantasy_position).strip().upper()
        if not pos_str or pos_str in ['', 'BN', 'BENCH', 'IR', 'IR1', 'IR2', 'IR3', 'NAN', 'NONE']:
            return 0
        if pos_str.startswith('IR'):
            return 0
        return 1

    df['started'] = df['fantasy_position'].apply(is_started)

    # Apply lineup position and optimal player logic
    df = assign_lineup_positions_and_optimal(df)

    # Add team_1, team_2, matchup_name columns
    def _safe_str(s: Any) -> str:
        return (str(s).strip()) if pd.notna(s) else ""

    def _team_pair(m: Any, o: Any) -> Tuple[str, str]:
        m_s, o_s = _safe_str(m), _safe_str(o)
        if not m_s or not o_s:
            return "", ""
        if m_s.lower() <= o_s.lower():
            return m_s.title(), o_s.title()
        return o_s.title(), m_s.title()

    t1_t2 = df.apply(lambda r: _team_pair(r.get("manager"), r.get("opponent")), axis=1, result_type="expand")
    t1_t2.columns = ["team_1", "team_2"]
    df = pd.concat([df, t1_t2], axis=1)
    df["matchup_name"] = df.apply(
        lambda r: f"{r['team_1']} vs {r['team_2']}" if r.get("team_1") and r.get("team_2") else "",
        axis=1,
    )

    # ------------------------------------------------------------------
    # FILL MISSING "points" USING THE FULL LEAGUE RULES
    # ------------------------------------------------------------------
    full_scoring_rules = rules

    def _normalize(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower())

    # Build flexible column alias map
    existing_cols = {c: c for c in df.columns}

    def _alias(*cands: str) -> Optional[str]:
        for c in cands:
            key = _normalize(c)
            if key in existing_cols:
                return existing_cols[key]
            for col in df.columns:
                if _normalize(col) == key:
                    return col
        return None

    # Common metric aliases
    name_to_col = {
        # Offense
        "Pass Yds": _alias("pass_yds", "passing_yards", "pass_yards"),
        "Pass TD": _alias("pass_td", "passing_tds", "pass_touchdowns"),
        "Int": _alias("int", "interceptions", "passing_interceptions"),
        "Rush Yds": _alias("rush_yds", "rushing_yards"),
        "Rush TD": _alias("rush_td", "rushing_tds"),
        "Rec": _alias("rec", "receptions"),
        "Rec Yds": _alias("rec_yds", "receiving_yards"),
        "Rec TD": _alias("rec_td", "receiving_tds"),
        "Ret TD": _alias("ret_td", "return_tds", "kick_punt_return_tds"),
        "2-PT": _alias("2-pt", "two_pt", "two_point"),
        "Fum Lost": _alias("fum_lost", "fumbles_lost"),
        "Fum Ret TD": _alias("fum_ret_td", "fumble_return_td"),
        "PAT Made": _alias("pat_made", "xp_made", "xpm"),
        "FG Yds": _alias("fg_yds", "field_goal_yards"),
        "XPR": _alias("xpr", "extra_point_return"),

        # Defense / ST
        "Sack": _alias("def_sack", "sacks_def", "sacks"),
        "Interception": _alias("def_interceptions", "interceptions_def", "int_def"),
        "Fumble Recovery": _alias("fum_rec", "fumble_recovery_def"),
        "Touchdown": _alias("def_td", "def_tds", "defensive_td"),
        "Safety": _alias("def_safeties", "safeties"),
        "Kickoff and Punt Return Touchdowns": _alias("special_teams_tds", "st_tds", "ret_td"),

        "4 Dwn Stops": _alias("4_dwn_stops", "fourth_down_stops"),
        "TFL": _alias("def_tackles_for_loss", "tfl"),
        "3 and Outs": _alias("3_and_outs", "three_and_outs"),

        "Pts Allow 0": _alias("pts_allow_0"),
        "Pts Allow 1-6": _alias("pts_allow_1_6"),
        "Pts Allow 7-13": _alias("pts_allow_7_13"),
        "Pts Allow 14-20": _alias("pts_allow_14_20"),
        "Pts Allow 21-27": _alias("pts_allow_21_27"),
        "Pts Allow 28-34": _alias("pts_allow_28_34"),
        "Pts Allow 35+": _alias("pts_allow_35_plus"),

        "Yds Allow Neg": _alias("yds_allow_neg"),
        "Yds Allow 0-99": _alias("yds_allow_0_99"),
        "Yds Allow 100-199": _alias("yds_allow_100_199"),
        "Yds Allow 200-299": _alias("yds_allow_200_299"),
        "Yds Allow 300-399": _alias("yds_allow_300_399"),
        "Yds Allow 400-499": _alias("yds_allow_400_499"),
        "Yds Allow 500+": _alias("yds_allow_500_plus"),
    }

    def compute_points_from_rules(row: pd.Series, rules: List[Dict[str, Any]]) -> float:
        total = 0.0
        for rule in rules:
            name = rule.get("name")
            points = rule.get("points")
            if name is None or points is None:
                continue
            col = name_to_col.get(str(name))
            if not col:
                continue
            try:
                val = float(row.get(col, 0) or 0)
                total += val * float(points)
            except Exception:
                continue
        return round(total, 2)

    if full_scoring_rules:
        pts_ser = df.get("points")
        is_blank_points = pts_ser.isna() | (pts_ser.astype(str).str.strip() == "")
        if bool(is_blank_points.any()):
            df.loc[is_blank_points, "points"] = df.loc[is_blank_points].apply(
                lambda row: compute_points_from_rules(row, full_scoring_rules), axis=1
            )

    return df


# ----------------------------
# Save
# ----------------------------
def save_outputs(df: pd.DataFrame, year: int, week: int) -> None:
    """
    Write ``df`` to CSV and Parquet in the configured output directory.
    """
    if year == 0 and week == 0:
        stem = "yahoo_player_stats_multi_year_all_weeks"
    elif year == 0:
        stem = f"yahoo_player_stats_multi_year_week_{week}"
    elif week == 0:
        stem = f"yahoo_player_stats_{year}_all_weeks"
    else:
        stem = f"yahoo_player_stats_{year}_week_{week}"

    csv_path = OUTPUT_DIR / f"{stem}.csv"
    parquet_path = OUTPUT_DIR / f"{stem}.parquet"

    df.to_csv(csv_path, index=False)
    wrote_parquet = False
    try:
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        wrote_parquet = True
    except Exception:
        try:
            df.to_parquet(parquet_path, engine="fastparquet", index=False)
            wrote_parquet = True
        except Exception as e:
            print(f"[save_outputs] Parquet write failed (install pyarrow or fastparquet?): {e}")

    print(f"rows: {len(df):,}")
    print(f"saved_csv: {csv_path}")
    if wrote_parquet:
        print(f"saved_parquet: {parquet_path}")


# ----------------------------
# CLI with Timeout Resume Logic (auto-pause+retry)
# ----------------------------
def _auto_save_on_no_leagues(err_msg: str) -> bool:
    """Return True if error indicates season has no leagues to fetch."""
    return "no league ids returned" in err_msg.lower()


def _pause_then_retry_notice() -> None:
    print(f"[timeout] Transient timeout/denial detected. Sleeping {PAUSE_SECONDS_ON_TIMEOUT//60} minutes, then retrying...")
    time.sleep(PAUSE_SECONDS_ON_TIMEOUT)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Yahoo Fantasy NFL -> CSV + Parquet (lean XML version)")
    p.add_argument("--year", type=int, default=None, help="0 = walk backward from current year")
    p.add_argument("--week", type=int, default=None, help="0 = all weeks for the season(s)")
    p.add_argument("--league-key", type=str, default=None, help="Yahoo league_key like 402.l.12345")
    args = p.parse_args()

    if args.year is None:
        try:
            args.year = int(input("Enter season year (0 = start from current year and go backward): ").strip())
        except ValueError:
            args.year = 0
    if args.week is None:
        try:
            args.week = int(input("Enter week number (0 = all weeks): ").strip())
        except ValueError:
            args.week = 0

    year = args.year
    week = args.week
    league_key_arg = args.league_key

    if year == 0:
        all_frames: List[pd.DataFrame] = []
        fetched_years: Set[int] = set()

        current_year = datetime.now().year
        y = current_year
        while True:
            if y in fetched_years:
                y -= 1
                if y < 2000:
                    break
                continue

            print(f"\n--- Attempting to fetch data for year {y} ---")
            while True:
                try:
                    df_y = yahoo_fantasy_data(y, week, league_key=league_key_arg)
                    if df_y.empty:
                        print(f"No data returned for year {y}")
                        break  # exit retry loop for this year; will stop outer walk
                    print(f"Successfully fetched {len(df_y)} rows for year {y}")
                    all_frames.append(df_y)
                    fetched_years.add(y)
                    break  # success for this year
                except APITimeoutError as e:
                    msg = str(e)
                    print(f"API Timeout/Transient: {msg}")
                    if _auto_save_on_no_leagues(msg):
                        # save partials and stop entirely
                        if all_frames:
                            df = pd.concat(all_frames, ignore_index=True)
                            save_outputs(df, 0, week)
                        else:
                            print("No data collected.")
                        return
                    _pause_then_retry_notice()
                    continue  # retry same year after pause
                except Exception as e:
                    print(f"Error fetching data for year {y}: {e}")
                    # keep legacy behavior for non-timeout unknown errors:
                    if all_frames:
                        try:
                            response = input("\nSave collected data and stop here? (yes/no): ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            response = "yes"
                        if response in ('y', 'yes'):
                            df = pd.concat(all_frames, ignore_index=True)
                            save_outputs(df, 0, week)
                            return
                        else:
                            # try previous year
                            break
                    else:
                        # nothing collected yet; try previous year
                        break

            # move to prior year
            y -= 1
            if y < 2000:
                break

        if not all_frames:
            print("No data collected.")
            return

        df = pd.concat(all_frames, ignore_index=True)
        print(f"\n--- Successfully combined data from {len(all_frames)} years ---")

    else:
        # Single season path: auto-pause and retry indefinitely on timeout-ish errors
        while True:
            try:
                df = yahoo_fantasy_data(year, week, league_key=league_key_arg)
                break
            except APITimeoutError as e:
                msg = str(e)
                print(f"API Timeout/Transient: {msg}")
                if _auto_save_on_no_leagues(msg):
                    print("No data collected.")
                    return
                _pause_then_retry_notice()
                continue

    if df.empty:
        print("No data returned (check oauth, league_key/year, or week range).")
        return

    save_outputs(df, year, week)


if __name__ == "__main__":
    main()
