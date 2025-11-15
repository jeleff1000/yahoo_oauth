#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import math
import subprocess
import xml.etree.ElementTree as ET
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import pandas as pd
import requests

# --- utils for name normalization ---
try:
    from imports_and_utils import clean_name, apply_manual_mapping, OAuth2, yfa  # type: ignore
except Exception:
    def clean_name(x: str) -> str: return str(x or "").strip()
    def apply_manual_mapping(x: str) -> str: return x
    OAuth2 = None
    yfa = None

# --------------------------------------------------------------------------------------
# Paths (context-aware with backward compatibility)
# --------------------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent  # .../fantasy_football_data_scripts/player_stats
REPO_ROOT = SCRIPT_DIR.parent.parent  # .../fantasy_football_data_downloads

# Default paths for backward compatibility (when no context provided)
DEFAULT_OAUTH_PATH = REPO_ROOT / "oauth" / "Oauth.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "fantasy_football_data" / "player_data"

YAHOO_SCRIPT = SCRIPT_DIR / "yahoo_fantasy_data.py"
NFL_COMBINE_SCRIPT = SCRIPT_DIR / "combine_dst_to_nfl.py"

# Add parent directories to path for imports
# SCRIPT_DIR is .../multi_league/data_fetchers, so SCRIPT_DIR.parent is .../multi_league
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Try to import LeagueContext for multi-league support
try:
    from core.league_context import LeagueContext
    LEAGUE_CONTEXT_AVAILABLE = True
except ImportError:
    try:
        from multi_league.core.league_context import LeagueContext
        LEAGUE_CONTEXT_AVAILABLE = True
    except ImportError:
        LEAGUE_CONTEXT_AVAILABLE = False
        LeagueContext = None
        print("[context] Warning: LeagueContext not available")

# --------------------------------------------------------------------------------------
# Yahoo helpers (settings fetch/parse)
# --------------------------------------------------------------------------------------
def _coalesce_str_upper(*vals: Any) -> str:
    for v in vals:
        if v is not pd.NA and pd.notna(v):
            s = str(v).strip()
            if s:
                return s.upper()
    return ""

def _fetch_url_xml(url: str, oauth: OAuth2, max_retries: int = 5, backoff: float = 0.5) -> ET.Element:
    last_err = None
    for i in range(max_retries):
        try:
            r = oauth.session.get(url, timeout=30)
            r.raise_for_status()
            txt = (r.text or "")
            if "Request denied" in txt:
                raise RuntimeError("Request denied")
            # strip default namespace
            txt = pd.Series(txt).str.replace(r' xmlns="[^"]+"', "", n=1, regex=True).iloc[0]
            return ET.fromstring(txt)
        except Exception as e:
            last_err = e
            if i == max_retries - 1:
                raise
            time.sleep(backoff * (2 ** i))
    raise last_err or RuntimeError("unknown fetch error")

def _discover_league_key(oauth: OAuth2, year: int, league_key_arg: Optional[str]) -> Optional[str]:
    if league_key_arg:
        return league_key_arg.strip()
    if yfa is None:
        return None
    try:
        gm = yfa.Game(oauth, "nfl")
        keys = gm.league_ids(year=year)
        if keys:
            return keys[-1]
    except Exception:
        return None
    return None

def fetch_yahoo_dst_scoring(year: int, league_key_arg: Optional[str], oauth_path: Path = None, settings_dir: Path = None) -> Optional[Dict[str, float]]:
    if oauth_path is None:
        oauth_path = DEFAULT_OAUTH_PATH
    if settings_dir is None:
        # NEW location (as of 2025): league_settings/
        # OLD location (backwards compatibility): player_data/yahoo_league_settings/
        settings_dir = DEFAULT_OUTPUT_DIR / "league_settings"
        if not settings_dir.exists():
            settings_dir = DEFAULT_OUTPUT_DIR / "player_data" / "yahoo_league_settings"

    if OAuth2 is None or not oauth_path.exists():
        return None
    oauth = OAuth2(None, None, from_file=str(oauth_path))
    if not oauth.token_is_valid():
        oauth.refresh_access_token()
    league_key = _discover_league_key(oauth, year, league_key_arg)
    if not league_key:
        return None
    try:
        root = _fetch_url_xml(f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/settings", oauth)
    except Exception:
        return None

    # stat_id -> display_name
    name_by_id: Dict[str, str] = {}
    for s in root.findall("league/settings/stat_categories/stats/stat"):
        sid = (s.findtext("stat_id") or "").strip()
        nm  = (s.findtext("display_name") or "").strip()
        if sid and nm:
            name_by_id[sid] = nm

    # stat_id -> value
    modifiers: Dict[str, float] = {}
    for m in root.findall("league/settings/stat_modifiers/stats/stat"):
        sid = (m.findtext("stat_id") or "").strip()
        val = (m.findtext("value") or "").strip()
        try:
            modifiers[sid] = float(val)
        except Exception:
            pass

    # Points Allowed buckets
    pa = {k: 0.0 for k in ("PA_0","PA_1_6","PA_7_13","PA_14_20","PA_21_27","PA_28_34","PA_35_plus")}
    for s in root.findall("league/settings/stat_categories/stats/stat"):
        disp = (s.findtext("display_name") or "").strip()
        if disp.lower().startswith("points allowed"):
            for b in s.findall("stat_buckets/stat_bucket"):
                pts = (b.findtext("points") or "0").strip()
                try:
                    pts_val = float(pts)
                except Exception:
                    continue
                start = (b.findtext("range/start") or "").strip()
                end   = (b.findtext("range/end") or "").strip()
                maxv  = (b.findtext("range/max") or "").strip()
                rng = f"{start}-{end}" if start and end else (f"{start}-{maxv}" if start and maxv else start or maxv)
                rng = rng.replace(" ", "")
                if rng in ("0", "0-0"):
                    pa["PA_0"] = pts_val
                elif rng == "1-6":
                    pa["PA_1_6"] = pts_val
                elif rng == "7-13":
                    pa["PA_7_13"] = pts_val
                elif rng == "14-20":
                    pa["PA_14_20"] = pts_val
                elif rng == "21-27":
                    pa["PA_21_27"] = pts_val
                elif rng == "28-34":
                    pa["PA_28_34"] = pts_val
                elif rng in ("35+", "35-"):
                    pa["PA_35_plus"] = pts_val

    wanted = {
        "Sack": "Sack",
        "Interception": "Interception",
        "Fumble Recovery": "Fumble Recovery",
        "Touchdown": "Touchdown",
        "Safety": "Safety",
        "Kickoff and Punt Return Touchdowns": "Kickoff and Punt Return Touchdowns",
    }
    weights = {v: 0.0 for v in wanted.values()}
    for sid, nm in name_by_id.items():
        if nm in wanted and sid in modifiers:
            weights[nm] = modifiers[sid]

    scoring = {**weights, **pa}

    safe = league_key.replace(".", "_")
    out = settings_dir / f"yahoo_dst_scoring_{year}_{safe}.json"
    try:
        out.write_text(json.dumps({"year": year, "league_key": league_key, "dst_scoring": scoring}, indent=2), encoding="utf-8")
        print(f"[dst] Saved DST settings -> {out.name}")
    except Exception as e:
        print(f"[dst] Could not save DST settings: {e}")

    return scoring

def load_saved_dst_scoring(year: int, league_key_arg: Optional[str], settings_dir: Path = None) -> Optional[Dict[str, float]]:
    if settings_dir is None:
        # NEW location (as of 2025): league_settings/
        # OLD location (backwards compatibility): player_data/yahoo_league_settings/
        settings_dir = DEFAULT_OUTPUT_DIR / "league_settings"
        if not settings_dir.exists():
            settings_dir = DEFAULT_OUTPUT_DIR / "player_data" / "yahoo_league_settings"
    candidates: List[Path] = []
    if league_key_arg:
        safe = league_key_arg.replace(".", "_")
        p = settings_dir / f"yahoo_dst_scoring_{year}_{safe}.json"
        if p.exists():
            candidates.append(p)
    if not candidates:
        candidates = sorted(settings_dir.glob(f"yahoo_dst_scoring_{year}_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    try:
        obj = json.loads(candidates[0].read_text(encoding="utf-8"))
        scoring = obj.get("dst_scoring") or {}
        for k in [
            "Sack","Interception","Fumble Recovery","Touchdown","Safety",
            "Kickoff and Punt Return Touchdowns",
            "PA_0","PA_1_6","PA_7_13","PA_14_20","PA_21_27","PA_28_34","PA_35_plus"
        ]:
            scoring.setdefault(k, 0.0)
        print(f"[dst] Loaded saved DST settings: {candidates[0].name}")
        return scoring
    except Exception as e:
        print(f"[dst] Failed to read saved settings: {e}")
        return None

def load_saved_full_scoring(year: int, league_key_arg: Optional[str], settings_dir: Path = None) -> Optional[List[Dict[str, Any]]]:
    if settings_dir is None:
        # NEW location (as of 2025): league_settings/
        # OLD location (backwards compatibility): player_data/yahoo_league_settings/
        settings_dir = DEFAULT_OUTPUT_DIR / "league_settings"
        if not settings_dir.exists():
            settings_dir = DEFAULT_OUTPUT_DIR / "player_data" / "yahoo_league_settings"
    candidates: List[Path] = []
    if league_key_arg:
        safe = league_key_arg.replace(".", "_")
        p = settings_dir / f"yahoo_full_scoring_{year}_{safe}.json"
        if p.exists(): candidates.append(p)
    if not candidates:
        candidates = sorted(settings_dir.glob(f"yahoo_full_scoring_{year}_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    try:
        obj = json.loads(candidates[0].read_text(encoding="utf-8"))
        rules = obj.get("full_scoring")
        if isinstance(rules, list):
            return rules
    except Exception as e:
        print(f"[rules] Failed to read full scoring settings: {e}")
    return None

# --------------------------------------------------------------------------------------
# Points calculators - REMOVED (handled by transformation modules)
# --------------------------------------------------------------------------------------
# Points calculation has been moved to:
# - initial_import_v2.py (lines 903-949): Calls scoring_calculator for ALL players
# - player_stats_v2.py (lines 311-350): Also calculates points for ALL players
# This merge file now only preserves Yahoo's original 'points' column without calculation.

# --------------------------------------------------------------------------------------
# Helpers to run the other scripts & locate outputs
# --------------------------------------------------------------------------------------
def run_script(pyfile: Path, year: int, week: int, context_path: str = None) -> None:
    if not pyfile.exists():
        raise FileNotFoundError(f"Script not found: {pyfile}")
    cmd = [sys.executable, str(pyfile), "--year", str(year), "--week", str(week)]
    if context_path:
        cmd.extend(["--context", context_path])
    print(f"\n[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def expected_yahoo_parquet(year: int, week: int, output_dir: Path = None) -> List[Path]:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    if year == 0 and week == 0:
        return [output_dir / "yahoo_player_stats_multi_year_all_weeks.parquet"]
    if year == 0 and week > 0:
        return [output_dir / f"yahoo_player_stats_multi_year_week_{week}.parquet"]
    if year > 0 and week == 0:
        return [output_dir / f"yahoo_player_stats_{year}_all_weeks.parquet"]
    return [output_dir / f"yahoo_player_stats_{year}_week_{week}.parquet"]

def locate_parquet_by_signature(modified_after_ts: float, include_cols_any: List[str],
                                year: int, week: int, output_dir: Path = None, prefer_most_rows: bool = True) -> Optional[Path]:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    candidates: List[Tuple[Path, int]] = []
    for p in output_dir.glob("*.parquet"):
        if p.stat().st_mtime < modified_after_ts: continue
        try: df = pd.read_parquet(p, columns=None)
        except Exception: continue
        df_cols = set(map(str.lower, df.columns))
        if not any(col.lower() in df_cols for col in include_cols_any): continue
        if "year" in df.columns and "week" in df.columns:
            sub = df
            try:
                sub = sub[(pd.to_numeric(sub["year"], errors="coerce") == year) | (year == 0)]
                if week not in (None, 0):
                    sub = sub[pd.to_numeric(sub["week"], errors="coerce") == week]
            except Exception:
                pass
            if sub.empty: continue
            candidates.append((p, len(sub)))
        else:
            candidates.append((p, len(df)))
    if not candidates: return None
    if prefer_most_rows:
        max_rows = max(n for _, n in candidates)
        biggest = [p for p, n in candidates if n == max_rows]
        return biggest[0] if len(biggest) == 1 else max(biggest, key=lambda pp: pp.stat().st_mtime)
    return max((p for p, _ in candidates), key=lambda pp: pp.stat().st_mtime)

def load_yahoo(year: int, week: int, started_ts: float, output_dir: Path = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # First, check if an all-weeks aggregated file already exists
    for p in expected_yahoo_parquet(year, week, output_dir):
        if p.exists():
            print(f"[yahoo] reading {p}")
            return pd.read_parquet(p)

    # If week=0 (all weeks), try to aggregate individual weekly files
    if week == 0 and year > 0:
        weekly_files = sorted(output_dir.glob(f"yahoo_player_data_{year}_week_*.parquet"))
        if weekly_files:
            print(f"[yahoo] Found {len(weekly_files)} weekly files for {year}, aggregating...")
            weekly_dfs = []
            for wf in weekly_files:
                try:
                    df = pd.read_parquet(wf)
                    if not df.empty:
                        weekly_dfs.append(df)
                        print(f"        Loaded {wf.name} ({len(df):,} rows)")
                except Exception as e:
                    print(f"        Warning: Failed to load {wf.name}: {e}")

            if weekly_dfs:
                combined = pd.concat(weekly_dfs, ignore_index=True)
                print(f"[yahoo] Aggregated {len(combined):,} total rows from weekly files")

                # Save aggregated file for future use
                agg_path = output_dir / f"yahoo_player_stats_{year}_all_weeks.parquet"
                try:
                    combined.to_parquet(agg_path, index=False)
                    print(f"[yahoo] Saved aggregated file: {agg_path.name}")
                except Exception as e:
                    print(f"[yahoo] Warning: Could not save aggregated file: {e}")

                return combined
            else:
                print(f"[yahoo] Warning: No valid weekly files found for {year}")

    # Fallback to signature-based search
    sig = ["yahoo_player_id", "manager", "yahoo_position", "fantasy_position", "points"]
    found = locate_parquet_by_signature(started_ts, sig, year, week, output_dir)
    if not found:
        raise FileNotFoundError(f"Could not locate Yahoo parquet output for year={year}, week={week}")
    print(f"[yahoo] reading {found}")
    return pd.read_parquet(found)

def load_nfl(year: int, week: int, started_ts: float, output_dir: Path = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    sig = ["fantasy_points_half_ppr", "nfl_team"]
    found = locate_parquet_by_signature(started_ts, sig, year, week, output_dir)
    if not found:
        raise FileNotFoundError("Could not locate NFL (combined) parquet output.")
    print(f"[nfl] reading {found}")
    return pd.read_parquet(found)

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

# --- key helpers ----------------------------------------------------------------------
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "junior", "senior"}
_NAME_VARIATIONS = {
    # Handle common name variations
    'trenton': 'trent',
    'trent': 'trent',
    'steven': 'stephen',
    'stephen': 'stephen',
    'stefon': 'stephen',
    'mike': 'michael',
    'michael': 'michael',
    'bob': 'robert',
    'robert': 'robert',
    'bill': 'william',
    'william': 'william',
    'dan': 'daniel',
    'daniel': 'daniel',
    'chris': 'christopher',
    'christopher': 'christopher',
    'matt': 'matthew',
    'matthew': 'matthew',
    'josh': 'joshua',
    'joshua': 'joshua',
}

def _norm_player_for_key(x: str) -> str:
    s = apply_manual_mapping(clean_name(str(x or ""))).strip().lower()

    # Remove suffixes
    tokens = s.split()
    tokens = [t for t in tokens if t not in _SUFFIXES]
    s = ' '.join(tokens)

    # Normalize common first name variations
    if tokens:
        first_name = tokens[0]
        if first_name in _NAME_VARIATIONS:
            tokens[0] = _NAME_VARIATIONS[first_name]
            s = ' '.join(tokens)

    # Remove periods and extra spaces
    s = s.replace('.', '').replace('  ', ' ').strip()

    return s

def _create_alt_name_keys(name: str) -> list:
    """
    Create alternative name keys to handle middle names and first name variations.

    For "John Parker Romo", this creates:
    - "john parker romo" (full name)
    - "parker romo" (middle + last, in case first name is omitted in other dataset)
    - "john romo" (first + last, in case middle name is omitted in other dataset)

    For "Joshua Palmer", this creates:
    - "joshua palmer" (full name)
    - "josh palmer" (first name variation + last)

    Returns list of alternative normalized keys.
    """
    normalized = _norm_player_for_key(name)
    tokens = normalized.split()

    if len(tokens) <= 1:
        return [normalized]

    alternatives = [normalized]  # Always include full normalized name

    # For names with 3+ parts (first, middle(s), last)
    if len(tokens) >= 3:
        # Create "middle + last" variant (e.g., "Parker Romo" from "John Parker Romo")
        middle_last = ' '.join(tokens[1:])
        alternatives.append(middle_last)

        # Create "first + last" variant (e.g., "John Romo" from "John Parker Romo")
        first_last = f"{tokens[0]} {tokens[-1]}"
        alternatives.append(first_last)

    # For 2-part names, add first name variation if available
    if len(tokens) == 2:
        first_name = tokens[0]
        last_name = tokens[1]

        # Add variation with alternate first name (e.g., "Josh Palmer" if input was "Joshua Palmer")
        if first_name in _NAME_VARIATIONS:
            canonical = _NAME_VARIATIONS[first_name]
            # Find all variations that map to the same canonical form
            for variant, canon in _NAME_VARIATIONS.items():
                if canon == canonical and variant != first_name:
                    alternatives.append(f"{variant} {last_name}")

    return list(set(alternatives))  # Remove duplicates

def _last_name_from_full(name_lc: str) -> str:
    tokens = [t for t in name_lc.split() if t]
    tokens = [t for t in tokens if t not in _SUFFIXES]
    if not tokens:
        return ""
    last = tokens[-1]
    return last.replace("-", "").replace("'", "").replace(".", "")

def derive_player_last(df: pd.DataFrame, player_col: str = "player") -> pd.Series:
    if "player_last_name" in df.columns:
        s = df["player_last_name"].astype(str).str.lower()
        return s.apply(_last_name_from_full)
    pl = df[player_col].astype(str).fillna("").str.lower()
    return pl.apply(_last_name_from_full)

def derive_position_key(df: pd.DataFrame) -> pd.Series:
    for c in ["nfl_position", "position", "yahoo_position", "fantasy_position"]:
        if c in df.columns:
            return df[c].astype(str).str.upper().str.strip()
    # align to df.index (avoid single-length series)
    return pd.Series([""] * len(df), index=df.index)

def round_points(v, ndigits=2):
    try:
        return round(float(v), ndigits)
    except Exception:
        return math.nan

def with_keys(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    out = df.copy()
    out["_origin"] = origin
    out["_rid"] = range(len(out))
    if "player" not in out.columns:
        out["player"] = pd.NA

    # Store original player names before cleaning
    out["_original_player_name"] = out["player"].copy()

    out["player_key"] = out["player"].astype(str).apply(_norm_player_for_key)
    base = out["player_last_name"].astype(str).str.lower() if "player_last_name" in out.columns else out["player"].astype(str).str.lower()
    out["player_last_name_key"] = base.apply(_last_name_from_full)
    out["year_key"] = pd.to_numeric(out.get("year"), errors="coerce").astype("Int64")
    out["week_key"] = pd.to_numeric(out.get("week"), errors="coerce").astype("Int64")
    out["position_key"] = derive_position_key(out)

    # Points key for matching: Yahoo uses API points, NFL uses average of standard/PPR (≈ half-PPR)
    if origin == "yahoo":
        pts_src = out.get("points")  # Yahoo API provides actual points scored
    else:
        # NFL: Calculate half-PPR as average of fantasy_points (0 PPR) and fantasy_points_ppr (1 PPR)
        # This approximates 0.5 PPR scoring for better matching across 0/0.5/1 PPR leagues
        fp_standard = pd.to_numeric(out.get("fantasy_points"), errors="coerce")
        fp_ppr = pd.to_numeric(out.get("fantasy_points_ppr"), errors="coerce")

        if fp_standard is not None and fp_ppr is not None:
            # Average of standard and PPR = approximate half-PPR
            pts_src = (fp_standard + fp_ppr) / 2.0
        elif fp_standard is not None:
            pts_src = fp_standard  # Fallback to standard if PPR missing
        elif fp_ppr is not None:
            pts_src = fp_ppr  # Fallback to PPR if standard missing
        else:
            # Try fantasy_points_half_ppr if neither standard nor PPR available
            pts_src = out.get("fantasy_points_half_ppr")

    if pts_src is None:
        pts_src = pd.Series([math.nan] * len(out), index=out.index)
    out["points_key"] = pts_src.apply(round_points)
    return out

# --------------------------------------------------------------------------------------
# Matching & assembly
# --------------------------------------------------------------------------------------
def commit_one_to_one(matches: pd.DataFrame, y_col: str = "_y_rid", n_col: str = "_n_rid") -> pd.DataFrame:
    c_y = matches.groupby(y_col).size()
    c_n = matches.groupby(n_col).size()
    one_to_one = matches[matches[y_col].map(c_y.get) == 1]
    one_to_one = one_to_one[one_to_one[n_col].map(c_n.get) == 1]
    return one_to_one

def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return numeric Series for df[col]; if missing, return 0-valued Series aligned to df.index."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype="float64")

def layer_merge_fuzzy_name(layer_name: str, y_unmatched: pd.DataFrame, n_unmatched: pd.DataFrame,
                           verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fuzzy name matching layer that handles middle names and first name variations.

    Matches records where ANY alternative name key from Yahoo matches ANY alternative
    name key from NFL, with same year and week.

    This catches cases like:
    - "John Parker Romo" (Yahoo) vs "Parker Romo" (NFL)
    - "Joshua Palmer" (Yahoo) vs "Josh Palmer" (NFL)
    """
    if y_unmatched.empty or n_unmatched.empty:
        return pd.DataFrame(), y_unmatched, n_unmatched

    # Create mapping of record ID to alternative name keys
    y_keys = {}
    for idx, row in y_unmatched.iterrows():
        player_name = row.get("player", "")
        year = row.get("year_key")
        week = row.get("week_key")
        rid = row.get("_rid")

        alt_keys = _create_alt_name_keys(str(player_name))
        for alt_key in alt_keys:
            y_keys.setdefault((alt_key, year, week), []).append((idx, rid))

    # Find matches in NFL data
    matches = []
    for idx, row in n_unmatched.iterrows():
        player_name = row.get("player", "")
        year = row.get("year_key")
        week = row.get("week_key")
        n_rid = row.get("_rid")

        alt_keys = _create_alt_name_keys(str(player_name))
        for alt_key in alt_keys:
            if (alt_key, year, week) in y_keys:
                for y_idx, y_rid in y_keys[(alt_key, year, week)]:
                    # Merge the two records
                    y_row = y_unmatched.loc[y_idx].to_dict()
                    n_row = row.to_dict()
                    merged_row = {**y_row, **n_row, "_y_rid": y_rid, "_n_rid": n_rid}
                    matches.append(merged_row)
                break  # Only match once per NFL record

    if not matches:
        if verbose:
            print(f"\n[{layer_name}] No fuzzy matches found")
        return pd.DataFrame(), y_unmatched, n_unmatched

    merged = pd.DataFrame(matches)

    # Apply same filtering logic as regular layer_merge
    if not merged.empty:
        # Rename columns to have _Y and _N suffixes for consistency
        y_cols = [c for c in y_unmatched.columns if not c.startswith("_")]
        n_cols = [c for c in n_unmatched.columns if not c.startswith("_")]

        for col in y_cols:
            if col in merged.columns and f"{col}_Y" not in merged.columns:
                merged[f"{col}_Y"] = merged[col]
        for col in n_cols:
            if col in merged.columns and f"{col}_N" not in merged.columns:
                merged[f"{col}_N"] = merged[col]

    if verbose:
        y_count, n_count = len(y_unmatched), len(n_unmatched)
        matched_y = merged["_y_rid"].nunique() if not merged.empty else 0
        matched_n = merged["_n_rid"].nunique() if not merged.empty else 0
        print(f"\n[{layer_name}] candidates={y_count} x {n_count}  matched={matched_y} y_unmatched~{(y_count-matched_y)} n_unmatched~{(n_count-matched_n)}")

    one_to_one = commit_one_to_one(merged, y_col="_y_rid", n_col="_n_rid")
    if verbose: print(f"[{layer_name}] committed 1:1: {len(one_to_one):,}")
    matched_y_ids = set(one_to_one["_y_rid"].tolist()); matched_n_ids = set(one_to_one["_n_rid"].tolist())
    y_remaining = y_unmatched[~y_unmatched["_rid"].isin(matched_y_ids)].copy()
    n_remaining = n_unmatched[~n_unmatched["_rid"].isin(matched_n_ids)].copy()
    if verbose: print(f"[{layer_name}] remaining: yahoo={len(y_remaining):,} | nfl={len(n_remaining):,}")
    return one_to_one, y_remaining, n_remaining

def layer_merge(layer_name: str, y_unmatched: pd.DataFrame, n_unmatched: pd.DataFrame,
                key_cols: List[Tuple[str, str]], verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    left = y_unmatched.copy(); right = n_unmatched.copy()
    for yk, nk in key_cols:
        if yk not in left.columns: left[yk] = pd.NA
        if nk not in right.columns: right[nk] = pd.NA
    left["_y_rid"] = left["_rid"]; right["_n_rid"] = right["_rid"]
    on_map = {yk: nk for yk, nk in key_cols}
    merged = left.merge(right, left_on=list(on_map.keys()), right_on=list(on_map.values()),
                        how="inner", suffixes=("_Y","_N"), copy=False, validate="m:m")

    # --------- FILTERING LOGIC ---------
    # Accept matches when keys matched (player name, year, week)
    # Position mismatches are OK - Yahoo and NFL often use different position labels
    # Only reject obvious false positives (different normalized names)
    if not merged.empty:
        # Compare normalized player keys (which were used for the merge)
        # If player_key matched, then the normalized names already matched
        key_match = merged.get("player_key_Y", pd.Series("", index=merged.index)) == \
                    merged.get("player_key_N", pd.Series("", index=merged.index))

        # For defensive position data quality: Defense often has position mismatches
        # (e.g., Yahoo="DEF", NFL="DB"/"LB"/"DL")
        yahoo_is_def = merged.get("position_key_Y", pd.Series("", index=merged.index)).astype(str).str.upper() == "DEF"
        nfl_is_def = merged.get("position_key_N", pd.Series("", index=merged.index)).astype(str).str.upper() == "DEF"

        # STRICTER DEF MATCHING: When Yahoo is DEF, NFL must also be DEF
        # This prevents matches like "DeeJay Dallas" (RB) with "Dallas" (DEF)
        both_are_def = yahoo_is_def & nfl_is_def

        # STRICTER MATCHING FOR YAHOO PLAYERS WITH 0 POINTS
        # Zero-point Yahoo players need position match to avoid false positives
        # (e.g., bye weeks, injured players, or misidentified players)
        yahoo_points = pd.to_numeric(merged.get("points_Y", pd.Series(0, index=merged.index)), errors="coerce").fillna(0)
        yahoo_zero_points = (yahoo_points == 0)

        # Position keys must match for zero-point players (unless BOTH are DEF)
        position_match = merged.get("position_key_Y", pd.Series("", index=merged.index)) == \
                        merged.get("position_key_N", pd.Series("", index=merged.index))

        # Keep match if:
        # 1. Player keys match (normalized names matched), OR
        # 2. BOTH sides are DEF position (defense-specific matching), OR
        # 3. Yahoo has non-zero points AND not a lone DEF (normal matching), OR
        # 4. Yahoo has zero points AND position matches AND not a lone DEF (stricter matching)
        keep_mask = key_match | both_are_def | ((~yahoo_zero_points) & (~yahoo_is_def)) | ((yahoo_zero_points & position_match) & (~yahoo_is_def))

        merged = merged[keep_mask]
    # ------------------------------------------------------------------------------------

    if verbose:
        y_count, n_count = len(left), len(right)
        matched_y = merged["_y_rid"].nunique() if not merged.empty else 0
        matched_n = merged["_n_rid"].nunique() if not merged.empty else 0
        print(f"\n[{layer_name}] candidates={y_count} x {n_count}  matched={matched_y} y_unmatched~{(y_count-matched_y)} n_unmatched~{(n_count-matched_n)}")
    one_to_one = commit_one_to_one(merged, y_col="_y_rid", n_col="_n_rid")
    if verbose: print(f"[{layer_name}] committed 1:1: {len(one_to_one):,}")
    matched_y_ids = set(one_to_one["_y_rid"].tolist()); matched_n_ids = set(one_to_one["_n_rid"].tolist())
    y_remaining = left[~left["_y_rid"].isin(matched_y_ids)].copy()
    n_remaining = right[~right["_n_rid"].isin(matched_n_ids)].copy()
    if verbose: print(f"[{layer_name}] remaining: yahoo={len(y_remaining):,} | nfl={len(n_remaining):,}")
    return one_to_one, y_remaining, n_remaining

def assemble_rows(one_to_one: pd.DataFrame) -> pd.DataFrame:
    if one_to_one.empty: return pd.DataFrame()
    df = one_to_one
    base_cols = [c for c in df.columns if not (c.endswith("_Y") or c.endswith("_N"))]
    out = df[base_cols].copy()
    y_cols = [c for c in df.columns if c.endswith("_Y")]
    n_cols = [c for c in df.columns if c.endswith("_N")]
    y_bases = {c[:-2] for c in y_cols}
    n_bases = {c[:-2] for c in n_cols}
    both   = sorted(y_bases & n_bases)
    only_y = sorted(y_bases - n_bases)
    only_n = sorted(n_bases - y_bases)
    for b in both:
        colY = f"{b}_Y"; colN = f"{b}_N"
        # Special handling for player names and original names
        if b == "_original_player_name":
            # Prefer Yahoo's original name, fallback to NFL's original name
            out[b] = df[colY].where(df[colY].notna() & (df[colY].astype(str).str.strip() != ""), df[colN])
        elif b == "player":
            # Store both for now, will restore later
            out[b] = df[colN].where(df[colN].notna(), df[colY])
        elif b == "points":
            # Always use Yahoo points (the source of truth for fantasy scoring)
            out[b] = df[colY]
        elif b in ("fantasy_position", "yahoo_position", "primary_position"):
            # Always use Yahoo position data (roster slots, eligibility)
            # fantasy_position = roster slot (QB, RB1, FLEX, BN, etc.)
            # NFL data should never have fantasy_position for non-DEF players
            out[b] = df[colY]
        elif b in ("nfl_team", "team"):
            # CRITICAL: Always use NFL team data (Yahoo shows CURRENT team, not historical)
            # Example: Tom Brady 2012 shows Tampa in Yahoo but New England in NFL (correct)
            out[b] = df[colN].where(df[colN].notna(), df[colY])
        else:
            out[b] = df[colN].where(df[colN].notna(), df[colY])
    for b in only_y:
        if b not in out.columns: out[b] = df[f"{b}_Y"]
    for b in only_n:
        if b not in out.columns: out[b] = df[f"{b}_N"]
    out = out[[c for c in out.columns if not c.startswith("_")]]
    return out

def finalize_union(matched_blocks: List[pd.DataFrame], y_remaining: pd.DataFrame, n_remaining: pd.DataFrame) -> pd.DataFrame:
    matched = pd.concat([blk for blk in matched_blocks if not blk.empty], ignore_index=True) if matched_blocks else pd.DataFrame()
    y_only = y_remaining.drop(columns=[c for c in y_remaining.columns if c.startswith("_")], errors="ignore")
    n_only = n_remaining.drop(columns=[c for c in n_remaining.columns if c.startswith("_")], errors="ignore")
    final = pd.concat([matched, n_only, y_only], ignore_index=True, sort=False)

    if "player" in final.columns:
        final["player"] = final["player"].astype(str).apply(lambda s: apply_manual_mapping(clean_name(s)))
    if "2-pt" in final.columns:
        final["2-pt"] = pd.to_numeric(final["2-pt"], errors="coerce").fillna(0).astype("Int64")

    # Position column should already exist (created before filtering)
    # But if somehow it's missing, recreate it with fallback logic
    # CRITICAL: yahoo_position = player's actual position from Yahoo (QB, RB, WR, etc.)
    #           nfl_position = player's actual position from NFLverse
    #           fantasy_position = roster SLOT from Yahoo (QB, RB1, FLEX, BN, IR, etc.)
    #           position = unified column for optimal lineup calculations

    if "position" not in final.columns or final["position"].isna().all():
        print("[position] Recreating unified position column (should have been created earlier)")

        def is_valid_position(val):
            """Check if position value is valid (not null, empty, 'nan', '0', etc.)"""
            if pd.isna(val):
                return False
            val_str = str(val).strip().upper()
            return val_str not in ("", "NAN", "NONE", "0", "NULL", "N/A")

        # Get yahoo_position and nfl_position columns
        yahoo_pos = final["yahoo_position"] if "yahoo_position" in final.columns else pd.Series([pd.NA] * len(final), index=final.index)
        nfl_pos = final["nfl_position"] if "nfl_position" in final.columns else pd.Series([pd.NA] * len(final), index=final.index)

        # Create position column: prefer yahoo_position, fallback to nfl_position
        final["position"] = yahoo_pos.where(yahoo_pos.apply(is_valid_position), nfl_pos)

        # Clean up position column: convert invalid values to NA
        final["position"] = final["position"].apply(lambda x: x if is_valid_position(x) else pd.NA)

    # Ensure position is string dtype for downstream processing
    final["position"] = final["position"].astype("string")

    # Create column aliases for NFLverse column names (full names → abbreviated names)
    # NFLverse uses: passing_yards, rushing_yards, receiving_yards
    # Code expects: pass_yds, rush_yds, rec_yds
    column_aliases = {
        'passing_yards': 'pass_yds',
        'rushing_yards': 'rush_yds',
        'receiving_yards': 'rec_yds',
        'passing_tds': 'pass_td',
        'rushing_tds': 'rush_td',
        'receiving_tds': 'rec_td',
        'receptions': 'rec',
        'passing_interceptions': 'pass_int',
        'fumbles_lost': 'fum_lost',
    }

    # Apply aliases (create new columns with abbreviated names)
    for full_name, abbrev_name in column_aliases.items():
        if full_name in final.columns and abbrev_name not in final.columns:
            final[abbrev_name] = final[full_name]

    # NFL-only safety: if Yahoo points missing, seed with NFL half-PPR now (will be refined later if rules exist)
    if "points" not in final.columns:
        final["points"] = pd.NA
    if "fantasy_points_half_ppr" in final.columns:
        fp = pd.to_numeric(final["fantasy_points_half_ppr"], errors="coerce")
        need = final["points"].isna() & fp.notna()
        final.loc[need, "points"] = fp.round(2)

    return final

# --- tidy column order ----------------------------------------------------------------
def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    IDENTITY = [
        "league_id","player","player_last_name","position","nfl_position","fantasy_position","yahoo_position",
        "year","week","season_type","nfl_team","opponent_nfl_team",
        "yahoo_player_id","NFL_player_id","headshot_url","bye","manager","manager_year","opponent","opponent_year",
    ]
    SCORING = [
        "points_original","points_dst_from_yahoo_settings","points",
        "fantasy_points_zero_ppr","fantasy_points_ppr","fantasy_points_half_ppr",
        "2-pt","fum_lost","fum_rec","fum_ret_td","ret_td","blk_kick",
    ]
    PASSING = ["pass_yds","pass_td","passing_interceptions","attempts","completions",
               "passing_air_yards","passing_yards_after_catch","passing_first_downs","passing_epa","passing_cpoe","pacr"]
    RUSHING = ["rush_att","rush_yds","rush_td","rushing_fumbles","rushing_fumbles_lost","rushing_first_downs","rushing_epa","rushing_2pt_conversions"]
    RECEIVING = ["targets","rec","rec_yds","rec_td","receiving_fumbles","receiving_fumbles_lost",
                 "receiving_air_yards","receiving_yards_after_catch","receiving_first_downs","receiving_epa","receiving_2pt_conversions",
                 "target_share","air_yards_share","racr","wopr"]
    KICKING = [
        "fg_att","fg_made","fg_miss","fg_pct","fg_long","fg_yds",
        "pat_att","pat_made","pat_missed","pat_blocked","pat_pct",
        "fg_blocked","fg_blocked_distance","fg_blocked_list",
        "fg_made_0_19","fg_made_20_29","fg_made_30_39","fg_made_40_49","fg_made_50_59","fg_made_60_",
        "fg_missed_0_19","fg_missed_20_29","fg_missed_30_39","fg_missed_40_49","fg_missed_50_59","fg_missed_60_",
        "fg_made_distance","fg_missed_distance","fg_made_list","fg_missed_list",
        "gwfg_att","gwfg_made","gwfg_missed","gwfg_blocked","gwfg_distance",
        "blocked_fg","blocked_pat","blocked_gwfg","blocked_fg_distance","blocked_fg_list",
    ]
    RETURNS = ["punt_returns","punt_return_yards","kickoff_returns","kickoff_return_yards"]
    DEF_STATS = [
        "def_sacks","def_sack_yards","def_qb_hits","def_interceptions","def_pass_defended",
        "def_tackles_solo","def_tackle_assists","def_tackles_with_assist","def_tackles_for_loss","def_tackles_for_loss_yards",
        "def_fumbles","def_fumbles_forced","def_safeties","def_tds","misc_yards","penalties","penalty_yards","timeouts",
    ]
    ALLOWED = [
        "pts_allow","dst_points_allowed","points_allowed","pass_yds_allowed","rushing_yds_allowed","total_yds_allowed",
        "two_pt_conversions_allowed","fg_made_allowed","pat_made_allowed","passing_tds_allowed","rushing_tds_allowed",
        "receiving_tds_allowed","special_teams_tds_allowed","def_tds_allowed",
    ]
    BUCKETS_PTS = ["pts_allow_0","pts_allow_1_6","pts_allow_7_13","pts_allow_14_20","pts_allow_21_27","pts_allow_28_34","pts_allow_35_plus"]
    BUCKETS_YDS = ["yds_allow_neg","yds_allow_0_99","yds_allow_100_199","yds_allow_200_299","yds_allow_300_399","yds_allow_400_499","yds_allow_500_plus"]
    INTERNAL = ["_origin","_rid","player_key","player_last_name_key","position_key","points_key","year_key","week_key"]

    groups = [IDENTITY,SCORING,PASSING,RUSHING,RECEIVING,KICKING,RETURNS,DEF_STATS,ALLOWED,BUCKETS_PTS,BUCKETS_YDS,INTERNAL]
    ordered: List[str] = []
    for g in groups: ordered.extend([c for c in g if c in df.columns])
    remainder = [c for c in df.columns if c not in ordered]
    ordered += sorted(remainder)
    return df.reindex(columns=ordered)

# DST points calculation removed - handled by transformation modules

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge Yahoo and NFL player stats")
    parser.add_argument("--year", type=int, default=None, help="Season year (0 = all years)")
    parser.add_argument("--week", type=int, default=None, help="Week number (0 = all weeks)")
    parser.add_argument("--league-key", type=str, default=None, help="Yahoo league key")
    parser.add_argument("--context", type=str, default=None, help="Path to league_context.json (multi-league support)")
    args = parser.parse_args()

    # ===== CONTEXT-AWARE PATH SETUP =====
    # Load context if provided for multi-league support
    ctx = None
    oauth_path = DEFAULT_OAUTH_PATH
    output_dir = DEFAULT_OUTPUT_DIR

    if args.context and LEAGUE_CONTEXT_AVAILABLE:
        try:
            ctx = LeagueContext.load(args.context)
            # Use context paths
            oauth_path = Path(ctx.oauth_file_path) if ctx.oauth_file_path else DEFAULT_OAUTH_PATH
            output_dir = Path(ctx.player_data_directory)
            print(f"[context] Using league: {ctx.league_name} ({ctx.league_id})")
            print(f"[context] OAuth: {oauth_path}")
            print(f"[context] Output: {output_dir}")
        except Exception as e:
            print(f"[context] Warning: Could not load context from {args.context}: {e}")
            print(f"[context] Falling back to default paths")
    elif args.context and not LEAGUE_CONTEXT_AVAILABLE:
        print(f"[context] Warning: LeagueContext not available (multi_league.core.league_context not found)")
        print(f"[context] Falling back to default paths")

    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = output_dir / "yahoo_league_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)

    # ===== ARGUMENT HANDLING =====
    # If no CLI args provided, fall back to interactive prompts
    if args.year is None:
        raw = input("Enter: season year (0 = start from current year and go backward) [optional league_key]:\n").strip()
        parts = raw.split()
        try:
            year = int(parts[0]) if parts else 0
        except Exception:
            year = 0
        league_key_arg = parts[1] if len(parts) > 1 else None
    else:
        year = args.year
        league_key_arg = args.league_key

    # Ask for week only if year != 0 and not provided via CLI
    if args.week is None:
        if year != 0:
            try:
                week = int(input("Enter week number (0 = all weeks): ").strip())
            except Exception:
                week = 0
        else:
            week = 0
    else:
        week = args.week

    # Pass context to both scripts so they save to the correct league-specific directory
    context_path = args.context if args.context else None

    # Try to load existing Yahoo data first (avoid redundant API calls if Phase 1 already fetched)
    y_df = None
    for p in expected_yahoo_parquet(year, week, output_dir):
        if p.exists():
            print(f"[yahoo] Using existing file: {p}")
            try:
                y_df = pd.read_parquet(p)
                break
            except Exception as e:
                print(f"[yahoo] Failed to read {p}: {e}")

    # If Yahoo data not found, fetch it
    if y_df is None:
        print(f"[yahoo] No existing file found, fetching from API...")
        t0 = time.time()
        run_script(YAHOO_SCRIPT, year, week, context_path=context_path)
        y_df = load_yahoo(year, week, started_ts=t0, output_dir=output_dir)

    # Try to load existing NFL combined data first
    # ALWAYS load ALL years (year=0) for league-agnostic NFL data
    n_df = None
    nfl_combined_candidates = [
        output_dir / "nfl_stats_merged_0_all_weeks.parquet",  # All years, all weeks
        output_dir / "nfl_stats_merged_0_allweeks.parquet",
        output_dir / "nfl_combined_0_all_weeks.parquet",
        output_dir / "nfl_stats_merged_all_years.parquet",   # Alternative naming
        output_dir / "nfl_combined_all_years.parquet",
    ]
    for p in nfl_combined_candidates:
        if p.exists():
            print(f"[nfl] Using existing ALL-YEARS file: {p}")
            try:
                n_df = pd.read_parquet(p)
                print(f"[nfl] Loaded ALL years NFL data: {len(n_df):,} rows")
                break
            except Exception as e:
                print(f"[nfl] Failed to read {p}: {e}")

    # If NFL combined data not found, create it
    # IMPORTANT: Always fetch ALL NFL years (1999+) for league-agnostic data
    # Yahoo data is league-specific, but NFL data should be comprehensive
    if n_df is None:
        print(f"[nfl] No existing combined file found, creating it...")
        print(f"[nfl] Fetching ALL NFL years (1999+) for league-agnostic data...")
        t1 = time.time()
        # Pass year=0 to get ALL years of NFL data, not filtered by year parameter
        run_script(NFL_COMBINE_SCRIPT, 0, week, context_path=context_path)
        n_df = load_nfl(0, week, started_ts=t1, output_dir=output_dir)

    # Filter Yahoo data by year/week (league-specific)
    if "year" in y_df.columns and year != 0:
        y_df = y_df[pd.to_numeric(y_df["year"], errors="coerce") == year]
    if "week" in y_df.columns and week != 0:
        y_df = y_df[pd.to_numeric(y_df["week"], errors="coerce") == week]

    # CRITICAL FIX: Filter NFL data by year to prevent duplicates
    # When year=0 (all years), keep all NFL years for historical data
    # When year>0 (specific year), filter to that year to match Yahoo data scope
    # This ensures yahoo_nfl_merged_{YEAR}_all_weeks.parquet only contains data for {YEAR}
    if "year" in n_df.columns and year != 0:
        n_df = n_df[pd.to_numeric(n_df["year"], errors="coerce") == year]
    if "week" in n_df.columns and week != 0:
        n_df = n_df[pd.to_numeric(n_df["week"], errors="coerce") == week]

    # CRITICAL: Filter NFL data to max Yahoo week (prevent incomplete week data)
    # This ensures we don't include partial week data from NFLverse that Yahoo hasn't fetched yet
    if "week" in y_df.columns and "week" in n_df.columns and week == 0:
        # When merging all weeks, use Yahoo data as source of truth for completed weeks
        # For each year, filter NFL weeks to match Yahoo's max week
        if "year" in y_df.columns and "year" in n_df.columns:
            # Find max week per year in Yahoo data
            yahoo_max_weeks = (
                y_df.groupby(pd.to_numeric(y_df["year"], errors="coerce"))["week"]
                .apply(lambda x: pd.to_numeric(x, errors="coerce").max())
                .to_dict()
            )

            if yahoo_max_weeks:
                print(f"[nfl] Filtering to completed weeks per year (based on Yahoo data):")
                for yr, max_wk in sorted(yahoo_max_weeks.items()):
                    if pd.notna(yr) and pd.notna(max_wk):
                        print(f"  {int(yr)}: weeks 1-{int(max_wk)}")

                # Filter NFL data: for each year, only keep weeks <= max Yahoo week for that year
                nfl_year_numeric = pd.to_numeric(n_df["year"], errors="coerce")
                nfl_week_numeric = pd.to_numeric(n_df["week"], errors="coerce")
                initial_nfl_count = len(n_df)

                # Build mask: for each row, check if week <= max_week for that year
                keep_mask = pd.Series([False] * len(n_df), index=n_df.index)
                for yr, max_wk in yahoo_max_weeks.items():
                    if pd.notna(yr) and pd.notna(max_wk):
                        year_mask = (nfl_year_numeric == yr) & (nfl_week_numeric <= max_wk)
                        keep_mask |= year_mask

                n_df = n_df[keep_mask]
                filtered_count = initial_nfl_count - len(n_df)

                if filtered_count > 0:
                    print(f"[nfl] Excluded {filtered_count:,} rows from incomplete weeks")
                else:
                    print(f"[nfl] All data within completed weeks")
        else:
            # Fallback: if year column missing, use global max week
            yahoo_weeks = pd.to_numeric(y_df["week"], errors="coerce").dropna()
            if len(yahoo_weeks) > 0:
                max_yahoo_week = int(yahoo_weeks.max())

                # Filter NFL data to only include weeks up to max Yahoo week
                nfl_week_numeric = pd.to_numeric(n_df["week"], errors="coerce")
                initial_nfl_count = len(n_df)
                n_df = n_df[nfl_week_numeric <= max_yahoo_week]
                filtered_count = initial_nfl_count - len(n_df)

                if filtered_count > 0:
                    print(f"[nfl] Filtered to weeks 1-{max_yahoo_week} (max Yahoo week)")
                    print(f"[nfl] Excluded {filtered_count:,} rows from incomplete weeks (weeks > {max_yahoo_week})")
                else:
                    print(f"[nfl] All data within completed weeks (max week: {max_yahoo_week})")

    # Log filtering results
    if year != 0:
        print(f"[yahoo] Filtered to year: {year}")
        print(f"[nfl] Filtered to year: {year}")
    else:
        nfl_years = sorted(n_df["year"].dropna().unique()) if "year" in n_df.columns else []
        if nfl_years:
            print(f"[nfl] Loaded ALL NFL years: {min(nfl_years)}-{max(nfl_years)} (league-agnostic)")

    # If week parameter specified (not 0), inform user about filtering
    if week != 0:
        print(f"[nfl+yahoo] Both filtered to week: {week}")

    print(f"\n[yahoo rows] {len(y_df):,}")
    print(f"[nfl rows]   {len(n_df):,}")

    # CRITICAL: Create unified position column BEFORE filtering
    # This ensures we filter on the actual position we'll use (Yahoo preferred, NFL fallback)
    # Not just NFL position which might differ from Yahoo position
    def create_unified_position(df, yahoo_col_name="yahoo_position", nfl_col_name="nfl_position"):
        """Create unified position column: prefer Yahoo, fallback to NFL (with mapping)"""
        def is_valid_position(val):
            if pd.isna(val):
                return False
            val_str = str(val).strip().upper()
            return val_str not in ("", "NAN", "NONE", "0", "NULL", "N/A")

        # Position equivalencies: map NFL-specific positions to Yahoo positions
        # This ensures FB (fullback) data merges with RB (running back) rosters
        position_mapping = {
            'FB': 'RB',   # Fullback → Running Back
            'HB': 'RB',   # Halfback → Running Back (historical)
        }

        yahoo_pos = df[yahoo_col_name] if yahoo_col_name in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
        nfl_pos = df[nfl_col_name] if nfl_col_name in df.columns else pd.Series([pd.NA] * len(df), index=df.index)

        # Apply position mapping to NFL position
        nfl_pos_mapped = nfl_pos.apply(
            lambda x: position_mapping.get(str(x).strip().upper(), x) if is_valid_position(x) else x
        )

        # Prefer yahoo_position, fallback to mapped nfl_position
        unified = yahoo_pos.where(yahoo_pos.apply(is_valid_position), nfl_pos_mapped)
        # Clean up: convert invalid values to NA
        unified = unified.apply(lambda x: x if is_valid_position(x) else pd.NA)
        return unified.astype("string")

    # Add unified position column to both Yahoo and NFL data
    y_df["position"] = create_unified_position(y_df, "yahoo_position", "nfl_position")
    n_df["position"] = create_unified_position(n_df, "yahoo_position", "nfl_position")

    print(f"[position] Created unified position column (Yahoo preferred, NFL fallback)")

    # ===================== NORMALIZE NFL TEAM CODES BEFORE MERGE =====================
    # Normalize nfl_team codes to match between Yahoo and NFLverse sources
    # This handles team relocations and ambiguous codes (e.g., 'LA' for Rams)
    def normalize_nfl_team_codes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize nfl_team codes to match between Yahoo and NFLverse.

        Generic mapping handles:
        - Team relocations (SD→LAC, STL→LAR, OAK→LV)
        - Ambiguous codes (LA→LAR to distinguish from LAC)

        This ensures Yahoo data using 'LAR' matches NFLverse data using 'LA'.
        """
        if "nfl_team" not in df.columns:
            return df

        df = df.copy()

        # Generic team code normalization (league-agnostic, scalable)
        team_code_normalization = {
            'LA': 'LAR',   # Disambiguate Los Angeles (Rams vs Chargers)
            'STL': 'LAR',  # St. Louis Rams → Los Angeles Rams (2016+)
            'SD': 'LAC',   # San Diego Chargers → Los Angeles Chargers (2017+)
            'OAK': 'LV',   # Oakland Raiders → Las Vegas Raiders (2020+)
        }

        df['nfl_team'] = df['nfl_team'].replace(team_code_normalization)

        return df

    # ===================== NORMALIZE DEFENSE TEAM NAMES BEFORE MERGE =====================
    # Yahoo API returns defense teams with shortened names (e.g., "Los Angeles" instead of "Los Angeles Chargers")
    # Expand these to full team names based on nfl_team code to ensure merge keys match
    def expand_defense_team_names(df: pd.DataFrame) -> pd.DataFrame:
        """Expand shortened defense team names to full names using nfl_team code."""
        if "player" not in df.columns or "nfl_team" not in df.columns:
            return df

        df = df.copy()

        # Map team codes to full defense names
        team_name_map = {
            # Los Angeles teams
            "LAC": "Los Angeles Chargers",
            "SD": "Los Angeles Chargers",  # Historical (pre-2017)
            "LAR": "Los Angeles Rams",
            "LA": "Los Angeles Rams",  # Alternative code
            "STL": "Los Angeles Rams",  # Historical (pre-2016)

            # New York teams
            "NYG": "New York Giants",
            "NYJ": "New York Jets",

            # Other relocated/multi-name teams
            "LV": "Las Vegas",
            "OAK": "Las Vegas",  # Historical (pre-2020)
        }

        # Identify defense rows (position = DEF)
        is_def = df.get("position", pd.Series(dtype=str)).astype(str).str.upper() == "DEF"

        if not is_def.any():
            return df  # No defenses to normalize

        # For each defense row, check if player name is shortened
        for idx in df[is_def].index:
            player_name = str(df.loc[idx, "player"]).strip()
            team_code = str(df.loc[idx, "nfl_team"]).strip().upper()

            # Check if team code maps to a full name
            if team_code in team_name_map:
                full_name = team_name_map[team_code]

                # Only update if current name is not already the full name
                # (handles cases where name is shortened like "Los Angeles" or "New York")
                if player_name != full_name and (
                    player_name.startswith("Los Angeles") or
                    player_name.startswith("New York") or
                    player_name == "Los Angeles" or
                    player_name == "New York" or
                    player_name.startswith("Las Vegas") or
                    player_name == "Las Vegas"
                ):
                    df.loc[idx, "player"] = full_name

        return df

    # Apply team code normalization to both Yahoo and NFL data (FIRST)
    y_df = normalize_nfl_team_codes(y_df)
    n_df = normalize_nfl_team_codes(n_df)
    print(f"[normalize] Normalized nfl_team codes (LA->LAR, SD->LAC, OAK->LV, STL->LAR)")

    # Apply team name expansion to both Yahoo and NFL data (SECOND)
    y_df = expand_defense_team_names(y_df)
    n_df = expand_defense_team_names(n_df)
    print(f"[normalize] Expanded defense team names for multi-city teams")

    yK = with_keys(y_df, origin="yahoo")
    nK = with_keys(n_df, origin="nfl")

    # DYNAMIC POSITION FILTERING - LEAGUE-AWARE
    # Filter NFL data to only include positions rosterable in this league
    # For standard leagues: {QB, RB, WR, TE, K, DEF} → filters out IDP (LB, DB, DL, etc.)
    # For IDP leagues: {QB, RB, WR, TE, K, DEF, LB, DB, DL} → keeps IDP positions
    #
    # IMPORTANT: This filters ALL years of NFL data (1999+) to eligible positions
    # Example: If league started in 2014, you'll still get Dan Marino's 1999 stats
    #          as long as QB is an eligible position in your league settings

    # Strategy:
    # 1. If Yahoo data exists for this year: use positions from Yahoo data
    # 2. If no Yahoo data (out-of-scope years): use positions from roster settings
    # 3. Fallback: use standard fantasy positions {QB, RB, WR, TE, K, DEF}

    rosterable_positions = None

    # Try to get positions from Yahoo data first (most accurate for this year)
    yahoo_position_col = None
    for col in ["yahoo_position", "primary_position", "position"]:
        if col in y_df.columns:
            yahoo_position_col = col
            break

    if yahoo_position_col and not y_df.empty:
        # Use positions from Yahoo data for this specific year
        rosterable_positions = set(y_df[yahoo_position_col].dropna().astype(str).str.upper().unique())
        print(f"[filter] Using roster positions from Yahoo data: {sorted(rosterable_positions)}")

    # Fallback: Load roster positions from league settings (for out-of-scope years)
    if rosterable_positions is None and ctx:
        try:
            # Load roster settings to determine which positions are rosterable
            settings_dir = output_dir / "yahoo_league_settings"
            if settings_dir.exists():
                import json
                normalized_league_id = ctx.league_id.replace(".", "_")

                # Try to find settings for this year or nearest year
                settings_files = sorted(settings_dir.glob(f"*{year}*{normalized_league_id}.json"))
                if not settings_files:
                    # Try any year's settings (assume roster structure is consistent)
                    settings_files = sorted(settings_dir.glob(f"*{normalized_league_id}.json"))

                if settings_files:
                    with open(settings_files[0], 'r') as f:
                        settings_data = json.load(f)

                    roster_positions = settings_data.get("roster_positions", [])
                    rosterable_positions = set()
                    for pos_info in roster_positions:
                        position = pos_info.get("position", "")
                        # Exclude bench and IR
                        if position and position not in ("BN", "IR"):
                            rosterable_positions.add(position.upper())

                    if rosterable_positions:
                        print(f"[filter] Using roster positions from league settings: {sorted(rosterable_positions)}")
        except Exception as e:
            print(f"[filter] Could not load roster settings: {e}")

    # Final fallback: standard fantasy positions
    if rosterable_positions is None:
        rosterable_positions = {"QB", "RB", "WR", "TE", "K", "DEF"}
        print(f"[filter] Using default fantasy positions: {sorted(rosterable_positions)}")

    # Apply position filter to NFL data using the UNIFIED position column
    # This ensures we filter based on the actual position we'll use (Yahoo preferred, NFL fallback)
    if "position" in n_df.columns and rosterable_positions:
        nfl_before = len(nK)

        # Expand flex positions to their constituent positions
        # E.g., "W/R/T" in roster settings means WR, RB, TE are all rosterable
        expanded_positions = set()
        for pos in rosterable_positions:
            if "/" in pos:
                # Flex position - split and add all parts
                expanded_positions.update(p.strip() for p in pos.split("/"))
            else:
                expanded_positions.add(pos)

        # Add position equivalencies (NFL positions that map to Yahoo positions)
        # This allows NFL-specific positions to match Yahoo roster positions
        # Example: FB (fullback) in NFL data should match RB in Yahoo data
        position_equivalencies = {
            'FB': 'RB',   # Fullback → Running Back (Yahoo treats FB as RB)
            'HB': 'RB',   # Halfback → Running Back (historical, rarely used)
        }

        # Expand equivalencies: if RB is rosterable, also allow FB and HB
        equivalence_additions = set()
        for nfl_pos, yahoo_pos in position_equivalencies.items():
            if yahoo_pos in expanded_positions:
                equivalence_additions.add(nfl_pos)

        if equivalence_additions:
            expanded_positions.update(equivalence_additions)
            print(f"[filter] Added position equivalencies: {sorted(equivalence_additions)}")

        # Filter NFL data based on UNIFIED position column (not just nfl_position)
        # This catches cases where Yahoo position is valid but NFL position differs
        unified_pos = n_df["position"].astype(str).str.upper()
        # Keep rows where: 1) position is rosterable, OR 2) position is NaN/empty (we'll check later)
        keep_mask = n_df["position"].isna() | unified_pos.isin(expanded_positions)
        nK = nK[keep_mask]

        nfl_after = len(nK)
        filtered_count = nfl_before - nfl_after
        if filtered_count > 0:
            print(f"[filter] Filtered out {filtered_count:,} non-rosterable players based on unified position")
            print(f"[filter] Kept positions: {sorted(expanded_positions)}")
            print(f"[filter] NFL rows after filter: {nfl_after:,}")
    else:
        print(f"[filter] Skipping position filter (no position column or rosterable_positions)")

    l1, yR, nR = layer_merge("1: player+year+week", yK, nK,
                             key_cols=[("player_key","player_key"),("year_key","year_key"),("week_key","week_key")])
    l1b, yR, nR = layer_merge("1b: player+year+week+position", yR, nR,
                              key_cols=[("player_key","player_key"),("year_key","year_key"),("week_key","week_key"),("position_key","position_key")])
    l2, yR, nR = layer_merge("2: last_name+year+week", yR, nR,
                             key_cols=[("player_last_name_key","player_last_name_key"),("year_key","year_key"),("week_key","week_key")])
    l2b, yR, nR = layer_merge_fuzzy_name("2b: FUZZY_NAME middle_names+variations+year+week", yR, nR)
    l3, yR, nR = layer_merge("3: last_name+year+week+position", yR, nR,
                             key_cols=[("player_last_name_key","player_last_name_key"),("year_key","year_key"),("week_key","week_key"),("position_key","position_key")])
    l4, yR, nR = layer_merge("4: last_name+year+week+points", yR, nR,
                             key_cols=[("player_last_name_key","player_last_name_key"),("year_key","year_key"),("week_key","week_key"),("points_key","points_key")])

    # NEW LAYER: Handle edge cases like Taysom Hill (same player, different positions)
    # Match on full name + year + week for offensive players (QB, RB, WR, TE) only
    # This catches cases where Yahoo and NFL disagree on position
    if not yR.empty and not nR.empty:
        # Filter for offensive positions only to avoid false matches
        offensive_pos_y = yR["position_key"].astype(str).str.upper().isin(["QB", "RB", "WR", "TE"])
        offensive_pos_n = nR["position_key"].astype(str).str.upper().isin(["QB", "RB", "WR", "TE"])
        yR_offense = yR[offensive_pos_y].copy()
        nR_offense = nR[offensive_pos_n].copy()

        if not yR_offense.empty and not nR_offense.empty:
            l5, yR_temp, nR_temp = layer_merge("5: EDGE_CASE player+year+week (offense only, ignore position)",
                                              yR_offense, nR_offense,
                                              key_cols=[("player_key","player_key"),("year_key","year_key"),("week_key","week_key")])
            # Update remaining records
            yR = pd.concat([yR[~offensive_pos_y], yR_temp], ignore_index=True)
            nR = pd.concat([nR[~offensive_pos_n], nR_temp], ignore_index=True)
        else:
            l5 = pd.DataFrame()
    else:
        l5 = pd.DataFrame()

    matched_blocks = [assemble_rows(l) for l in (l1, l1b, l2, l2b, l3, l4, l5)]
    final_df = finalize_union(matched_blocks, yR, nR)

    # ===================== DEDUPLICATE DEFENSES AFTER MERGE =====================
    # Remove duplicate defense rows where Yahoo and NFL data both exist but didn't merge
    # This handles cases where merge keys matched but downstream logic created duplicates
    def deduplicate_defenses(df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate defense rows where Yahoo and NFL data both exist but didn't merge.

        Keep Yahoo data (rostered) over NFL data (unrostered) when duplicates exist.
        Generic logic: prefers rows with manager over 'Unrostered'.

        This handles cases like:
        - Same defense (e.g., "New York Jets") appearing twice with identical keys
        - One row rostered (has manager), one row unrostered
        """
        if df.empty or 'position' not in df.columns:
            return df

        # Only deduplicate DEF rows
        def_mask = df['position'].astype(str).str.upper() == 'DEF'

        if not def_mask.any():
            return df

        # Separate DEF and non-DEF
        def_df = df[def_mask].copy()
        non_def_df = df[~def_mask].copy()

        # Find duplicates (same player, year, week)
        def_df['_dup_key'] = (
            def_df['player'].astype(str).str.strip().str.lower() + '|' +
            def_df['year'].astype(str) + '|' +
            def_df['week'].astype(str)
        )

        # Count duplicates before deduplication
        dup_counts = def_df['_dup_key'].value_counts()
        num_duplicates = (dup_counts > 1).sum()
        total_duplicate_rows = (dup_counts[dup_counts > 1]).sum()

        if num_duplicates > 0:
            # Prefer rostered (has manager) over unrostered
            def_df['_is_rostered'] = (def_df['manager'].notna() &
                                       (def_df['manager'].astype(str).str.strip() != 'Unrostered'))

            # Sort by rostered status (True first) and drop duplicates
            def_df = def_df.sort_values('_is_rostered', ascending=False)
            def_df = def_df.drop_duplicates(subset='_dup_key', keep='first')

            # Remove temporary columns
            def_df = def_df.drop(columns=['_dup_key', '_is_rostered'])

            removed = total_duplicate_rows - num_duplicates
            print(f"[deduplicate] Removed {removed} duplicate defense rows (kept rostered over unrostered)")
        else:
            # Remove temporary column
            def_df = def_df.drop(columns=['_dup_key'])

        # Recombine
        result = pd.concat([non_def_df, def_df], ignore_index=True)

        return result

    # Apply deduplication to remove duplicate defenses
    final_df = deduplicate_defenses(final_df)

    # ===================== Points Column Handling (NO CALCULATION) =====================
    # Points calculation has been moved to transformation modules (initial_import_v2.py and player_stats_v2.py)
    # This merge file only preserves Yahoo's original "points" column from the API

    # Preserve Yahoo's original points in a backup column (for rostered players)
    if "points" in final_df.columns and "points_original" not in final_df.columns:
        final_df["points_original"] = final_df["points"].copy()
        print("[points] Preserved Yahoo's original points in 'points_original' column")

    # Ensure "points" column exists (will be NULL for unrostered/NFL-only players)
    if "points" not in final_df.columns:
        final_df["points"] = pd.NA
        print("[points] Initialized 'points' column (will be calculated by transformation modules)")

    # Log statistics
    has_yahoo_points = final_df["points"].notna().sum() if "points" in final_df.columns else 0
    total_rows = len(final_df)
    print(f"[points] Yahoo API provided points for {has_yahoo_points:,}/{total_rows:,} rows")
    print(f"[points] Transformation modules will calculate points for ALL players (including {total_rows - has_yahoo_points:,} unrostered)")

    # Add placeholder columns and new columns
    all_cols = [
        "kept_next_year", "is_keeper_status", "keeper_price", "avg_points_this_year",
        "avg_points_next_year", "avg_cost_next_year", "cost", "faab_bid", "total_points_next_year",
        "rolling_point_total",
        "manager_player_all_time_history", "manager_position_all_time_history", "player_personal_all_time_history",
        "position_all_time_history",
        "manager_player_season_history", "manager_position_season_history", "player_personal_season_history",
        "position_season_history",
        "manager_player_all_time_history_percentile", "manager_position_all_time_history_percentile",
        "player_personal_all_time_history_percentile", "position_all_time_history_percentile",
        "manager_player_season_history_percentile", "manager_position_season_history_percentile",
        "player_personal_season_history_percentile", "position_season_history_percentile",
    ]
    final_df = final_df.reindex(columns=final_df.columns.tolist() + [c for c in all_cols if c not in final_df.columns],
                                fill_value=pd.NA)

    # Create manager_year, player_year, and opponent_year columns (concat with year, no spaces)
    if "manager" in final_df.columns and "year" in final_df.columns:
        mgr = final_df["manager"].astype(str).fillna("").str.replace(" ", "")
        yr = final_df["year"].astype(str).fillna("")
        final_df["manager_year"] = mgr + yr
        final_df["manager_year"] = final_df["manager_year"].replace("", pd.NA)
    else:
        final_df["manager_year"] = pd.NA

    if "player" in final_df.columns and "year" in final_df.columns:
        plyr = final_df["player"].astype(str).fillna("").str.replace(" ", "")
        yr = final_df["year"].astype(str).fillna("")
        final_df["player_year"] = plyr + yr
        final_df["player_year"] = final_df["player_year"].replace("", pd.NA)
    else:
        final_df["player_year"] = pd.NA

    if "opponent" in final_df.columns and "year" in final_df.columns:
        opp = final_df["opponent"].astype(str).fillna("").str.replace(" ", "")
        yr = final_df["year"].astype(str).fillna("")
        final_df["opponent_year"] = opp + yr
    else:
        final_df["opponent_year"] = pd.NA

    final_df = reorder_columns(final_df)

    # ===================== RESTORE ORIGINAL PLAYER NAMES =====================
    # Revert player names to their original form from Yahoo (preferred) or NFL (fallback)
    # This preserves the original naming conventions from the data sources
    if "_original_player_name" in final_df.columns:
        has_original = final_df["_original_player_name"].notna() & (final_df["_original_player_name"].astype(str).str.strip() != "")
        final_df.loc[has_original, "player"] = final_df.loc[has_original, "_original_player_name"]
        # Drop the temporary column
        final_df = final_df.drop(columns=["_original_player_name"], errors="ignore")
        print("[names] Restored original player names (Yahoo preferred, NFL fallback)")

    # ===================== DISAMBIGUATE MULTI-TEAM CITIES =====================
    # For cities with multiple teams (Los Angeles, New York), ensure consistent naming
    # based on nfl_team code. Handle both generic names and already-mapped names.
    # Also standardize team relocations (Oakland->Las Vegas, San Diego->LA, St. Louis->LA).
    if "player" in final_df.columns and "nfl_team" in final_df.columns:
        # Los Angeles: Rams (LAR, LA, STL historical) vs Chargers (LAC, SD historical)
        # Match any LA-related name to standardize naming
        is_la = final_df["player"].astype(str).str.contains("Los Angeles|St. Louis|St Louis|San Diego", case=False, na=False)
        final_df.loc[is_la & (final_df["nfl_team"].isin(["LAR", "LA", "STL"])), "player"] = "Los Angeles Rams"
        final_df.loc[is_la & (final_df["nfl_team"].isin(["LAC", "SD"])), "player"] = "Los Angeles Chargers"

        # New York: Giants (NYG) vs Jets (NYJ)
        # Match any NY-related name
        is_ny = final_df["player"].astype(str).str.contains("New York", case=False, na=False)
        final_df.loc[is_ny & (final_df["nfl_team"] == "NYG"), "player"] = "New York Giants"
        final_df.loc[is_ny & (final_df["nfl_team"] == "NYJ"), "player"] = "New York Jets"

        # Las Vegas Raiders: Standardize Oakland -> Las Vegas (LV, OAK historical)
        # NFLverse uses LV for all years, but handle OAK just in case
        is_raiders = final_df["player"].astype(str).str.contains("Oakland|Las Vegas|Raiders", case=False, na=False)
        final_df.loc[is_raiders & (final_df["nfl_team"].isin(["LV", "OAK"])), "player"] = "Las Vegas"

        # Count standardizations
        la_rams_mask = final_df["nfl_team"].isin(["LAR", "LA", "STL"]) & (final_df["player"] == "Los Angeles Rams")
        la_chargers_mask = final_df["nfl_team"].isin(["LAC", "SD"]) & (final_df["player"] == "Los Angeles Chargers")
        ny_giants_mask = (final_df["nfl_team"] == "NYG") & (final_df["player"] == "New York Giants")
        ny_jets_mask = (final_df["nfl_team"] == "NYJ") & (final_df["player"] == "New York Jets")
        lv_raiders_mask = final_df["nfl_team"].isin(["LV", "OAK"]) & (final_df["player"] == "Las Vegas")

        la_rams = la_rams_mask.sum()
        la_chargers = la_chargers_mask.sum()
        ny_giants = ny_giants_mask.sum()
        ny_jets = ny_jets_mask.sum()
        lv_raiders = lv_raiders_mask.sum()

        total_standardized = la_rams + la_chargers + ny_giants + ny_jets + lv_raiders
        if total_standardized > 0:
            print(f"[disambiguate] Standardized {total_standardized} multi-team/relocated city names:")
            if la_rams > 0:
                print(f"  Los Angeles Rams (LAR/LA/STL): {la_rams}")
            if la_chargers > 0:
                print(f"  Los Angeles Chargers (LAC/SD): {la_chargers}")
            if ny_giants > 0:
                print(f"  New York Giants: {ny_giants}")
            if ny_jets > 0:
                print(f"  New York Jets: {ny_jets}")
            if lv_raiders > 0:
                print(f"  Las Vegas (LV/OAK): {lv_raiders}")

    # Numeric columns rounding/casting
    list_columns = {
        "fg_blocked_list", "fg_made_list", "fg_missed_list", "blocked_fg_list"
    }
    for col in [
        "def_safeties", "def_tackles_for_loss", "def_sacks", "def_interceptions",
        "fum_rec", "def_tds", "special_teams_tds",
        "pts_allow_0", "pts_allow_1_6", "pts_allow_7_13", "pts_allow_14_20",
        "pts_allow_21_27", "pts_allow_28_34", "pts_allow_35_plus",
        "fg_blocked", "fg_blocked_distance",
        "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
        "fg_missed_0_19", "fg_missed_20_29", "fg_missed_30_39", "fg_missed_40_49", "fg_missed_50_59", "fg_missed_60_",
        "fg_made_distance", "fg_missed_distance",
        "gwfg_att", "gwfg_made", "gwfg_missed", "gwfg_blocked", "gwfg_distance",
        "blocked_fg", "blocked_pat", "blocked_gwfg", "blocked_fg_distance"
    ]:
        if col in final_df.columns and col not in list_columns:
            final_df[col] = (
                pd.to_numeric(final_df[col], errors="coerce")
                .fillna(0)
                .round()
                .astype(int)
                .astype("Int64")
            )

    # ===================== FIX DATA TYPES FOR PARQUET =====================
    # Standardize year and week columns to Int64 to prevent parquet write errors
    # (Parquet requires consistent types; concat may produce mixed int/string/Int64)
    if "year" in final_df.columns:
        final_df["year"] = pd.to_numeric(final_df["year"], errors="coerce").astype("Int64")
    if "week" in final_df.columns:
        final_df["week"] = pd.to_numeric(final_df["week"], errors="coerce").astype("Int64")

    # ===================== CREATE COMPOSITE KEYS =====================
    # Create cumulative_week (year * 100 + week) - MUST happen after year/week type conversion
    if "year" in final_df.columns and "week" in final_df.columns:
        final_df["cumulative_week"] = (
            final_df["year"].fillna(-1).astype("Int64") * 100 +
            final_df["week"].fillna(-1).astype("Int64")
        ).astype("Int64")
        print(f"[keys] Created cumulative_week column (year*100 + week)")
    else:
        final_df["cumulative_week"] = pd.NA

    # Create manager_week (manager + cumulative_week)
    if "manager" in final_df.columns and "cumulative_week" in final_df.columns:
        mgr = final_df["manager"].astype(str).fillna("").str.replace(" ", "")
        cw_str = final_df["cumulative_week"].astype("Int64").astype(str)
        final_df["manager_week"] = (mgr + cw_str.fillna("")).astype("string")
        final_df["manager_week"] = final_df["manager_week"].replace("", pd.NA)
        print(f"[keys] Created manager_week column (manager + cumulative_week)")
    else:
        final_df["manager_week"] = pd.NA

    # Create player_week (player + cumulative_week)
    if "player" in final_df.columns and "cumulative_week" in final_df.columns:
        plyr = final_df["player"].astype(str).fillna("").str.replace(" ", "")
        cw_str = final_df["cumulative_week"].astype("Int64").astype(str)
        final_df["player_week"] = (plyr + cw_str.fillna("")).astype("string")
        final_df["player_week"] = final_df["player_week"].replace("", pd.NA)
        print(f"[keys] Created player_week column (player + cumulative_week)")
    else:
        final_df["player_week"] = pd.NA

    csv_path = output_dir / f"yahoo_nfl_merged_{year}_week_{week}.csv"
    pq_path = output_dir / f"yahoo_nfl_merged_{year}_week_{week}.parquet"

    # Use proper naming convention for all-weeks files
    if week == 0:
        csv_path = output_dir / f"yahoo_nfl_merged_{year}_all_weeks.csv"
        pq_path = output_dir / f"yahoo_nfl_merged_{year}_all_weeks.parquet"

    final_df.to_csv(csv_path, index=False)
    try:
        final_df.to_parquet(pq_path, index=False)
    except Exception as e:
        print(f"[warn] Parquet write failed: {e}")
        print(f"[debug] Problematic columns with mixed types:")
        for col in final_df.columns:
            try:
                # Try to detect columns with mixed types
                types = final_df[col].apply(type).unique()
                if len(types) > 1:
                    print(f"  {col}: {types}")
            except Exception:
                pass

    print(f"\n[saved] CSV: {csv_path}")
    print(f"[saved] Parquet: {pq_path}")
    print(f"[saved] Rows: {len(final_df):,}")

if __name__ == "__main__":
    main()
