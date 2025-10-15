#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import math
import argparse
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
# Paths (all relative to this file's location)
# --------------------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent  # .../fantasy_football_data_scripts/player_stats
REPO_ROOT = SCRIPT_DIR.parent.parent  # .../fantasy_football_data_downloads
OUTPUT_DIR = REPO_ROOT / "fantasy_football_data" / "player_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_DIR = OUTPUT_DIR / "yahoo_league_settings"
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

OAUTH_PATH = REPO_ROOT / "oauth" / "Oauth.json"

YAHOO_SCRIPT = SCRIPT_DIR / "yahoo_fantasy_data.py"
NFL_COMBINE_SCRIPT = SCRIPT_DIR / "combine_dst_to_nfl.py"

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

def fetch_yahoo_dst_scoring(year: int, league_key_arg: Optional[str]) -> Optional[Dict[str, float]]:
    if OAuth2 is None or not OAUTH_PATH.exists():
        return None
    oauth = OAuth2(None, None, from_file=str(OAUTH_PATH))
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
    out = SETTINGS_DIR / f"yahoo_dst_scoring_{year}_{safe}.json"
    try:
        out.write_text(json.dumps({"year": year, "league_key": league_key, "dst_scoring": scoring}, indent=2), encoding="utf-8")
        print(f"[dst] Saved DST settings -> {out.name}")
    except Exception as e:
        print(f"[dst] Could not save DST settings: {e}")

    return scoring

def load_saved_dst_scoring(year: int, league_key_arg: Optional[str]) -> Optional[Dict[str, float]]:
    candidates: List[Path] = []
    if league_key_arg:
        safe = league_key_arg.replace(".", "_")
        p = SETTINGS_DIR / f"yahoo_dst_scoring_{year}_{safe}.json"
        if p.exists():
            candidates.append(p)
    if not candidates:
        candidates = sorted(SETTINGS_DIR.glob(f"yahoo_dst_scoring_{year}_*.json"),
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

def load_saved_full_scoring(year: int, league_key_arg: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    candidates: List[Path] = []
    if league_key_arg:
        safe = league_key_arg.replace(".", "_")
        p = SETTINGS_DIR / f"yahoo_full_scoring_{year}_{safe}.json"
        if p.exists(): candidates.append(p)
    if not candidates:
        candidates = sorted(SETTINGS_DIR.glob(f"yahoo_full_scoring_{year}_*.json"),
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
# Points calculators
# --------------------------------------------------------------------------------------
def compute_points_from_full_scoring(row: pd.Series, rules: List[Dict[str, Any]]) -> float:
    total = 0.0

    def two_pt_total(r: pd.Series) -> float:
        acc = 0.0
        for col in ["2-pt","rushing_2pt_conversions","receiving_2pt_conversions","passing_2pt_conversions"]:
            try: acc += float(r.get(col) or 0)
            except Exception: pass
        return acc

    for rule in rules:
        pts = rule.get("points")
        if pts is None: continue
        try: pts_val = float(pts)
        except Exception: continue

        name = str(rule.get("name") or "").strip()
        if not name: continue

        key = name.lower().replace(" ", "").replace("-", "_").replace("+", "plus")
        if key in {"pointsallowed","pointsallowedpts","ptsallow"}:
            continue  # DEF buckets handled via DST settings

        val = 0.0
        try:
            if key == "passyds":                       val = float(row.get("pass_yds") or 0)
            elif key in {"passtd","passtdd"}:          val = float(row.get("pass_td") or 0)
            elif key == "int":                         val = float(row.get("passing_interceptions") or 0)
            elif key == "rushyds":                     val = float(row.get("rush_yds") or 0)
            elif key == "rushtd":                      val = float(row.get("rush_td") or 0)
            elif key in {"rec","receptions"}:          val = float(row.get("rec") or 0)
            elif key in {"recyds","receivingyds"}:     val = float(row.get("rec_yds") or 0)
            elif key in {"rectd","receivingtd"}:       val = float(row.get("rec_td") or 0)
            elif key in {"retd","returntd","rettd"}:   val = float(row.get("ret_td") or 0)
            elif key in {"2-pt","two_pt","2pt"}:       val = two_pt_total(row)
            elif key in {"fumlost","fumbleslost"}:     val = float(row.get("fum_lost") or 0)
            elif key in {"fumretd","fumrettd"}:        val = float(row.get("fum_ret_td") or 0)
            elif key in {"patmade","pat"}:             val = float(row.get("pat_made") or 0)
            elif key in {"patmiss","patmissed"}:       val = float(row.get("pat_missed") or row.get("pat_miss") or 0)
            elif key in {"fgyds","fgyards"}:           val = float(row.get("fg_yds") or 0)
            elif key == "fg0_19":                      val = float(row.get("fg_made_0_19") or 0)
            elif key == "fg20_29":                     val = float(row.get("fg_made_20_29") or 0)
            elif key == "fg30_39":                     val = float(row.get("fg_made_30_39") or 0)
            elif key == "fg40_49":                     val = float(row.get("fg_made_40_49") or 0)
            elif key in {"fg50+","fg50_plus","fg50"}:  val = float(row.get("fg_made_50_59") or 0) + float(row.get("fg_made_60_", 0))
            elif key == "fgm0_19":                     val = float(row.get("fg_missed_0_19") or 0)
            elif key == "fgm20_29":                    val = float(row.get("fg_missed_20_29") or 0)
            elif key == "fgm30_39":                    val = float(row.get("fg_missed_30_39") or 0)
            elif key == "fgm40_49":                    val = float(row.get("fg_missed_40_49") or 0)
            elif key in {"fgm50+","fgm50_plus","fgm50"}: val = float(row.get("fg_missed_50_59") or 0) + float(row.get("fg_missed_60_", 0))
            elif key == "sack":                        val = float(row.get("def_sacks") or row.get("def_sack") or 0)
            elif key == "interception":                 val = float(row.get("def_interceptions") or row.get("int") or 0)
            elif key in {"fumblerecovery","fumrec"}:   val = float(row.get("fum_rec") or 0)
            elif key in {"touchdown","td"}:            val = float(row.get("def_tds") or row.get("td") or 0)
            elif key in {"safety","safe"}:             val = float(row.get("def_safeties") or 0)
            elif key in {"kickoffandpuntreturntouchdowns","kickandpuntrettd"}:
                                                       val = float(row.get("special_teams_tds") or 0)
            elif key.startswith("ydsallow"):           # if league rules included yards allowed
                suffix = key.replace("ydsallow", "").replace("yardsallowed", "")
                suffix = suffix.replace("plus", "_plus").replace("neg", "_neg")
                col = f"yds_allow_{suffix}" if suffix else "yds_allow"
                val = float(row.get(col) or 0)
            elif key in {"blkick","blkkick","blockedkick"}:
                val = float(row.get("blk_kick") or 0)
            elif key in {"3andouts","threeandouts","3outs"}:
                val = float(row.get("3_and_outs") or 0)
            elif key in {"4dwnstops","4thdownstops","4dwstops"}:
                val = float(row.get("4_dwn_stops") or 0)
            elif key in {"tfl","tacklesforloss"}:
                val = float(row.get("def_tackles_for_loss") or 0)
            elif key == "xpr":
                val = float(row.get("xpr") or 0)
            else:
                continue
        except Exception:
            val = 0.0
        total += val * pts_val

    return round(total, 2)

def compute_default_points(row: pd.Series) -> float:
    """Yahoo-ish half-PPR fallback for offense/kickers. DEF scored elsewhere."""
    try:
        pos = str(row.get("nfl_position") or row.get("yahoo_position") or row.get("fantasy_position") or "").upper()
    except Exception:
        pos = ""

    if pos == "DEF":
        return float(row.get("points_dst_from_yahoo_settings") or 0.0)

    if pos == "K":
        def v(c):
            try: return float(row.get(c) or 0)
            except Exception: return 0.0
        total = (
            v("fg_made_0_19") * 3.0 +
            v("fg_made_20_29") * 3.0 +
            v("fg_made_30_39") * 3.0 +
            v("fg_made_40_49") * 4.0 +
            (v("fg_made_50_59") + v("fg_made_60_")) * 5.0 +
            v("pat_made") * 1.0
        )
        return round(total, 2)

    def n(c):
        try: return float(row.get(c) or 0)
        except Exception: return 0.0

    total = 0.0
    total += n("pass_yds") / 25.0 + n("pass_td") * 4.0 - n("passing_interceptions") * 1.0
    total += n("rush_yds") / 10.0 + n("rush_td") * 6.0
    total += n("rec") * 0.5 + n("rec_yds") / 10.0 + n("rec_td") * 6.0
    total += n("ret_td") * 6.0 + n("fum_ret_td") * 6.0 + n("2-pt") * 2.0 - n("fum_lost") * 2.0
    return round(total, 2)

# --------------------------------------------------------------------------------------
# Helpers to run the other scripts & locate outputs
# --------------------------------------------------------------------------------------
def run_script(pyfile: Path, year: int, week: int) -> None:
    if not pyfile.exists():
        raise FileNotFoundError(f"Script not found: {pyfile}")
    cmd = [sys.executable, str(pyfile), "--year", str(year), "--week", str(week)]
    print(f"\n[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def expected_yahoo_parquet(year: int, week: int) -> List[Path]:
    if year == 0 and week == 0:
        return [OUTPUT_DIR / "yahoo_player_stats_multi_year_all_weeks.parquet"]
    if year == 0 and week > 0:
        return [OUTPUT_DIR / f"yahoo_player_stats_multi_year_week_{week}.parquet"]
    if year > 0 and week == 0:
        return [OUTPUT_DIR / f"yahoo_player_stats_{year}_all_weeks.parquet"]
    return [OUTPUT_DIR / f"yahoo_player_stats_{year}_week_{week}.parquet"]

def locate_parquet_by_signature(modified_after_ts: float, include_cols_any: List[str],
                                year: int, week: int, prefer_most_rows: bool = True) -> Optional[Path]:
    candidates: List[Tuple[Path, int]] = []
    for p in OUTPUT_DIR.glob("*.parquet"):
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

def load_yahoo(year: int, week: int, started_ts: float) -> pd.DataFrame:
    for p in expected_yahoo_parquet(year, week):
        if p.exists():
            print(f"[yahoo] reading {p}")
            return pd.read_parquet(p)
    sig = ["yahoo_player_id", "manager", "yahoo_position", "fantasy_position", "points"]
    found = locate_parquet_by_signature(started_ts, sig, year, week)
    if not found:
        raise FileNotFoundError("Could not locate Yahoo parquet output.")
    print(f"[yahoo] reading {found}")
    return pd.read_parquet(found)

def load_nfl(year: int, week: int, started_ts: float) -> pd.DataFrame:
    sig = ["fantasy_points_half_ppr", "nfl_team"]
    found = locate_parquet_by_signature(started_ts, sig, year, week)
    if not found:
        raise FileNotFoundError("Could not locate NFL (combined) parquet output.")
    print(f"[nfl] reading {found}")
    return pd.read_parquet(found)

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

# --- key helpers ----------------------------------------------------------------------
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
def _norm_player_for_key(x: str) -> str:
    s = apply_manual_mapping(clean_name(str(x or ""))).strip()
    return s.lower()

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
    pts_src = out.get("points") if origin == "yahoo" else out.get("fantasy_points_half_ppr")
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

    # --------- IMPORTANT: Yahoo players with points MUST match NFL data ---------
    # Yahoo records ALL rostered players (including DNP/injured/bye)
    # NFL only records players who had stats
    # So: Yahoo player WITH POINTS should ALWAYS have NFL data
    # Exception: Allow looser matching ONLY if Yahoo points > 0 (player actually played)
    if not merged.empty:
        points_y = _num_series(merged, "points_Y")
        has_points = points_y > 0  # Player actually played

        # For players WITH points, allow the match (they played, should be in NFL data)
        # For players WITHOUT points (0 or NA), require stricter name+position match
        # This prevents false matches for DNP/injured players

        def _norm_name(s):
            return s.astype(str).str.strip().str.casefold()

        name_match = _norm_name(merged.get("player_Y", pd.Series("", index=merged.index))) == \
                     _norm_name(merged.get("player_N", pd.Series("", index=merged.index)))

        pos_match = merged.get("position_key_Y", pd.Series("", index=merged.index)).astype(str) == \
                    merged.get("position_key_N", pd.Series("", index=merged.index)).astype(str)

        # Keep match if: player has points (actually played) OR (no points but name+position both match exactly)
        keep_mask = has_points | (name_match & pos_match)
        merged = merged[keep_mask]
    # ------------------------------------------------------------------------------------

    if verbose:
        y_count, n_count = len(left), len(right)
        matched_y = merged["_y_rid"].nunique() if not merged.empty else 0
        matched_n = merged["_n_rid"].nunique() if not merged.empty else 0
        print(f"\n[{layer_name}] candidates={y_count} x {n_count}  matched={matched_y} y_unmatched≈{(y_count-matched_y)} n_unmatched≈{(n_count-matched_n)}")
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
        "player","player_last_name","nfl_position","fantasy_position","yahoo_position",
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

def compute_dst_points_from_settings(df: pd.DataFrame, scoring: Dict[str, float]) -> pd.Series:
    s = pd.Series(0.0, index=df.index)
    num = lambda c: pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)
    s += num("def_sacks")              * scoring.get("Sack", 0.0)
    s += num("def_interceptions")      * scoring.get("Interception", 0.0)
    s += num("fum_rec")                * scoring.get("Fumble Recovery", 0.0)
    s += num("def_tds")                * scoring.get("Touchdown", 0.0)
    s += num("special_teams_tds")      * scoring.get("Kickoff and Punt Return Touchdowns", 0.0)
    s += num("def_safeties")           * scoring.get("Safety", 0.0)
    for col, key in {
        "pts_allow_0":"PA_0","pts_allow_1_6":"PA_1_6","pts_allow_7_13":"PA_7_13",
        "pts_allow_14_20":"PA_14_20","pts_allow_21_27":"PA_21_27","pts_allow_28_34":"PA_28_34","pts_allow_35_plus":"PA_35_plus"
    }.items():
        s += num(col) * scoring.get(key, 0.0)
    return s.round(2)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Merge Yahoo and NFL player stats")
    parser.add_argument('--year', type=int, default=None, help='Season year (0 for all years)')
    parser.add_argument('--week', type=int, default=None, help='Week number (0 for all weeks)')
    parser.add_argument('--league-key', type=str, default=None, help='Optional league key')
    args = parser.parse_args()

    # Use CLI args if provided, otherwise prompt
    if args.year is not None:
        year = args.year
        league_key_arg = args.league_key
        # Ask for week only if year != 0 and not provided via CLI
        if year != 0 and args.week is not None:
            week = args.week
        elif year != 0 and args.week is None:
            try:
                week = int(input("Enter week number (0 = all weeks): ").strip())
            except Exception:
                week = 0
        else:
            week = 0
    else:
        raw = input("Enter: season year (0 = start from current year and go backward) [optional league_key]:\n").strip()
        parts = raw.split()
        try:
            year = int(parts[0]) if parts else 0
        except Exception:
            year = 0

        # Ask for week only if year != 0
        if year != 0:
            try:
                week = int(input("Enter week number (0 = all weeks): ").strip())
            except Exception:
                week = 0
        else:
            week = 0

        league_key_arg = parts[1] if len(parts) > 1 else None

    t0 = time.time(); run_script(YAHOO_SCRIPT, year, week)
    t1 = time.time(); run_script(NFL_COMBINE_SCRIPT, year, week)
    t2 = time.time()

    y_df = load_yahoo(year, week, started_ts=t0)
    n_df = load_nfl(year, week, started_ts=t1)

    if "year" in y_df.columns and year != 0:
        y_df = y_df[pd.to_numeric(y_df["year"], errors="coerce") == year]
    if "week" in y_df.columns and week != 0:
        y_df = y_df[pd.to_numeric(y_df["week"], errors="coerce") == week]
    if "year" in n_df.columns and year != 0:
        n_df = n_df[pd.to_numeric(n_df["year"], errors="coerce") == year]
    if "week" in n_df.columns and week != 0:
        n_df = n_df[pd.to_numeric(n_df["week"], errors="coerce") == week]

    print(f"\n[yahoo rows] {len(y_df):,}")
    print(f"[nfl rows]   {len(n_df):,}")

    yK = with_keys(y_df, origin="yahoo")
    nK = with_keys(n_df, origin="nfl")

    l1, yR, nR = layer_merge("1: player+year+week", yK, nK,
                             key_cols=[("player_key","player_key"),("year_key","year_key"),("week_key","week_key")])
    l1b, yR, nR = layer_merge("1b: player+year+week+position", yR, nR,
                              key_cols=[("player_key","player_key"),("year_key","year_key"),("week_key","week_key"),("position_key","position_key")])
    l2, yR, nR = layer_merge("2: last_name+year+week", yR, nR,
                             key_cols=[("player_last_name_key","player_last_name_key"),("year_key","year_key"),("week_key","week_key")])
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

    matched_blocks = [assemble_rows(l) for l in (l1, l1b, l2, l3, l4, l5)]
    final_df = finalize_union(matched_blocks, yR, nR)

    # ===================== DST scoring (DEF only) =====================
    scoring_year = year if year != 0 else pd.to_numeric(final_df.get("year"), errors="coerce").max()
    scoring = load_saved_dst_scoring(int(scoring_year) if pd.notna(scoring_year) else 0, league_key_arg) \
              or fetch_yahoo_dst_scoring(int(scoring_year) if pd.notna(scoring_year) else 0, league_key_arg)

    if scoring and isinstance(scoring, dict):
        if all(abs(scoring.get(k, 0)) < 1e-9 for k in scoring):
            scoring = {
                "Sack":1.0,"Interception":2.0,"Fumble Recovery":2.0,"Touchdown":6.0,"Safety":2.0,
                "Kickoff and Punt Return Touchdowns":6.0,
                "PA_0":10.0,"PA_1_6":7.0,"PA_7_13":4.0,"PA_14_20":1.0,"PA_21_27":0.0,"PA_28_34":-1.0,"PA_35_plus":-4.0
            }
            print("[dst] Fallback to default DST scoring rules.")

        if "points_original" not in final_df.columns:
            final_df["points_original"] = final_df.get("points")

        # ensure needed DEF columns exist
        for c in ["def_sacks","def_interceptions","fum_rec","def_tds","special_teams_tds","def_safeties",
                  "pts_allow_0","pts_allow_1_6","pts_allow_7_13","pts_allow_14_20","pts_allow_21_27","pts_allow_28_34","pts_allow_35_plus"]:
            if c not in final_df.columns:
                final_df[c] = 0

        nfl_pos = final_df.get("nfl_position", "")
        yah_pos = final_df.get("yahoo_position", "")
        mask_def = nfl_pos.astype(str).str.upper().eq("DEF") | yah_pos.astype(str).str.upper().eq("DEF")

        dst_points = compute_dst_points_from_settings(final_df[mask_def], scoring)
        final_df.loc[mask_def, "points_dst_from_yahoo_settings"] = dst_points

        if "points" not in final_df.columns:
            final_df["points"] = pd.NA
        blank_points = final_df["points"].isna() | (final_df["points"].astype(str).str.strip() == "")
        final_df.loc[mask_def & blank_points, "points"] = final_df.loc[mask_def & blank_points, "points_dst_from_yahoo_settings"]
    else:
        print("[dst] No DST settings available.")

    # ===================== Offense/K/Kicker scoring =====================
    # Ensure "points" column exists
    final_df["points"] = final_df.get("points", pd.NA)

    # Identify missing or invalid points
    points_num = pd.to_numeric(final_df["points"], errors="coerce")
    looks_empty = final_df["points"].astype(str).str.strip().str.lower().isin(["", "na", "nan", "none", "<na>"])
    missing_points = points_num.isna() | looks_empty

    # Detect offensive activity
    offense_cols = ["pass_yds", "pass_td", "rush_yds", "rush_td", "rec", "rec_yds", "rec_td", "ret_td", "fum_lost"]
    offense_activity = sum(
        pd.to_numeric(final_df.get(col, 0), errors="coerce").fillna(0).abs() for col in offense_cols) > 0
    zero_points = points_num.fillna(0).eq(0)
    needs_points = missing_points | (zero_points & offense_activity)

    # Compute points
    try:
        ry = int(scoring_year) if pd.notna(scoring_year) else None
        rules = load_saved_full_scoring(ry, league_key_arg) if ry is not None else None
    except Exception:
        rules = None

    idx = needs_points[needs_points].index
    if rules:
        final_df.loc[idx, "points"] = final_df.loc[idx].apply(lambda r: compute_points_from_full_scoring(r, rules),
                                                              axis=1)
    else:
        final_df.loc[idx, "points"] = final_df.loc[idx].apply(compute_default_points, axis=1)

    # Fill from half-PPR if still missing
    final_df["points"] = pd.to_numeric(final_df["points"], errors="coerce")
    if "fantasy_points_half_ppr" in final_df.columns:
        fp = pd.to_numeric(final_df["fantasy_points_half_ppr"], errors="coerce")
        need = final_df["points"].isna() & fp.notna()
        final_df.loc[need, "points"] = fp.round(2)

    final_df["points"] = final_df["points"].fillna(0.0).astype(float).round(2)

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

    csv_path = OUTPUT_DIR / f"yahoo_nfl_merged_{year}_week_{week}.csv"
    pq_path = OUTPUT_DIR / f"yahoo_nfl_merged_{year}_week_{week}.parquet"

    final_df.to_csv(csv_path, index=False)
    try:
        final_df.to_parquet(pq_path, index=False)
    except Exception as e:
        print(f"[warn] Parquet write failed: {e}")

if __name__ == "__main__":
    main()
