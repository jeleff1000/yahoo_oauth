"""
Fantasy Points Calculator

Computes fantasy points for players based on league scoring rules.
Handles both offensive stats and DST/DEF scoring.

This module is extracted from yahoo_nfl_merge.py for better modularity.
"""

from typing import List, Dict, Any
import pandas as pd


def compute_points_from_full_scoring(row: pd.Series, rules: List[Dict[str, Any]]) -> float:
    """
    Compute fantasy points for a player based on full scoring rules.

    Args:
        row: DataFrame row with player stats
        rules: List of scoring rules from Yahoo league settings

    Returns:
        Total fantasy points

    Example rules format:
        [
            {"name": "Pass Yds", "points": 0.04},
            {"name": "Pass TD", "points": 4},
            ...
        ]
    """
    total = 0.0

    def two_pt_total(r: pd.Series) -> float:
        """Sum all 2-pt conversion stats."""
        acc = 0.0
        for col in ["2-pt","rushing_2pt_conversions","receiving_2pt_conversions","passing_2pt_conversions"]:
            try:
                acc += float(r.get(col) or 0)
            except Exception:
                pass
        return acc

    for rule in rules:
        pts = rule.get("points")
        if pts is None:
            continue
        try:
            pts_val = float(pts)
        except Exception:
            continue

        name = str(rule.get("name") or "").strip()
        if not name:
            continue

        # Normalize rule name for matching
        key = name.lower().replace(" ", "").replace("-", "_").replace("+", "plus")

        # Skip DEF points allowed (handled separately via DST settings)
        if key in {"pointsallowed","pointsallowedpts","ptsallow"}:
            continue

        # Map rule name to stat column
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
            elif key == "fg50plus":                    val = float(row.get("fg_made_50_59") or 0) + float(row.get("fg_made_60_") or 0)
            elif key == "fgmiss":                      val = float(row.get("fg_miss") or 0)
            # Add more mappings as needed
        except Exception:
            pass

        total += val * pts_val

    return total


def compute_default_points(row: pd.Series) -> float:
    """
    Compute fantasy points using default half-PPR scoring.

    Fallback when league-specific scoring rules unavailable.

    Args:
        row: DataFrame row with player stats

    Returns:
        Total fantasy points (half-PPR)
    """
    try:
        # Prefer half_ppr if available
        if "fantasy_points_half_ppr" in row.index:
            fp = row.get("fantasy_points_half_ppr")
            if pd.notna(fp):
                return float(fp)

        # Fallback: compute half-PPR manually
        total = 0.0

        # Passing
        total += float(row.get("pass_yds") or 0) * 0.04  # 25 yds = 1 pt
        total += float(row.get("pass_td") or 0) * 4
        total += float(row.get("passing_interceptions") or 0) * -2

        # Rushing
        total += float(row.get("rush_yds") or 0) * 0.1  # 10 yds = 1 pt
        total += float(row.get("rush_td") or 0) * 6

        # Receiving
        total += float(row.get("rec") or 0) * 0.5  # Half-PPR
        total += float(row.get("rec_yds") or 0) * 0.1  # 10 yds = 1 pt
        total += float(row.get("rec_td") or 0) * 6

        # Special teams / misc
        total += float(row.get("ret_td") or 0) * 6
        total += float(row.get("fum_ret_td") or 0) * 6
        total += float(row.get("fum_lost") or 0) * -2

        # 2-pt conversions
        for col in ["2-pt","rushing_2pt_conversions","receiving_2pt_conversions","passing_2pt_conversions"]:
            total += float(row.get(col) or 0) * 2

        # Kicking
        total += float(row.get("pat_made") or 0) * 1
        total += float(row.get("pat_missed") or 0) * -1
        total += float(row.get("fg_made_0_19") or 0) * 3
        total += float(row.get("fg_made_20_29") or 0) * 3
        total += float(row.get("fg_made_30_39") or 0) * 3
        total += float(row.get("fg_made_40_49") or 0) * 4
        total += float(row.get("fg_made_50_59") or 0) * 5
        total += float(row.get("fg_made_60_") or 0) * 5
        total += float(row.get("fg_miss") or 0) * -1

        return round(total, 2)

    except Exception:
        return 0.0


def compute_dst_points(row: pd.Series, dst_scoring: Dict[str, float]) -> float:
    """
    Compute DST/DEF fantasy points.

    Args:
        row: DataFrame row with DST stats
        dst_scoring: Dictionary of DST scoring weights from Yahoo settings

    Returns:
        Total DST fantasy points

    Example dst_scoring:
        {
            "Sack": 1.0,
            "Interception": 2.0,
            "Fumble Recovery": 2.0,
            "Touchdown": 6.0,
            "Safety": 2.0,
            "Kickoff and Punt Return Touchdowns": 6.0,
            "PA_0": 10.0,
            "PA_1_6": 7.0,
            ...
        }
    """
    total = 0.0

    try:
        # Defensive stats
        total += float(row.get("def_sacks") or 0) * dst_scoring.get("Sack", 0.0)
        total += float(row.get("def_interceptions") or 0) * dst_scoring.get("Interception", 0.0)
        total += float(row.get("fum_rec") or row.get("fumble_recovery_opp") or 0) * dst_scoring.get("Fumble Recovery", 0.0)
        # Cap def_tds at 3 - nflverse has bad data for some old games (e.g., 2001 PIT vs JAX shows 6 TDs)
        # No NFL defense has ever scored more than 3 TDs in a single game
        def_tds = min(float(row.get("def_tds") or 0), 3.0)
        total += def_tds * dst_scoring.get("Touchdown", 0.0)
        total += float(row.get("def_safeties") or 0) * dst_scoring.get("Safety", 0.0)
        total += float(row.get("special_teams_tds") or row.get("ret_td") or 0) * dst_scoring.get("Kickoff and Punt Return Touchdowns", 0.0)

        # Points Allowed buckets
        pts_allowed = float(row.get("pts_allow") or 0)
        if pts_allowed == 0:
            total += dst_scoring.get("PA_0", 0.0)
        elif 1 <= pts_allowed <= 6:
            total += dst_scoring.get("PA_1_6", 0.0)
        elif 7 <= pts_allowed <= 13:
            total += dst_scoring.get("PA_7_13", 0.0)
        elif 14 <= pts_allowed <= 20:
            total += dst_scoring.get("PA_14_20", 0.0)
        elif 21 <= pts_allowed <= 27:
            total += dst_scoring.get("PA_21_27", 0.0)
        elif 28 <= pts_allowed <= 34:
            total += dst_scoring.get("PA_28_34", 0.0)
        elif pts_allowed >= 35:
            total += dst_scoring.get("PA_35_plus", 0.0)

    except Exception as e:
        print(f"[dst] Error computing DST points: {e}")

    return round(total, 2)
