#!/usr/bin/env python3
"""
Combined data loader for Draft tab.

Main entry point for loading all data needed by the draft tab.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Set
import streamlit as st
from .draft_data import load_draft_data
from md.core import run_query, T


# Position code mapping for parsing flex slots
# Yahoo uses single letters: Q=QB, W=WR, R=RB, T=TE, K=K, D=DEF
POSITION_CODE_MAP = {
    "Q": "QB",
    "W": "WR",
    "R": "RB",
    "T": "TE",
    "K": "K",
    "D": "DEF",
}

# Known bench/IR slot names
BENCH_SLOT_NAMES = {"BN", "IR", "IL"}


def parse_flex_eligibility(slot_name: str) -> Set[str]:
    """
    Parse a flex slot name to determine which positions are eligible.

    Examples:
        'W/R/T' -> {'WR', 'RB', 'TE'}
        'Q/W/R/T' -> {'QB', 'WR', 'RB', 'TE'}  (Superflex)
        'W/R' -> {'WR', 'RB'}
        'W/T' -> {'WR', 'TE'}

    Returns:
        Set of eligible position names
    """
    if "/" not in slot_name:
        return set()

    eligible = set()
    # Split by / and map each code
    for code in slot_name.replace("/", ""):
        if code in POSITION_CODE_MAP:
            eligible.add(POSITION_CODE_MAP[code])

    return eligible


def is_flex_slot(slot_name: str) -> bool:
    """Check if a slot name represents a flex position."""
    return "/" in slot_name and slot_name not in BENCH_SLOT_NAMES


def is_bench_slot(slot_name: str) -> bool:
    """Check if a slot name represents a bench/IR position."""
    return slot_name in BENCH_SLOT_NAMES


def is_dedicated_slot(slot_name: str) -> bool:
    """Check if a slot is a dedicated (non-flex, non-bench) position."""
    return not is_flex_slot(slot_name) and not is_bench_slot(slot_name)


@st.cache_data(show_spinner=True, ttl=120)
def load_optimized_draft_data() -> Dict[str, Any]:
    """
    Load all data for draft tab in one optimized call.

    Returns:
        Dict with "Draft History" key containing DataFrame, or "error" key on failure
    """
    try:
        # Load draft data with column selection
        draft_data = load_draft_data()

        # Check for errors
        if "error" in draft_data:
            return {"error": draft_data["error"]}

        # Return the data
        return draft_data

    except Exception as e:
        st.error(f"Failed to load draft tab data: {e}")
        return {"error": str(e)}


@st.cache_data(show_spinner=False, ttl=120)
def load_roster_config_for_optimizer() -> Optional[Dict[str, Any]]:
    """
    Load roster configuration for the draft optimizer.

    Uses league_settings table (canonical source) to get roster_positions.
    Falls back to inferring from player data if unavailable.

    Returns a fully generic config that the optimizer can use dynamically:
    - roster_slots: Dict of all slot names -> counts (e.g., {'QB': 1, 'RB': 2, 'W/R/T': 1})
    - dedicated_slots: Dict of dedicated position slots -> counts (non-flex, non-bench)
    - flex_slots: List of flex slot definitions with eligibility
    - bench_count: Total bench/IR slots
    - flex_eligible_positions: Set of all positions that can fill flex slots
    - all_positions: Set of all position types in the league
    - budget: Default draft budget

    Also includes legacy keys (qb, rb, wr, etc.) for backwards compatibility.
    """
    import json

    position_counts = None

    # Try league_settings table first (canonical source)
    try:
        sql = f"""
            SELECT settings_json
            FROM {T['league_settings']}
            WHERE year = (SELECT MAX(year) FROM {T['league_settings']})
            LIMIT 1
        """
        df = run_query(sql)

        if df is not None and not df.empty and "settings_json" in df.columns:
            settings_str = df.iloc[0]["settings_json"]
            if settings_str:
                settings = (
                    json.loads(settings_str)
                    if isinstance(settings_str, str)
                    else settings_str
                )
                roster_positions = settings.get("roster_positions", [])

                if roster_positions:
                    # Debug: show raw roster_positions
                    st.info(f"🔍 Raw roster_positions from DB: {roster_positions}")
                    position_counts = {}
                    for slot in roster_positions:
                        pos = slot.get("position", "").upper()
                        count = int(slot.get("count", 0))
                        if pos and count > 0:
                            position_counts[pos] = count
                    st.info(f"🔍 Parsed position_counts: {position_counts}")

    except Exception:
        pass  # Fall through to player-based detection

    # Fallback: infer from player data
    if not position_counts:
        try:
            sql = f"""
                SELECT fantasy_position, COUNT(*) as slot_count
                FROM {T['player']}
                WHERE year = (SELECT MAX(year) FROM {T['player']})
                  AND week = 1
                  AND fantasy_position IS NOT NULL
                  AND fantasy_position != ''
                  AND manager = (
                      SELECT manager FROM {T['player']}
                      WHERE year = (SELECT MAX(year) FROM {T['player']})
                        AND week = 1
                        AND manager IS NOT NULL
                        AND manager != ''
                      LIMIT 1
                  )
                GROUP BY fantasy_position
            """
            df = run_query(sql)

            if df is not None and not df.empty:
                position_counts = {}
                for _, row in df.iterrows():
                    pos = str(row["fantasy_position"]).upper().strip()
                    count = int(row["slot_count"])
                    position_counts[pos] = count
                st.info(f"🔍 Roster from player data: {position_counts}")

        except Exception as e:
            st.warning(f"Roster config detection failed: {e}")
            return None

    if not position_counts:
        st.warning("⚠️ No roster configuration found")
        return None

    # === BUILD GENERIC CONFIG ===

    # Separate slots into dedicated, flex, and bench
    dedicated_slots = {}
    flex_slots = []
    bench_count = 0
    flex_eligible_positions = set()
    all_positions = set()

    for slot_name, count in position_counts.items():
        if is_bench_slot(slot_name):
            bench_count += count
        elif is_flex_slot(slot_name):
            eligibility = parse_flex_eligibility(slot_name)
            flex_slots.append(
                {"name": slot_name, "count": count, "eligible_positions": eligibility}
            )
            flex_eligible_positions.update(eligibility)
        else:
            # Dedicated position slot
            dedicated_slots[slot_name] = count
            all_positions.add(slot_name)

    # Add flex-eligible positions to all_positions
    all_positions.update(flex_eligible_positions)

    # Calculate total flex slot count
    total_flex_count = sum(fs["count"] for fs in flex_slots)

    # Calculate total starters (dedicated + flex, excluding bench)
    total_starters = sum(dedicated_slots.values()) + total_flex_count

    # === BUILD CONFIG ===
    config = {
        # Generic structure (use these for dynamic optimizer)
        "roster_slots": position_counts,  # Raw slot counts
        "dedicated_slots": dedicated_slots,  # Non-flex, non-bench slots
        "flex_slots": flex_slots,  # List of flex slot definitions
        "bench_count": bench_count,
        "total_flex_count": total_flex_count,
        "flex_eligible_positions": flex_eligible_positions,
        "all_positions": all_positions,
        "total_starters": total_starters,
        "budget": 200,
        # Legacy keys for backwards compatibility
        "qb": dedicated_slots.get("QB", 0),
        "rb": dedicated_slots.get("RB", 0),
        "wr": dedicated_slots.get("WR", 0),
        "te": dedicated_slots.get("TE", 0),
        "flex": total_flex_count,
        "def": dedicated_slots.get("DEF", 0),
        "k": dedicated_slots.get("K", 0),
        "bench": bench_count,
        # Debug info
        "_position_counts": position_counts,
    }

    # Validate: should have at least some positions
    if not all_positions:
        st.warning("⚠️ No positions detected")
        return None

    # Validate: should have at least 5 starter positions
    if total_starters < 5:
        st.warning("⚠️ Too few starter positions detected")
        return None

    return config
