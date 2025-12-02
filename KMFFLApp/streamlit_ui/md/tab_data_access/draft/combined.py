#!/usr/bin/env python3
"""
Combined data loader for Draft tab.

Main entry point for loading all data needed by the draft tab.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import re
import streamlit as st
from .draft_data import load_draft_data
from md.data_access import run_query, T


@st.cache_data(show_spinner=True, ttl=600)
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


@st.cache_data(show_spinner=False, ttl=600)
def load_roster_config_for_optimizer() -> Optional[Dict[str, Any]]:
    """
    Load roster configuration for the draft optimizer.

    Uses league_settings table (canonical source) to get roster_positions.
    Falls back to inferring from player data if unavailable.

    Returns:
        Dict with position counts (qb, rb, wr, te, flex, def, k, bench) and budget,
        or None if detection fails
    """
    import json

    bench_positions = ['BN', 'IR', 'IL']
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

        if df is not None and not df.empty and 'settings_json' in df.columns:
            settings_str = df.iloc[0]['settings_json']
            if settings_str:
                settings = json.loads(settings_str) if isinstance(settings_str, str) else settings_str
                roster_positions = settings.get('roster_positions', [])

                if roster_positions:
                    position_counts = {}
                    for slot in roster_positions:
                        pos = slot.get('position', '').upper()
                        count = int(slot.get('count', 0))
                        if pos and count > 0:
                            position_counts[pos] = count
                    st.info(f"🔍 Roster from league_settings: {position_counts}")

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
                    pos = str(row['fantasy_position']).upper().strip()
                    count = int(row['slot_count'])
                    position_counts[pos] = count
                st.info(f"🔍 Roster from player data: {position_counts}")

        except Exception as e:
            st.warning(f"Roster config detection failed: {e}")
            return None

    if not position_counts:
        st.warning("⚠️ No roster configuration found")
        return None

    # Count bench slots
    bench_count = sum(position_counts.get(bp, 0) for bp in bench_positions)

    # Count flex positions (anything with / that's not bench)
    flex_count = 0
    for pos, count in position_counts.items():
        if '/' in pos and pos not in bench_positions:
            flex_count += count

    config = {
        "qb": position_counts.get('QB', 0),
        "rb": position_counts.get('RB', 0),
        "wr": position_counts.get('WR', 0),
        "te": position_counts.get('TE', 0),
        "flex": flex_count,
        "def": position_counts.get('DEF', 0),
        "k": position_counts.get('K', 0),
        "bench": bench_count,
        "budget": 200,
        "_position_counts": position_counts,
    }

    # Validate: should have at least QB, RB, or WR (core fantasy positions)
    core_positions = config['qb'] + config['rb'] + config['wr']
    if core_positions == 0:
        st.warning("⚠️ No core positions (QB/RB/WR) detected")
        return None

    # Validate: should have at least 5 starter positions
    total_starters = sum(config[k] for k in ['qb', 'rb', 'wr', 'te', 'flex', 'def', 'k'])
    if total_starters < 5:
        st.warning("⚠️ Too few starter positions detected")
        return None

    return config
