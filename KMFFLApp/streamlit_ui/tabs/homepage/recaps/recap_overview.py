#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import re

import pandas as pd
import streamlit as st

from .displays import weekly_recap, season_recap, player_recap
from md.tab_data_access.homepage.recaps_player_data import load_player_two_week_slice  # ✅ Optimized: 18 cols vs 270+

# ----- helpers -----
def _as_dataframe(obj: Any) -> Optional[pd.DataFrame]:
    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
    except Exception:
        return None
    return None

def _get_matchup_df(df_dict: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not isinstance(df_dict, dict):
        return None
    if "Matchup Data" in df_dict:
        return _as_dataframe(df_dict["Matchup Data"])
    for k, v in df_dict.items():
        if str(k).strip().lower() == "matchup data":
            return _as_dataframe(v)
    return None

def _unique_numeric(df: pd.DataFrame, col: str) -> List[int]:
    if col not in df.columns:
        return []
    ser = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    return sorted(ser.unique().tolist())

def _weeks_for_year(df: pd.DataFrame, year: int) -> List[int]:
    if not {"year", "week"}.issubset(set(df.columns)):
        return []
    df_yr = df[pd.to_numeric(df["year"], errors="coerce").astype("Int64") == year]
    if df_yr.empty:
        return []
    return _unique_numeric(df_yr, "week")

def _find_manager_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ["manager","manager_name","owner","owner_name","team_owner","team_manager"]
    lower = {str(c).lower(): c for c in df.columns}
    for p in preferred:
        if p in lower:
            return lower[p]
    for k,v in lower.items():
        if "manager" in k or "owner" in k:
            return v
    return None

def _manager_options(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None:
        return []
    col = _find_manager_column(df)
    if not col:
        return []
    try:
        ser = df[col].astype(str).str.strip()
    except Exception:
        return []
    ser = ser[(ser.notna()) & (ser != "")]
    return sorted(set(ser.tolist()), key=lambda x: x.lower())

# ----- main -----
@st.fragment
def display_recap_overview(df_dict: Optional[Dict[str, Any]] = None, key_prefix: str = "") -> None:
    matchup_df = _get_matchup_df(df_dict)
    if matchup_df is None or matchup_df.empty:
        st.info("No matchup data available.")
        return

    mode = st.radio("", options=["Start from Today's Date", "Choose a Date"], horizontal=True, key=f"{key_prefix}recap_overview_date_mode")
    if mode == "Start from Today's Date":
        years = _unique_numeric(matchup_df, "year")
        selected_year = max(years) if years else datetime.now().year
        weeks = _weeks_for_year(matchup_df, selected_year)
        selected_week = max(weeks) if weeks else 1
        st.caption(f"Selected Year: {selected_year} — Week: {selected_week}")
    else:
        years = _unique_numeric(matchup_df, "year") or [datetime.now().year]
        col_year, col_week = st.columns(2)
        with col_year:
            selected_year = st.selectbox("Year", options=years, index=len(years) - 1, key=f"{key_prefix}recap_overview_year")
        weeks = _weeks_for_year(matchup_df, selected_year) or _unique_numeric(matchup_df, "week") or list(range(1,19))
        with col_week:
            selected_week = st.selectbox("Week", options=weeks, index=len(weeks) - 1, key=f"{key_prefix}recap_overview_week")
        st.caption(f"Selected Year: {selected_year} — Week: {selected_week}")

    st.subheader("Manager")
    managers = _manager_options(matchup_df)
    if not managers:
        st.info("No managers found in the dataset.")
        return
    selected_manager = st.selectbox("Select Manager", options=managers, index=0, key=f"{key_prefix}recap_overview_manager")

    # ---------- Recap sections that depend only on Matchup Data ----------
    st.divider()
    st.header("Weekly Recap")
    weekly_recap.display_weekly_recap(
        df_dict=df_dict,
        year=selected_year,
        week=selected_week,
        manager=selected_manager,
    )

    st.divider()
    st.header("Season Analysis")
    season_recap.display_season_recap(
        df_dict=df_dict,
        year=selected_year,
        week=selected_week,
        manager=selected_manager,
    )

    # ---------- Player Weekly Recap (now fetches the 2-week slice AFTER selection) ----------
    try:
        player_two_week = load_player_two_week_slice(int(selected_year), int(selected_week))
    except Exception as e:
        player_two_week = None
        st.warning(f"Two-week player slice unavailable: {e}")

    st.divider()
    st.header("Player Weekly Recap")
    if player_two_week is None or player_two_week.empty:
        st.info("No player data for the selected week.")
        return

    df_dict_player = dict(df_dict or {})
    df_dict_player["Player Data"] = player_two_week  # pass only the tiny slice
    player_recap.display_player_weekly_recap(
        df_dict=df_dict_player,
        year=int(selected_year),
        week=int(selected_week),
        manager=selected_manager,
    )
