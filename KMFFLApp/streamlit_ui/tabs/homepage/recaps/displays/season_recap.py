from typing import Any, Dict, Optional
import re

import pandas as pd
import streamlit as st

# Import contextual helpers module to avoid circular imports
from ..helpers import contextual_helpers as ctx


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


def _get_matchup_df(df_dict: Optional[Dict[Any, Any]]) -> Optional[pd.DataFrame]:
    if not isinstance(df_dict, dict):
        return None
    if "Matchup Data" in df_dict:
        return _as_dataframe(df_dict["Matchup Data"])
    for k, v in df_dict.items():
        if str(k).strip().lower() == "matchup data":
            return _as_dataframe(v)
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower().strip())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _find_manager_column(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "manager", "manager_name", "owner", "owner_name",
        "team_owner", "team_manager",
    ]
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for p in preferred:
        if p in cols_lower:
            return cols_lower[p]
    for lower, original in cols_lower.items():
        if "manager" in lower or "owner" in lower:
            return original
    return None


def _val(row: pd.Series, col: Optional[str], default=None):
    if not col:
        return default
    try:
        v = row[col]
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def _to_int(v, default=None) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return int(float(v))
    except Exception:
        return default


def _to_float(v, default=None) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    except Exception:
        return default


def _fmt_number(v) -> str:
    f = _to_float(v, None)
    if f is None:
        return "N/A"
    return f"{f:.2f}".rstrip("0").rstrip(".")


def _fmt_percent(v) -> str:
    f = _to_float(v, None)
    if f is None:
        return "N/A"
    pct = f * 100.0 if 0.0 <= f <= 1.0 else f
    return f"{pct:.1f}%".replace(".0%", "%")


# --- Bolding helpers (parentheticals and number+unit phrases) ---
_UNIT_WORDS = (
    "point", "points", "win", "wins", "seed", "seeds",
    "chance", "favorite", "favorites", "spread", "margin",
    "victory", "loss", "losses", "team", "teams",
    "percent", "percentage", "game", "games",
)
_ADJ_WORDS = ("extra", "total", "more", "fewer", "additional")

# number (with optional %, decimals, commas) + optional adjective + required unit
_RE_NUM_UNIT = re.compile(
    rf"(?<!\w)("
    rf"(?:\d{{1,3}}(?:,\d{{3}})*|\d+)(?:\.\d+)?%?"
    rf"(?:\s+(?:{'|'.join(_ADJ_WORDS)}))?"
    rf"(?:\s+(?:{'|'.join(_UNIT_WORDS)}))"
    rf")(?!\w)",
    re.IGNORECASE
)

def _escape_md_text(s: str) -> str:
    # Escape Markdown special chars so content inside bold stays literal
    return re.sub(r"([\\`*_~\-])", r"\\\1", str(s or ""))

def _bold_parentheticals(s: str) -> str:
    # Make any (...) segment bold, including the parentheses
    def repl(m: re.Match) -> str:
        inner = _escape_md_text(m.group(1))
        return f"**{inner}**"
    return re.sub(r"\(([^()]*)\)", repl, str(s or ""))

def _bold_numbers_with_units(s: str) -> str:
    # Bold number+unit phrases outside parentheses (avoid nested bold)
    def bold_nums_segment(seg: str) -> str:
        def repl(m: re.Match) -> str:
            return f"**{_escape_md_text(m.group(1))}**"
        return _RE_NUM_UNIT.sub(repl, seg)

    text = str(s or "")
    out = []
    depth = 0
    buf = []
    for ch in text:
        if ch == "(":
            if depth == 0 and buf:
                out.append(bold_nums_segment("".join(buf)))
                buf = []
            depth += 1
            out.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            out.append(ch)
        else:
            if depth == 0:
                buf.append(ch)
            else:
                out.append(ch)
    if buf:
        out.append(bold_nums_segment("".join(buf)))
    return "".join(out)

def _apply_bolding(s: str) -> str:
    # Bold number+units outside parentheses, then bold entire parentheticals
    return _bold_parentheticals(_bold_numbers_with_units(s))


@st.fragment
def display_season_recap(
    df_dict: Optional[Dict[Any, Any]] = None,
    *,
    year: int,
    week: int,
    manager: str,
) -> None:
    """
    Text-only season recap. Assumes selection is handled by the caller.
    """
    matchup_df = _get_matchup_df(df_dict)
    if matchup_df is None:
        st.info("No `Matchup Data` dataset available.")
        return

    df = matchup_df.copy()

    col_year = _find_col(df, ["year"])
    col_week = _find_col(df, ["week"])
    col_manager = _find_manager_column(df)

    col_wins_to_date = _find_col(df, ["Wins to Date", "wins_to_date", "wins to date"])
    col_losses_to_date = _find_col(df, ["Losses to Date", "losses_to_date", "losses to date"])
    col_seed_to_date = _find_col(df, ["Playoff Seed to Date", "playoff_seed_to_date", "seed to date"])
    col_avg_seed = _find_col(df, ["avg_seed", "average_seed"])
    col_p_playoffs = _find_col(df, ["p_playoffs", "prob_playoffs", "p playoffs"])
    col_p_champ = _find_col(df, ["p_champ", "prob_championship", "p champ"])

    # Shuffled schedule metrics
    col_shuffle_avg_wins = _find_col(df, [
        "shuffle_avg_wins", "shuffle avg wins", "avg_shuffle_wins", "avg shuffle wins",
        "shuffled_avg_wins", "shuffled avg wins", "simulated_avg_wins", "simulated avg wins",
        "shuffle wins", "expected wins shuffled", "sched_adj_avg_wins", "schedule adjusted avg wins"
    ])
    col_shuffle_avg_playoffs = _find_col(df, [
        "shuffle_avg_playoffs", "shuffle avg playoffs", "avg_shuffle_playoffs", "avg shuffle playoffs",
        "shuffled_avg_playoffs", "shuffled avg playoffs", "simulated_playoff_pct", "sim playoff pct",
        "shuffle_playoff_pct", "playoff pct shuffled", "p_playoffs_shuffle", "p playoffs shuffle",
        "playoff_odds_shuffle", "playoff odds shuffle"
    ])
    # Schedule luck vs shuffled wins
    col_wins_vs_shuffle = _find_col(df, [
        "wins_vs_shuffle_wins", "wins vs shuffle wins", "wins_minus_shuffle_avg_wins",
        "wins minus shuffle avg wins", "wins_minus_expected", "wins minus expected",
        "wins_above_shuffle", "wins above shuffle", "schedule_luck_wins", "schedule luck wins"
    ])

    if col_year:
        df = df[pd.to_numeric(df[col_year], errors="coerce").astype("Int64") == year]
    if col_week:
        df = df[pd.to_numeric(df[col_week], errors="coerce").astype("Int64") == week]
    if col_manager:
        df = df[df[col_manager].astype(str).str.strip() == str(manager).strip()]

    if df.empty:
        st.warning("No record found for the selected Manager, Week, and Year.")
        return

    row = df.iloc[0]

    wins_to_date = _to_int(_val(row, col_wins_to_date, None), None)
    losses_to_date = _to_int(_val(row, col_losses_to_date, None), None)
    seed_to_date = _to_int(_val(row, col_seed_to_date, None), None)

    avg_seed = _val(row, col_avg_seed, None)
    p_playoffs = _val(row, col_p_playoffs, None)
    p_champ = _val(row, col_p_champ, None)

    shuffle_avg_wins = _val(row, col_shuffle_avg_wins, None)
    shuffle_avg_playoffs = _val(row, col_shuffle_avg_playoffs, None)

    # --- Line 1: Record + seed message
    seed_msg = None
    if seed_to_date is not None:
        if 1 <= seed_to_date <= 2:
            seed_msg = "Good job getting yourself in position for a bye week! You can smell that championship!"
        elif 3 <= seed_to_date <= 6:
            seed_msg = "Good job getting in playoff position! Keep it up!"
        elif seed_to_date >= 7:
            seed_msg = "You are not in playoff position."
    line1 = (
        f"So far your record is "
        f"({wins_to_date if wins_to_date is not None else 'N/A'} - "
        f"{losses_to_date if losses_to_date is not None else 'N/A'}) and you would be the "
        f"({seed_to_date if seed_to_date is not None else 'N/A'} seed) in the playoffs if the season ended today."
    )
    if seed_msg:
        line1 += f" {seed_msg}"
    st.markdown(_apply_bolding(line1))

    # --- Line 2: Projections + postseason chance message
    proj_msg = None
    p_raw = _to_float(p_playoffs, None)
    if p_raw is not None:
        p_pct = p_raw * 100.0 if 0.0 <= p_raw <= 1.0 else p_raw
        if p_pct > 80:
            proj_msg = "You got this locked up!"
        elif 50 <= p_pct <= 80:
            proj_msg = "You're in good position to get a postseason spot. Just don't get cocky."
        elif 25 < p_pct < 50:
            proj_msg = "It is an uphill battle to make the playoffs from here but you can do it!"
        elif 0 <= p_pct <= 25:
            proj_msg = "Time to start planning for next year, bud."
    line2 = (
        "Based on current projections you are expected to finish the season with a projected final playoff position of "
        f"({_fmt_number(avg_seed)} seed), a ({_fmt_percent(p_playoffs)} chance) of making the postseason, and "
        f"({_fmt_percent(p_champ)} chance) of winning the championship."
    )
    if proj_msg:
        line2 += f" {proj_msg}"
    st.markdown(_apply_bolding(line2))

    # --- Line 3a: Shuffled wins sentence (single paragraph with tiered message)
    wins_vs_shuffle = _to_float(_val(row, col_wins_vs_shuffle, None), None)
    if wins_vs_shuffle is None:
        wt = _to_float(wins_to_date, None)
        sav = _to_float(shuffle_avg_wins, None)
        if wt is not None and sav is not None:
            wins_vs_shuffle = wt - sav

    line3a = (
        "Based on every possible schedule and opponent, you should have about "
        f"({_fmt_number(shuffle_avg_wins)} wins) so far this year."
    )

    # Append the extra-wins message into the same paragraph
    extra_msg = None
    if wins_vs_shuffle is not None:
        diff = _to_float(wins_vs_shuffle, None)
        if diff is not None:
            if diff > 1.0:
                extra_msg = (
                    f"You have been gifted ({_fmt_number(diff)} extra wins) so far this year because of an easy schedule, but who's counting?"
                )
            elif 0.50 <= diff <= 1.0:
                extra_msg = "Your schedule is giving you a little extra help."
            elif -0.49 <= diff <= 0.49:
                extra_msg = "Your record is just about where it should be."
            elif -1.0 <= diff <= -0.50:
                extra_msg = "You are getting a little unlucky with your schedule."
            elif diff < -1.0:
                extra_msg = (
                    f"You should have ({_fmt_number(abs(diff))} more wins) this year. "
                    "Your schedule has derailed your season. You can blame it all on the schedule."
                )

    if extra_msg:
        line3a += f" {extra_msg}"

    st.markdown(_apply_bolding(line3a))

    # --- Line 3b: Shuffle playoff-position rate + message based on current seed
    shuffle_msg = None
    sp_raw = _to_float(shuffle_avg_playoffs, None)
    if sp_raw is not None and seed_to_date is not None:
        sp_pct = sp_raw * 100.0 if 0.0 <= sp_raw <= 1.0 else sp_raw
        if sp_pct > 50 and seed_to_date >= 7:
            shuffle_msg = "Life's not fair!"
        elif sp_pct < 50 and seed_to_date >= 7:
            shuffle_msg = "Next time, try and score more points. Ok?"
        elif sp_pct > 50 and seed_to_date <= 6:
            shuffle_msg = "You earned your spot in the standings! Keep it up"
        elif sp_pct < 50 and seed_to_date <= 6:
            shuffle_msg = "But hey, you're living in the real world. Don't listen to the haters, they're just jealous."
    line3b = (
        f"About ({_fmt_percent(shuffle_avg_playoffs)} of possible schedules) would currently have you in playoff position."
    )
    if shuffle_msg:
        line3b += f" {shuffle_msg}"
    st.markdown(_apply_bolding(line3b))

    # ========================================================================
    # ADD CONTEXTUAL MESSAGES - Removed to avoid contradictions
    # ========================================================================
    # The narrative above already covers playoff positioning, schedule luck,
    # and alternate schedules. Adding more messages here creates contradictions.
