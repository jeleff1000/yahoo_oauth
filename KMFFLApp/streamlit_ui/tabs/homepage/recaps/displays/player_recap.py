#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Optional
import re
import html
import pandas as pd
import streamlit as st

# Import safe helpers for NA/NaN handling
from ..narrative_engine import (
    build_player_spotlight_lines,
    build_player_spotlight_paragraph,
)

# -----------------------------
# Heuristics / constants
# -----------------------------
PLAYER_POINTS_CANDIDATES = [
    "points", "Points", "weekly_points", "Week Points", "rolling_point_total"
]
FANTASY_POSITION_CANDIDATES = [
    "fantasy_position", "fantasy pos", "fantasyposition",
    "fantasy_pos", "lineup_slot", "lineup_position", "slot",
    "roster_slot", "Roster_Slot", "roster_position", "player_fantasy_position"
]
EXCLUDED_LINEUP_POSITIONS = {"IR", "BN"}

AWARD_CSS = """
<style>
.awards-row {display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 18px 0;}
.awards-row.row1 .award-card,.awards-row.row2 .award-card {flex:1 1 48%;max-width:48%;}
@media (max-width:960px){
  .awards-row.row1 .award-card,.awards-row.row2 .award-card {flex:1 1 100%;max-width:100%;}
}
.award-card {position:relative;display:flex;align-items:center;gap:0.85rem;padding:10px 16px 10px 12px;
  border-radius:18px;box-shadow:0 3px 8px rgba(0,0,0,0.18);border:2px solid #334155;background:#f1f5f9;min-height:82px;overflow:hidden;}
.award-card.star {background:linear-gradient(135deg,#ecfdf5,#a7f3d0,#34d399);border-color:#059669;}
.award-card.dud {background:linear-gradient(135deg,#fef2f2,#fecaca,#f87171);border-color:#dc2626;}
.award-card.whatif {background:linear-gradient(135deg,#eef2ff,#c7d2fe,#818cf8);border-color:#6366f1;}
.award-card.improved {background:linear-gradient(135deg,#fefce8,#fde68a,#facc15);border-color:#f59e0b;}
.award-img-wrap {position:relative;width:70px;height:70px;flex-shrink:0;border-radius:50%;background:#fff;padding:6px;
  display:flex;align-items:center;justify-content:center;overflow:hidden;box-shadow:0 2px 4px rgba(0,0,0,0.15);}
.award-img-wrap.star {box-shadow:0 0 0 4px #6ee7b7,0 0 0 8px #34d39933;}
.award-img-wrap.dud {box-shadow:0 0 0 4px #f87171,0 0 0 8px #dc262633;}
.award-img-wrap.whatif {box-shadow:0 0 0 4px #818cf8,0 0 0 8px #6366f133;}
.award-img-wrap.improved {box-shadow:0 0 0 4px #fbbf24,0 0 0 8px #f59e0b33;}
.award-img-wrap img {width:100%;height:100%;object-fit:cover;border-radius:50%;border:2px solid #334155;}
.award-emoji {position:absolute;right:12px;top:50%;transform:translateY(-50%);font-size:40px;line-height:1;
  filter:drop-shadow(0 2px 2px rgba(0,0,0,0.25));pointer-events:none;user-select:none;}
.award-content {display:flex;flex-direction:column;justify-content:center;min-width:0;padding-right:60px;}
.award-content h4 {margin:0 0 3px 0;font-size:0.82rem;font-weight:800;color:#1e293b;letter-spacing:.3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:360px;}
.award-content .player-name {margin:0 0 3px 0;font-size:1.12rem;font-weight:800;color:#0f172a;line-height:1.20rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:360px;}
.award-content .pts-line {font-size:0.78rem;font-weight:600;color:#374151;line-height:0.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:360px;}
.award-content .pts-line span.value {font-weight:700;color:#111827;}
.no-img {font-size:0.60rem;font-weight:600;text-align:center;color:#475569;}
</style>
"""

# -----------------------------
# Small DF utilities
# -----------------------------
def _get_player_df(df_dict: Optional[Dict[Any, Any]]) -> Optional[pd.DataFrame]:
    if not isinstance(df_dict, dict):
        return None
    obj = df_dict.get("Player Data")
    if obj is None:
        for k, v in df_dict.items():
            if str(k).strip().lower() == "player data":
                obj = v
                break
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None
    return None

def _detect_points_col(df: pd.DataFrame) -> Optional[str]:
    for c in PLAYER_POINTS_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if "point" in c.lower():
            return c
    return None

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _to_int(v, default=None) -> Optional[int]:
    """Convert value to int, return default if not possible"""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return int(float(v))
    except Exception:
        return default

def _to_float(v, default=None) -> Optional[float]:
    """Convert value to float, return default if not possible"""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    except Exception:
        return default

def _find_headshot_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["headshot_url","headshot","player_headshot","Headshot","Headshot_URL","image_url","player_image","Player_Headshot"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "head" in c.lower() and "shot" in c.lower():
            return c
    return None

def _find_fantasy_position_col(df: pd.DataFrame) -> Optional[str]:
    targets = {re.sub(r"\W+","", s.lower()) for s in FANTASY_POSITION_CANDIDATES}
    for c in df.columns:
        if re.sub(r"\W+","", c.lower()) in targets:
            return c
    for c in df.columns:
        lc = c.lower()
        if "fantasy" in lc and ("pos" in lc or "position" in lc):
            return c
    return None

def _find_manager_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["manager","Manager","manager_name","owner","Owner","owner_name","team_owner","team_manager"]:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = c.lower()
        if "manager" in lc or "owner" in lc:
            return c
    return None

def _filter_manager_year(df: pd.DataFrame, manager: str, year: int) -> pd.DataFrame:
    year_col = next((c for c in ["season_year","year","Year","season","Season"] if c in df.columns), None)
    mgr_col  = _find_manager_col(df)
    out = df
    if year_col:
        out = out[_safe_numeric(out[year_col]) == year]
    if mgr_col:
        m = out[mgr_col].astype(str).str.lower() == str(manager).lower()
        if m.any():
            out = out[m]
    return out

# -----------------------------
# Two-week slicer (selected week + previous cumulative)
# -----------------------------

def _detect_week_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["week", "Week"]:
        if c in df.columns:
            return c
    return None

def _detect_year_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["season_year","year","Year","season","Season"]:
        if c in df.columns:
            return c
    return None

def _detect_cum_week_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["cumulative_week","cumulativeweek","cumul_week","cumulweek"]:
        if c in df.columns:
            return c
    # fallback to week if cumulative not present
    return _detect_week_col(df)

def _two_week_slice(player_df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    if player_df is None or player_df.empty:
        return player_df.iloc[0:0]
    cum_col  = _detect_cum_week_col(player_df)
    week_col = _detect_week_col(player_df)
    year_col = _detect_year_col(player_df)
    if not cum_col or not week_col:
        return player_df.iloc[0:0]

    df = player_df.copy()
    df["_cum_"] = _safe_numeric(df[cum_col])
    df["_wk_"]  = _safe_numeric(df[week_col])
    if year_col:
        df["_yr_"] = _safe_numeric(df[year_col])

    cur_mask = (df["_wk_"] == week) & ((df["_yr_"] == year) if year_col else True)
    cur_rows = df[cur_mask]
    if cur_rows.empty:
        return player_df.iloc[0:0]

    cur_cums = cur_rows["_cum_"].dropna().unique()
    if len(cur_cums) == 0:
        return player_df.iloc[0:0]
    cur_cum = float(max(cur_cums))
    prev_rows = df[df["_cum_"] < cur_cum]
    if prev_rows.empty:
        out = df[df["_cum_"] == cur_cum]
    else:
        prev_cum = float(prev_rows["_cum_"].max())
        out = df[df["_cum_"].isin([prev_cum, cur_cum])]
    return out.drop(columns=["_cum_","_wk_"] + (["_yr_"] if year_col else []), errors="ignore")

# -----------------------------
# Awards helpers
# -----------------------------

def _ensure_award_css() -> None:
    st.markdown(AWARD_CSS, unsafe_allow_html=True)

def _award_emoji(kind: str) -> str:
    return {"star":"🌟","dud":"💩","whatif":"🤦","improved":"📈"}.get(kind,"🏅")

@st.fragment
def _render_award(row: Optional[pd.Series],
                  name_col: str,
                  points_col: str,
                  headshot_col: Optional[str],
                  title: str,
                  kind: str) -> str:
    if row is None:
        return ""
    name = html.escape(str(row.get(name_col, "")))
    pts_val = row.get(points_col)
    pts_txt = "—"
    if isinstance(pts_val, (int, float)) and pd.notna(pts_val):
        pts_txt = f"{pts_val:.2f} pts"
    if kind == "improved":
        delta = row.get("__improvement_delta__")
        if isinstance(delta, (int,float)) and pd.notna(delta):
            pts_txt = f"+{delta:.2f} pts" if delta >= 0 else f"{delta:.2f} pts"
    url = row.get(headshot_col) if headshot_col else None
    if url is not None and pd.notna(url):
        img_html = f'<div class="award-img-wrap {kind}"><img src="{html.escape(str(url))}" alt="{title} - {name}"></div>'
    else:
        img_html = f'<div class="award-img-wrap {kind}"><div class="no-img">No Image</div></div>'
    content_html = (
        f'<div class="award-content">'
        f'<h4>{html.escape(title)}</h4>'
        f'<div class="player-name">{name}</div>'
        f'<div class="pts-line"><span class="value">{pts_txt}</span></div>'
        f'</div>'
    )
    return f'<div class="award-card {kind}">{img_html}{content_html}<div class="award-emoji">{_award_emoji(kind)}</div></div>'

def _get_exceptional_players(df: pd.DataFrame,
                            name_col: str,
                            points_col: str) -> Dict[str, Optional[pd.Series]]:
    """
    Find players with exceptional performances based on rankings and percentiles.

    Returns dict with:
        - 'elite_week': Player with elite weekly performance (top 5% at position this week)
        - 'career_best': Player having their career-best week (rank #1 in their history)
        - 'all_time_great': Player with an all-time great performance (top 100 ever)
    """
    result = {
        'elite_week': None,
        'career_best': None,
        'all_time_great': None,
    }

    if df.empty:
        return result

    # Elite weekly performance at position (top 5% this week)
    if 'position_week_pct' in df.columns:
        work = df[pd.notna(df['position_week_pct']) & pd.notna(df[points_col])].copy()
        if not work.empty:
            # Top 5% = percentile >= 95
            elite = work[work['position_week_pct'] >= 95]
            if not elite.empty:
                result['elite_week'] = elite.sort_values(points_col, ascending=False).iloc[0]

    # Career-best performance (rank #1 in player's personal history across all weeks)
    if 'player_personal_week_rank' in df.columns:
        work = df[pd.notna(df['player_personal_week_rank']) & pd.notna(df[points_col])].copy()
        if not work.empty:
            career_best = work[work['player_personal_week_rank'] == 1]
            if not career_best.empty:
                result['career_best'] = career_best.sort_values(points_col, ascending=False).iloc[0]

    # All-time great performance (top 100 ever across all players, all-time ranking)
    if 'all_players_alltime_rank' in df.columns:
        work = df[pd.notna(df['all_players_alltime_rank']) & pd.notna(df[points_col])].copy()
        if not work.empty:
            all_time = work[work['all_players_alltime_rank'] <= 100]
            if not all_time.empty:
                result['all_time_great'] = all_time.sort_values('all_players_alltime_rank', ascending=True).iloc[0]

    return result

def _compute_most_improved_two_week(df: pd.DataFrame,
                                    name_col: str,
                                    points_col: str) -> Optional[pd.Series]:
    if "cumulative_week" not in df.columns:
        return None
    work = df.copy()
    work["__cum__"] = _safe_numeric(work["cumulative_week"])
    work[points_col] = _safe_numeric(work[points_col])
    work = work[pd.notna(work["__cum__"]) & pd.notna(work[points_col])]
    if work.empty:
        return None
    cum_vals = sorted(work["__cum__"].unique())
    if len(cum_vals) < 2:
        return None
    cur_cum = cum_vals[-1]
    prev_cum = cum_vals[-2]
    # Ensure DataFrame result with explicit agg, not Series
    agg = (work[[name_col, "__cum__", points_col]]
           .groupby([name_col, "__cum__"], as_index=False)
           .agg({points_col: 'max'}))
    # Ensure DataFrame operations (avoid Series.rename static warnings)
    cur = agg[agg["__cum__"] == cur_cum].copy()
    cur["cur_pts"] = cur[points_col]
    cur = cur.drop(columns=[points_col])

    prev = agg[agg["__cum__"] == prev_cum].copy()
    prev["prev_pts"] = prev[points_col]
    prev = prev.drop(columns=[points_col])
    merged = pd.merge(cur, prev, on=name_col, how="inner")
    if merged.empty:
        return None
    merged["improvement"] = merged["cur_pts"] - merged["prev_pts"]
    merged = merged[merged["improvement"] > 0]
    if merged.empty:
        return None
    merged = merged.sort_values(["improvement","cur_pts",name_col], ascending=[False,False,True])
    best = merged.iloc[0]
    rep_row = work[(work[name_col] == best[name_col]) & (work["__cum__"] == cur_cum)].iloc[0].copy()
    # Add improvement delta without direct scalar assignment to avoid static checker issues
    rep_row = pd.concat([rep_row, pd.Series({"__improvement_delta__": float(best["improvement"])})])
    return rep_row

# -----------------------------
# Public entrypoint
# -----------------------------

@st.fragment
def display_player_weekly_recap(
    df_dict: Optional[Dict[Any, Any]],
    year: Optional[int],
    week: Optional[int],
    manager: Optional[str],
) -> None:
    if year is None or week is None or not manager:
        st.info("Select year, week and manager.")
        return

    raw_player_df = _get_player_df(df_dict)
    if raw_player_df is None or raw_player_df.empty:
        st.warning("No player data.")
        return

    # 🔽 Ensure we only work with (selected week + prior cumulative)
    player_df = _two_week_slice(raw_player_df, int(year), int(week))
    if player_df.empty:
        st.info("No rows for selected week slice.")
        return

    if "week" not in player_df.columns:
        st.warning("Column `week` missing.")
        return
    if "cumulative_week" not in player_df.columns:
        st.warning("Column `cumulative_week` missing (required for Most Improved).")
        return

    points_col = _detect_points_col(player_df)
    if not points_col:
        st.warning("Points column not found.")
        return
    player_df[points_col] = _safe_numeric(player_df[points_col])

    name_col = next((c for c in ["player","Player","player_name","Player_Name"] if c in player_df.columns), points_col)
    manager_col = _find_manager_col(player_df)
    year_col = _detect_year_col(player_df)

    # current-week players for selected manager
    if manager_col:
        cur_mask = (_safe_numeric(player_df["week"]) == week) & \
                   (player_df[manager_col].astype(str).str.lower() == str(manager).lower())
        if year_col:
            cur_mask &= (_safe_numeric(player_df[year_col]) == year)
        current_names = set(player_df.loc[cur_mask, name_col].astype(str))
    else:
        current_names = set()

    # Most Improved (needs both weeks but limited to manager’s current-week player set)
    improvement_pool = player_df[player_df[name_col].astype(str).isin(current_names)] if current_names else player_df.iloc[0:0]
    improved_row = _compute_most_improved_two_week(improvement_pool, name_col, points_col)

    # Visible rows: current week only for this manager (still sourced from two-week slice)
    subset = _filter_manager_year(player_df, manager, int(year))
    if subset.empty:
        st.info("No rows for selection.")
        return
    week_rows = subset[_safe_numeric(subset["week"]) == week]
    if week_rows.empty:
        st.info(f"No rows for Week {week}.")
        return

    fantasy_pos_col     = _find_fantasy_position_col(week_rows)
    headshot_col_all    = _find_headshot_col(player_df)
    headshot_col_subset = _find_headshot_col(week_rows)

    eligible = week_rows.copy()
    if fantasy_pos_col and fantasy_pos_col in eligible.columns:
        eligible = eligible[~eligible[fantasy_pos_col].isin(EXCLUDED_LINEUP_POSITIONS)]
    eligible = eligible[pd.notna(eligible[points_col])]
    star_row = eligible.sort_values(points_col, ascending=False).iloc[0] if not eligible.empty else None
    dud_row  = eligible.sort_values(points_col, ascending=True ).iloc[0] if not eligible.empty else None

    def _is_opt(v: Any) -> bool:
        if isinstance(v, bool): return v
        if isinstance(v, (int, float)) and not pd.isna(v): return int(v) == 1
        if isinstance(v, str): return v.strip().lower() in {"1","true","yes","y","t"}
        return False

    whatif_row = None
    if fantasy_pos_col and fantasy_pos_col in week_rows.columns:
        bench = week_rows[(week_rows[fantasy_pos_col] == "BN") & pd.notna(week_rows[points_col])].copy()
        opt_col = None
        for col in ["optimal_player", "optimal_player"]:  # keep alias
            if col in bench.columns:
                opt_col = col; break
        if opt_col:
            bench = bench[bench[opt_col].apply(_is_opt)]
        else:
            bench = bench.iloc[0:0]
        if not bench.empty:
            whatif_row = bench.sort_values(points_col, ascending=False).iloc[0]

    st.markdown("## Weekly Awards")
    _ensure_award_css()
    star_html     = _render_award(star_row,     name_col, points_col, headshot_col_subset, "Star of the Week",   "star")
    dud_html      = _render_award(dud_row,      name_col, points_col, headshot_col_subset, "Dud of the Week",    "dud")
    whatif_html   = _render_award(whatif_row,   name_col, points_col, headshot_col_subset, "I Almost Started Him!", "whatif")
    improved_html = _render_award(improved_row, name_col, points_col, headshot_col_all,    "Most Improved",      "improved")

    if any([star_html, dud_html, whatif_html, improved_html]):
        if star_html or dud_html:
            st.markdown(f"<div class='awards-row row1'>{star_html}{dud_html}</div>", unsafe_allow_html=True)
        if whatif_html or improved_html:
            st.markdown(f"<div class='awards-row row2'>{whatif_html}{improved_html}</div>", unsafe_allow_html=True)
    else:
        st.info("No award candidates.")

    # ================================================================
    # Spotlight & Context (paragraph-style)
    # ================================================================
    st.divider()
    st.subheader("🔦 Player Spotlight")

    # Check for exceptional performances
    exceptional = _get_exceptional_players(week_rows, name_col, points_col)

    # Display exceptional performance highlights first
    if exceptional['all_time_great'] is not None:
        player = exceptional['all_time_great']
        player_name = player[name_col]
        pts = _to_float(player[points_col], 0)
        rank = _to_int(player.get('all_players_alltime_rank', None), None)
        position = player.get('fantasy_position', 'player')
        st.success(f"🏆 **Historic Performance!** **{player_name}** ({position}) scored **{pts:.1f} points** - the **#{rank} best weekly performance in league history!**")

    if exceptional['career_best'] is not None:
        player = exceptional['career_best']
        player_name = player[name_col]
        pts = _to_float(player[points_col], 0)
        position = player.get('fantasy_position', 'player')
        pct = _to_float(player.get('player_personal_week_pct', None), None)
        # Only show if it's truly a career-best (rank 1 means top percentile)
        if pct is None or pct >= 99:
            st.success(f"⭐ **Career Week!** **{player_name}** ({position}) had their **#1 best game ever** with **{pts:.1f} points**!")

    if exceptional['elite_week'] is not None:
        player = exceptional['elite_week']
        player_name = player[name_col]
        pts = _to_float(player[points_col], 0)
        position = player.get('fantasy_position', 'player')
        pct = _to_float(player.get('position_week_pct', None), None)
        rank = _to_int(player.get('position_week_rank', None), None)
        if pct and pct >= 95:
            st.info(f"🔥 **Elite Weekly Performance!** **{player_name}** was **#{rank} at {position}** this week (**top {100-pct:.0f}%**) with **{pts:.1f} points**!")

    # Original spotlight paragraph
    spotlight_para = build_player_spotlight_paragraph(
        week_rows,
        points_col=points_col,
        improved_row=improved_row,
        max_players=3,
    )
    if spotlight_para:
        st.markdown(spotlight_para)
    else:
        st.caption("No standout player context available for this week.")

    # ========================================================================
    # PLAYER BREAKDOWN BY POSITION
    # ========================================================================
    st.divider()
    st.subheader("📊 Position Group Breakdown")

    if fantasy_pos_col and fantasy_pos_col in week_rows.columns:
        # Group by position and show stats
        pos_groups = week_rows[week_rows[fantasy_pos_col].notna()].groupby(fantasy_pos_col)

        cols = st.columns(4)
        col_idx = 0

        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in pos_groups.groups:
                group = pos_groups.get_group(pos)
                total_pts = group[points_col].sum() if not group.empty else 0
                avg_pts = group[points_col].mean() if not group.empty else 0
                count = len(group)

                with cols[col_idx]:
                    st.metric(
                        label=f"{pos}s ({count})",
                        value=f"{total_pts:.1f} pts",
                        delta=f"{avg_pts:.1f} avg"
                    )
                col_idx += 1

    # ========================================================================
    # DETAILED PLAYER TABLE
    # ========================================================================
    st.divider()
    st.subheader("📋 Detailed Player Statistics")

    # Create a more readable display table
    if not week_rows.empty:
        display_cols = [name_col, points_col]

        # Add position if available
        if fantasy_pos_col and fantasy_pos_col in week_rows.columns:
            display_cols.insert(1, fantasy_pos_col)

        # Add other useful columns if they exist
        useful_cols = ['opponent', 'team', 'game_status', 'projected_points']
        for col in useful_cols:
            if col in week_rows.columns:
                display_cols.append(col)

        # Filter to only existing columns
        display_cols = [c for c in display_cols if c in week_rows.columns]

        # Sort by points descending
        display_df = week_rows[display_cols].sort_values(points_col, ascending=False)

        # Rename for better display
        rename_map = {
            points_col: "Points",
            name_col: "Player",
            fantasy_pos_col: "Pos" if fantasy_pos_col else None,
        }
        rename_map = {k: v for k, v in rename_map.items() if k and v}
        display_df = display_df.rename(columns=rename_map)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Points": st.column_config.NumberColumn(
                    "Points",
                    format="%.2f",
                ),
            }
        )
    else:
        st.info("No player data to display")
