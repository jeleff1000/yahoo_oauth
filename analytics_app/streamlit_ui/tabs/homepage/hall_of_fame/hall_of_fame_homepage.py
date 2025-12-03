# tabs/hall_of_fame/hall_of_fame_homepage.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import streamlit as st
import duckdb
import pandas as pd
import importlib.util
import importlib.machinery
import sys
from pathlib import Path

# Ensure streamlit_ui directory is in path for imports
_streamlit_ui_dir = Path(__file__).parent.parent.parent.parent.resolve()
if str(_streamlit_ui_dir) not in sys.path:
    sys.path.insert(0, str(_streamlit_ui_dir))

from shared.dataframe_utils import as_dataframe, get_matchup_df

# Import all Hall of Fame modules
from .playoff_brackets import PlayoffBracketsViewer
from .legendary_games import LegendaryGamesViewer
from .records import RecordsViewer
from .leaderboards import LeaderboardsViewer
from .styles import apply_hall_of_fame_styles

_FALLBACK_CSV = "matchup.csv"

# ---------- Styling / Helpers ----------
@st.fragment
def _render_df(df: pd.DataFrame, highlight_cols: Tuple[str, ...] = (), index: bool = False):
    """Pretty, consistent dataframe styling with auto numeric formatting."""
    if df is None or df.empty:
        st.info("No data available.")
        return
    df_fmt = df.copy()
    # Treat any column that looks like a year as a string to avoid thousands separators (e.g. "2,015").
    for c in df_fmt.columns:
        if "year" in c.lower():
            # Safely convert values to integer-like strings when possible
            df_fmt[c] = df_fmt[c].map(lambda x: None if pd.isna(x) else str(int(float(x))) if str(x).strip() != "" else None)

    for c in df_fmt.columns:
        if pd.api.types.is_numeric_dtype(df_fmt[c]):
            if any(k in c.lower() for k in ["ppg", "avg", "mean", "margin", "proj", "points", "score_num"]):
                df_fmt[c] = df_fmt[c].map(lambda x: None if pd.isna(x) else round(float(x), 2))
            else:
                df_fmt[c] = df_fmt[c].map(lambda x: None if pd.isna(x) else round(float(x), 1))

    col_cfg = {}
    for c in df_fmt.columns:
        pretty = c.replace("_", " ").title()
        col_cfg[c] = st.column_config.Column(label=pretty)

    st.dataframe(
        df_fmt,
        use_container_width=True,
        hide_index=not index,
        column_config=col_cfg,
    )

def _col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first column in df (case-insensitive match) from candidates."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def _required(df: pd.DataFrame, *candidates: str) -> str:
    name = _col(df, *candidates)
    if not name:
        raise KeyError(f"Missing required column; tried {candidates}")
    return name

def _numericify(df: pd.DataFrame, cols: Tuple[str, ...]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _flagify(series: pd.Series) -> pd.Series:
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    s = series.map(lambda x: mapping.get(x, mapping.get(str(x).lower(), None)))
    s = s.where(s.notna(), pd.to_numeric(series, errors="coerce"))
    s = s.fillna(series.astype(str).str.lower().isin(["true", "t", "yes", "y"]).astype(int))
    return s.fillna(0).astype(int)

def _pill_nav(options: List[str], key: str, default: str) -> str:
    # Lightweight segmented control using radio (horizontal)
    choice = st.radio(
        "View",
        options,
        index=options.index(default) if default in options else 0,
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )
    return choice

def _apply_hero():
    st.markdown("""
        <div style="background: linear-gradient(135deg,
                    var(--gradient-start, rgba(102, 126, 234, 0.1)) 0%,
                    var(--gradient-end, rgba(118, 75, 162, 0.06)) 100%);
                    padding: 1.75rem 1.5rem; border-radius: 16px; margin-bottom: 1.25rem;
                    border: 1px solid var(--border, #E5E7EB);">
          <h1 style="margin:0; color: var(--text-primary, #1F2937); font-size: 2.2rem; text-align: center;">
            🏛️ KMFFL Hall of Fame
          </h1>
          <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary, #6B7280);
                     text-align: center; font-size: 1.05rem;">
            Champions, legends, records — and the full playoff story in one place.
          </p>
        </div>
    """, unsafe_allow_html=True)

class HallOfFameViewer:
    def __init__(self, df_dict: Optional[Dict[str, Any]]):
        self.df: Optional[pd.DataFrame] = get_matchup_df(df_dict)
        if self.df is None:
            try:
                self.df = pd.read_csv(_FALLBACK_CSV)
            except Exception:
                self.df = None

        if self.df is not None and not self.df.empty:
            df = self.df.copy()

            # Detect columns (case-insensitive)
            self.year_col = _required(df, "year")
            self.manager_col = _required(df, "manager", "Manager")
            self.opp_col = _required(df, "opponent", "Opponent")
            self.week_col = _required(df, "week")
            self.win_col = _required(df, "win", "Win")
            self.loss_col = _col(df, "loss", "Loss")
            self.is_playoffs_col = _required(df, "is_playoffs", "Is_Playoffs", "playoffs")
            self.is_consolation_col = _col(df, "is_consolation", "Is_Consolation")
            self.team_pts_col = _required(df, "team_points", "Team_Points")
            self.opp_pts_col = _required(df, "opponent_points", "Opponent_Points")

            # Optional extras
            self.team_proj_col = _col(df, "team_projected_points")
            self.opp_proj_col = _col(df, "opponent_projected_points")
            self.season_mean_col = _col(df, "personal_season_mean")
            self.final_w_col = _col(df, "final_wins")
            self.final_l_col = _col(df, "final_losses")

            # Championship flags if present
            self.championship_flag_col = _col(df, "championship", "is_championship", "final_game")
            self.champion_flag_col = _col(df, "champion", "is_champion")

            # Coerce types
            _numericify(df, (self.year_col, self.week_col, self.team_pts_col, self.opp_pts_col))
            for maybe_num in (self.team_proj_col, self.opp_proj_col, self.season_mean_col, self.final_w_col, self.final_l_col):
                if maybe_num:
                    _numericify(df, (maybe_num,))

            # Standardize flags to 0/1
            for f in (self.win_col, self.is_playoffs_col):
                df[f] = _flagify(df[f])
            if self.is_consolation_col:
                df[self.is_consolation_col] = _flagify(df[self.is_consolation_col])
            if self.champion_flag_col:
                df[self.champion_flag_col] = _flagify(df[self.champion_flag_col])
            if self.championship_flag_col:
                df[self.championship_flag_col] = _flagify(df[self.championship_flag_col])

            self.df = df
        else:
            self.year_col = self.manager_col = self.opp_col = self.week_col = self.win_col = None
            self.is_playoffs_col = self.is_consolation_col = None
            self.team_pts_col = self.opp_pts_col = None
            self.championship_flag_col = self.champion_flag_col = None
            self.team_proj_col = self.opp_proj_col = self.season_mean_col = None
            self.final_w_col = self.final_l_col = None

        self.con = duckdb.connect(database=":memory:")
        if self.df is not None and not self.df.empty:
            self.con.register("m", self.df)

    # ---------- Core playoff queries ----------
    def _championship_rows(self) -> pd.DataFrame:
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        if self.championship_flag_col:
            champ_where = (
                f"COALESCE(CAST({self.championship_flag_col} AS INT),0)=1 "
                f"AND CAST({self.win_col} AS INT)=1"
            )
        else:
            champ_where = (
                f"CAST({self.is_playoffs_col} AS INT)=1 AND CAST({self.win_col} AS INT)=1 "
                f"AND {self.week_col} = (SELECT max({self.week_col}) FROM m mm WHERE mm.{self.year_col}=m.{self.year_col})"
            )
        q = f"""
          WITH champ_candidates AS (
            SELECT
              CAST({self.year_col} AS INT) AS year,
              {self.manager_col} AS winner,
              {self.opp_col} AS runner_up,
              CAST({self.team_pts_col} AS DOUBLE) AS winner_pts,
              CAST({self.opp_pts_col} AS DOUBLE) AS runner_pts,
              ROUND(CAST({self.team_pts_col} AS DOUBLE),1) || ' - ' || ROUND(CAST({self.opp_pts_col} AS DOUBLE),1) AS score
            FROM m
            WHERE {champ_where}
            QUALIFY ROW_NUMBER() OVER (
              PARTITION BY CAST({self.year_col} AS INT)
              ORDER BY CAST({self.team_pts_col} AS DOUBLE) DESC
            ) = 1
          )
          SELECT year, winner, runner_up, score, winner_pts, runner_pts
          FROM champ_candidates
          ORDER BY year DESC;
        """
        return self.con.execute(q).fetchdf()

    def _rings(self) -> pd.DataFrame:
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        if self.championship_flag_col:
            where_clause = (
                f"COALESCE(CAST({self.championship_flag_col} AS INT),0)=1 "
                f"AND CAST({self.win_col} AS INT)=1"
            )
        else:
            where_clause = (
                f"CAST({self.is_playoffs_col} AS INT)=1 "
                f"AND CAST({self.win_col} AS INT)=1 "
                f"AND {self.week_col} = (SELECT max({self.week_col}) FROM m mm WHERE mm.{self.year_col}=m.{self.year_col})"
            )
        q = f"""
          SELECT {self.manager_col} AS manager, COUNT(*) AS rings
          FROM m
          WHERE {where_clause}
          GROUP BY 1
          ORDER BY rings DESC, manager
        """
        return self.con.execute(q).fetchdf()

    def _finals_appearances(self) -> pd.DataFrame:
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        if self.championship_flag_col:
            champ_where = f"COALESCE(CAST({self.championship_flag_col} AS INT),0)=1"
        else:
            champ_where = (
                f"CAST({self.is_playoffs_col} AS INT)=1 "
                f"AND {self.week_col} = (SELECT max({self.week_col}) FROM m mm WHERE mm.{self.year_col}=m.{self.year_col})"
            )
        q = f"""
          WITH finals AS (
            SELECT {self.year_col} AS year, {self.manager_col} AS mgr FROM m WHERE {champ_where}
            UNION ALL
            SELECT {self.year_col} AS year, {self.opp_col} AS mgr FROM m WHERE {champ_where}
          )
          SELECT CAST(year AS INT) AS year, mgr AS manager
          FROM finals
          ORDER BY year DESC
        """
        finals = self.con.execute(q).fetchdf()
        if finals.empty:
            return finals
        tally = finals.value_counts("manager").reset_index()
        tally.columns = ["manager", "appearances"]
        return tally.sort_values(["appearances", "manager"], ascending=[False, True])

    def _best_playoff_ppg(self) -> pd.DataFrame:
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        q = f"""
            WITH playoff_games AS (
              SELECT {self.manager_col} AS manager,
                     CAST({self.team_pts_col} AS DOUBLE) AS pts
              FROM m
              WHERE CAST({self.is_playoffs_col} AS INT)=1
                AND COALESCE(CAST({self.is_consolation_col} AS INT),0)=0
            )
            SELECT manager,
                   COUNT(*) AS games,
                   ROUND(AVG(pts), 2) AS ppg
            FROM playoff_games
            GROUP BY 1
            HAVING COUNT(*) >= 3
            ORDER BY ppg DESC, games DESC, manager
            LIMIT 50
        """
        return self.con.execute(q).fetchdf()

    def _blowouts(self) -> pd.DataFrame:
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        # De-duplicate matchups (one row per unique matchup) by creating a normalized matchup key
        # and keeping the row with the higher team points so we don't show the same game twice.
        q = f"""
          WITH pl AS (
            SELECT
              {self.manager_col} AS manager,
              {self.opp_col} AS opponent,
              CAST({self.year_col} AS INT) AS year,
              CAST({self.week_col} AS INT) AS week,
              CAST({self.team_pts_col} AS DOUBLE) AS for_pts,
              CAST({self.opp_pts_col} AS DOUBLE) AS ag_pts,
              CAST({self.win_col} AS INT) AS win,
              -- normalized key to identify the matchup regardless of ordering
              CASE WHEN LOWER(CAST({self.manager_col} AS VARCHAR)) <= LOWER(CAST({self.opp_col} AS VARCHAR))
                   THEN LOWER(CAST({self.manager_col} AS VARCHAR)) || '|' || LOWER(CAST({self.opp_col} AS VARCHAR))
                   ELSE LOWER(CAST({self.opp_col} AS VARCHAR)) || '|' || LOWER(CAST({self.manager_col} AS VARCHAR)) END AS match_key
            FROM m
            WHERE CAST({self.is_playoffs_col} AS INT)=1
              AND COALESCE(CAST({self.is_consolation_col} AS INT),0)=0
          )
          SELECT manager, opponent, year, week,
                 ROUND(for_pts - ag_pts, 2) AS margin,
                 for_pts, ag_pts, win
          FROM pl
          QUALIFY ROW_NUMBER() OVER (PARTITION BY year, week, match_key ORDER BY for_pts DESC) = 1
          ORDER BY ABS(margin) DESC
          LIMIT 25
        """
        return self.con.execute(q).fetchdf()

    def _projection_upsets(self) -> pd.DataFrame:
        if self.df is None or self.df.empty or not (self.team_proj_col and self.opp_proj_col):
            return pd.DataFrame()
        # De-duplicate matchups (one row per unique matchup) same as in _blowouts so upsets aren't doubled.
        q = f"""
          WITH pl AS (
            SELECT
              {self.manager_col} AS manager,
              {self.opp_col} AS opponent,
              CAST({self.year_col} AS INT) AS year,
              CAST({self.week_col} AS INT) AS week,
              CAST({self.team_pts_col} AS DOUBLE) AS for_pts,
              CAST({self.opp_pts_col} AS DOUBLE) AS ag_pts,
              CAST({self.team_proj_col} AS DOUBLE) AS proj_for,
              CAST({self.opp_proj_col} AS DOUBLE) AS proj_ag,
              CAST({self.win_col} AS INT) AS win,
              CASE WHEN LOWER(CAST({self.manager_col} AS VARCHAR)) <= LOWER(CAST({self.opp_col} AS VARCHAR))
                   THEN LOWER(CAST({self.manager_col} AS VARCHAR)) || '|' || LOWER(CAST({self.opp_col} AS VARCHAR))
                   ELSE LOWER(CAST({self.opp_col} AS VARCHAR)) || '|' || LOWER(CAST({self.manager_col} AS VARCHAR)) END AS match_key
            FROM m
            WHERE CAST({self.is_playoffs_col} AS INT)=1
              AND COALESCE(CAST({self.is_consolation_col} AS INT),0)=0
          )
          SELECT manager, opponent, year, week,
                 ROUND(proj_for - proj_ag, 2) AS proj_edge,
                 ROUND(for_pts - ag_pts, 2) AS actual_margin
          FROM pl
          WHERE (proj_for < proj_ag AND win=1) OR (proj_for > proj_ag AND win=0)
          QUALIFY ROW_NUMBER() OVER (PARTITION BY year, week, match_key ORDER BY for_pts DESC) = 1
          ORDER BY ABS(proj_edge) DESC, ABS(actual_margin) DESC
          LIMIT 25
        """
        return self.con.execute(q).fetchdf()

    # ---------- Section renderers used inside the single Playoffs tab ----------
    @st.fragment
    def _render_playoff_kpis(self, champs_df: pd.DataFrame):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Championships", len(champs_df))
        with col2:
            st.metric("Unique Champions", champs_df["winner"].nunique() if not champs_df.empty else 0)
        with col3:
            try:
                max_score = champs_df["winner_pts"].max() if "winner_pts" in champs_df else None
                st.metric("Highest Champ Score", f"{max_score:.1f}" if max_score is not None else "—")
            except Exception:
                st.metric("Highest Champ Score", "—")
        with col4:
            if not champs_df.empty:
                st.metric("Era", f"{int(champs_df['year'].min())}–{int(champs_df['year'].max())}")
            else:
                st.metric("Era", "—")

    @st.fragment
    def _render_champions(self):
        st.markdown("""
            <div class='hof-gradient-header hof-header-gold'>
                <h3>🏆 Championship History</h3>
                <p>Every champion, every season — the legacy of greatness</p>
            </div>
        """, unsafe_allow_html=True)

        champs = self._championship_rows()
        if champs.empty:
            st.info("No championship rows found.")
            return

        # Dynasty badges - enhanced styling
        counts = champs["winner"].value_counts().sort_values(ascending=False)
        if len(counts) > 0:
            st.markdown("##### 👑 Championship Leaders")
            badge_cols = st.columns(min(6, max(1, len(counts))))
            for i, (mgr, cnt) in enumerate(counts.items()):
                with badge_cols[i % len(badge_cols)]:
                    if cnt >= 3:
                        badge_class = "hof-champ-badge-gold"
                        icon = "🔥"
                        label = "Dynasty"
                    elif cnt >= 2:
                        badge_class = "hof-champ-badge-silver"
                        icon = "⭐"
                        label = "Multi-time"
                    else:
                        badge_class = "hof-champ-badge-bronze"
                        icon = "🏆"
                        label = "Champion"

                    st.markdown(f"""
                        <div class='hof-champ-badge {badge_class}'>
                            <div class='badge-icon'>{icon}</div>
                            <div class='badge-name'>{mgr}</div>
                            <div class='badge-count'>{cnt} {'Title' if cnt == 1 else 'Titles'}</div>
                            <div class='badge-label'>{label}</div>
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📅 Championship Timeline")

        # Show just the last 6 championships as compact tiles
        recent_champs = champs.head(6)
        num_cols = 3
        for i in range(0, len(recent_champs), num_cols):
            cols = st.columns(num_cols)
            for j, col in enumerate(cols):
                if i + j < len(recent_champs):
                    row = recent_champs.iloc[i + j]
                    try:
                        year = int(float(row['year']))
                    except:
                        year = row['year']
                    winner = str(row['winner'])
                    runner_up = str(row['runner_up'])
                    score = str(row['score'])

                    with col:
                        # Compact card design
                        st.markdown(f"""
                            <div class='hof-timeline-card'>
                                <div class='timeline-year'>{year}</div>
                                <div class='timeline-winner'>🏆 {winner}</div>
                                <div class='timeline-details'>def. {runner_up} • {score}</div>
                            </div>
                        """, unsafe_allow_html=True)

        # Show expander for full championship history if there are more
        if len(champs) > 6:
            with st.expander(f"📜 View All {len(champs)} Championships"):
                _render_df(champs[["year", "winner", "runner_up", "score"]])

    @st.fragment
    def _render_rings(self):
        st.markdown("""
            <div class='hof-gradient-header hof-header-purple'>
                <h4>💍 Rings Leaderboard</h4>
                <p>Total championships won by each manager</p>
            </div>
        """, unsafe_allow_html=True)

        rings = self._rings()
        if rings.empty:
            st.info("No ring data found.")
            return

        # Two-column layout: table + chart
        col1, col2 = st.columns([1, 1])
        with col1:
            _render_df(rings)
        with col2:
            try:
                st.bar_chart(rings.set_index("manager")["rings"], height=300)
            except Exception:
                pass

    @st.fragment
    def _render_finals(self):
        st.markdown("""
            <div class='hof-gradient-header hof-header-red'>
                <h4>🥈 Finals Appearances</h4>
                <p>Managers who made it to the championship game</p>
            </div>
        """, unsafe_allow_html=True)

        finals = self._finals_appearances()
        if finals.empty:
            st.info("No finals appearance data found.")
            return
        _render_df(finals)

    @st.fragment
    def _render_ppg(self):
        st.markdown("""
            <div class='hof-gradient-header hof-header-orange'>
                <h4>🔥 Best Playoff PPG</h4>
                <p>Highest scoring averages in playoff games (min. 3 games)</p>
            </div>
        """, unsafe_allow_html=True)

        ppg = self._best_playoff_ppg()
        if ppg.empty:
            st.info("No playoff game data found.")
            return
        _render_df(ppg)

    @st.fragment
    def _render_blowouts_and_upsets(self):
        st.markdown("""
            <div class='hof-gradient-header hof-header-fire'>
                <h3>💥 Largest Playoff Margins</h3>
                <p>The most dominant playoff victories in league history</p>
            </div>
        """, unsafe_allow_html=True)

        blow = self._blowouts()
        if blow.empty:
            st.info("No playoff margin data found.")
        else:
            _render_df(blow)

        upsets = self._projection_upsets()
        if not upsets.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class='hof-gradient-header hof-header-violet'>
                    <h3>😲 Biggest Projection Upsets</h3>
                    <p>When the underdog defied the odds and shocked the favorite</p>
                </div>
            """, unsafe_allow_html=True)
            _render_df(upsets)

    @st.fragment
    def _render_brackets(self):
        st.markdown("#### 🗺️ Playoff Brackets")
        PlayoffBracketsViewer(self.df).display()

    # ---------- Main page ----------
    @st.fragment
    def display(self, section: Optional[str] = None):
        try:
            apply_hall_of_fame_styles()
        except Exception:
            pass

        _apply_hero()

        if self.df is None or self.df.empty:
            st.error("📊 Matchup Data not available.")
            return

        # ===== 4 main tabs (merged Top Teams + Top Players into Leaderboards) =====
        tabs = st.tabs([
            "🏆 Playoffs",
            "🎮 Legendary Games",
            "📊 Records",
            "👑 Leaderboards",
        ])

        # ------------------- PLAYOFFS TAB -------------------
        with tabs[0]:
            # Optional quick filters (improve clarity, but non-destructive)
            with st.expander("Filters", expanded=False):
                years = sorted(self.df[self.year_col].dropna().astype(int).unique().tolist())
                managers = sorted(self.df[self.manager_col].dropna().astype(str).unique().tolist())

                sel_years = st.multiselect("Years", years, default=years)
                sel_mgrs = st.multiselect("Managers (any role)", managers, default=[])
                include_consolation = st.toggle("Include consolation games", value=False, help="Affects some sections (blowouts/PPG)")

                # Filter view dataframe for subviews
                dfv = self.df.copy()
                if sel_years:
                    dfv = dfv[dfv[self.year_col].astype(int).isin(sel_years)]
                if sel_mgrs:
                    dfv = dfv[(dfv[self.manager_col].astype(str).isin(sel_mgrs)) | (dfv[self.opp_col].astype(str).isin(sel_mgrs))]
                if not include_consolation and self.is_consolation_col in dfv.columns:
                    dfv = dfv[dfv[self.is_consolation_col] == 0]
                # Re-register filtered view for queries in this tab
                self.con.unregister("m")
                self.con.register("m", dfv)

            st.markdown("### Playoff Overview")
            # Combine key playoff summary views into real Streamlit tabs (not radio buttons).
            # Place the requested `section` first so it appears selected by default.
            tab_names = ["Championships", "Blowouts & Upsets", "Brackets"]
            default_name = section or "Championships"
            ordered = tab_names.copy()
            if default_name in ordered:
                ordered.remove(default_name)
                ordered.insert(0, default_name)

            sub_tabs = st.tabs(ordered)
            for i, name in enumerate(ordered):
                with sub_tabs[i]:
                    if name == "Championships":
                        # Render the main championship-related sections together
                        self._render_champions()
                        st.markdown("---")
                        self._render_ppg()
                    elif name == "Blowouts & Upsets":
                        self._render_blowouts_and_upsets()
                    elif name == "Brackets":
                        self._render_brackets()

        # ------------------- OTHER TABS -------------------
        with tabs[1]:
            LegendaryGamesViewer(self.df).display()
        with tabs[2]:
            RecordsViewer(self.df).display()
        with tabs[3]:
            LeaderboardsViewer(self.df).display()
