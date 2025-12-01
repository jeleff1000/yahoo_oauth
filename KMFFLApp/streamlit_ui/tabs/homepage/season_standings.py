import streamlit as st
import duckdb
import sys
from pathlib import Path

# Ensure streamlit_ui directory is in path for imports
_streamlit_ui_dir = Path(__file__).parent.parent.parent.resolve()
if str(_streamlit_ui_dir) not in sys.path:
    sys.path.insert(0, str(_streamlit_ui_dir))

from ..shared.modern_styles import apply_modern_styles
from shared.themes import detect_theme


class SeasonStandingsViewer:
    def __init__(self, df):
        self.con = duckdb.connect(database=":memory:")
        self.table = "games"
        self.con.register(self.table, df)

    def _get_theme_aware_colors(self):
        """Get colors for pandas styling (requires actual values, not CSS vars)."""
        is_dark = detect_theme() == 'dark'

        return {
            # Championship/medals - same in both modes
            'gold': '#FFD700',
            'silver': '#C0C0C0',
            'bronze': '#D4A574' if is_dark else '#CD7F32',

            # Consolation greens (adjusted for dark mode visibility)
            'green_1': '#66BB6A' if is_dark else '#81C784',
            'green_2': '#81C784' if is_dark else '#A5D6A7',
            'green_3': '#A5D6A7' if is_dark else '#C8E6C9',
            'green_4': '#C8E6C9' if is_dark else '#E8F5E9',

            # Consolation losses
            'gray_1': '#546E7A' if is_dark else '#CFD8DC',
            'gray_2': '#607D8B' if is_dark else '#ECEFF1',
            'gray_3': '#78909C' if is_dark else '#F5F5F5',

            # Quarterfinals
            'quarterfinal': '#455A64' if is_dark else '#E8E3D3',

            # Consolation semifinals
            'consol_semi': '#37474F' if is_dark else '#E3F2FD',

            # Special
            'sacko': '#EF4444',  # Using design token error color
            'in_progress': '#3B82F6',  # Using design token info color
            'missed': '#757575' if is_dark else '#9E9E9E',

            # Text colors based on background
            'text_dark': '#000000',
            'text_light': '#FFFFFF',

            # Is dark mode flag
            'is_dark': is_dark,
        }

    @st.fragment
    def display(self, prefix: str = ""):
        # Apply theme-aware styles (includes CSS variables)
        apply_modern_styles()

        colors = self._get_theme_aware_colors()
        is_dark = colors['is_dark']

        # Theme-aware CSS using CSS variables
        st.markdown("""
            <style>
            /* Standings table container - static display, no heavy shadows */
            .standings-container {
                border-radius: var(--radius-md, 8px);
                overflow: hidden;
                border: 1px solid var(--border, #e0e0e0);
            }

            /* Modern table styling */
            div[data-testid="stDataFrame"] {
                border-radius: var(--radius-md, 8px);
                overflow: hidden;
                border: 1px solid var(--border, #e0e0e0);
            }
            div[data-testid="stDataFrame"] > div {
                border: none !important;
            }

            </style>
        """, unsafe_allow_html=True)

        # Header with inline controls
        st.markdown("### Current Standings")
        per_game = st.toggle("Per Game", value=False, key=f"{prefix}_aggregation_type")

        q = f"""
        WITH base AS (
            SELECT *
            FROM {self.table}
            WHERE COALESCE(is_consolation, 0) <> 1
        ),
        -- Get max week per manager/year
        max_weeks AS (
            SELECT
                manager,
                year,
                MAX(week) as max_week
            FROM {self.table}
            GROUP BY manager, year
        ),
        -- Get final week results per manager/year (for playoff/consolation outcomes)
        final_results AS (
            SELECT
                t.manager,
                CAST(t.year AS VARCHAR) AS Year,
                MAX(t.playoff_round) as final_playoff_round,
                MAX(t.consolation_round) as final_consolation_round,
                MAX(t.win) as final_win,
                MAX(t.is_playoffs) as final_is_playoffs,
                MAX(t.is_consolation) as final_is_consolation,
                MAX(CASE WHEN COALESCE(t.sacko, 0) = 1 THEN 1 ELSE 0 END) as final_sacko
            FROM {self.table} t
            INNER JOIN max_weeks mw
                ON t.manager = mw.manager
                AND t.year = mw.year
                AND t.week = mw.max_week
            GROUP BY t.manager, t.year
        ),
        agg AS (
            SELECT
                manager                                            AS Manager,
                ANY_VALUE(team_name)                               AS Team,
                CAST(year AS VARCHAR)                              AS Year,

                { 'ROUND(AVG(CAST(team_points AS DOUBLE)), 2)' if per_game else 'SUM(CAST(team_points AS DOUBLE))' }     AS PF,
                { 'ROUND(AVG(CAST(opponent_points AS DOUBLE)), 2)' if per_game else 'SUM(CAST(opponent_points AS DOUBLE))' } AS PA,

                { 'ROUND(AVG(CASE WHEN win=1 THEN 1 ELSE 0 END), 3)' if per_game else 'SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)' } AS W,
                { 'ROUND(AVG(CASE WHEN win=1 THEN 0 ELSE 1 END), 3)' if per_game else 'SUM(CASE WHEN win=1 THEN 0 ELSE 1 END)' } AS L,

                MAX(CASE WHEN COALESCE(champion, 0) = 1 THEN 1 ELSE 0 END) AS champion,

                ANY_VALUE(final_playoff_seed) AS Seed
            FROM base
            GROUP BY manager, year
        ),
        champ_years AS (
            SELECT DISTINCT CAST(year AS VARCHAR) AS Year
            FROM base
            WHERE COALESCE(champion, 0) = 1
        ),
        with_status AS (
            SELECT
                a.Seed,
                a.Manager,
                a.Team,
                a.Year,
                a.W,
                a.L,
                a.PF,
                a.PA,
                a.champion,
                fr.final_sacko as sacko,
                fr.final_playoff_round,
                fr.final_consolation_round,
                fr.final_win,
                fr.final_is_playoffs,
                fr.final_is_consolation,
                CASE WHEN cy.Year IS NULL THEN TRUE ELSE FALSE END AS in_progress
            FROM agg a
            LEFT JOIN final_results fr ON a.Manager = fr.manager AND a.Year = fr.Year
            LEFT JOIN champ_years cy ON a.Year = cy.Year
        )
        SELECT
            Seed, Manager, Team, Year, W, L, PF, PA,
            CASE
              -- Season not complete
              WHEN in_progress THEN 'Season in Progress'

              -- Playoff Results (wins)
              WHEN champion > 0 THEN 'Won Championship'

              -- Sacko award (check early so it takes priority)
              WHEN sacko > 0 THEN 'Sacko'

              -- Playoff Results (losses)
              WHEN final_playoff_round = 'championship' AND COALESCE(final_win, 0) = 0
                THEN 'Lost Championship'
              WHEN final_playoff_round = 'semifinal' AND COALESCE(final_win, 0) = 0
                THEN 'Lost Semifinals'
              WHEN final_playoff_round = 'quarterfinal' AND COALESCE(final_win, 0) = 0
                THEN 'Lost Quarterfinals'

              -- Consolation Results (wins and losses)
              WHEN final_consolation_round = 'third_place_game' AND COALESCE(final_win, 0) = 1
                THEN 'Won Third Place Game'
              WHEN final_consolation_round = 'third_place_game' AND COALESCE(final_win, 0) = 0
                THEN 'Lost Third Place Game'

              WHEN final_consolation_round = 'fifth_place_game' AND COALESCE(final_win, 0) = 1
                THEN 'Won Fifth Place Game'
              WHEN final_consolation_round = 'fifth_place_game' AND COALESCE(final_win, 0) = 0
                THEN 'Lost Fifth Place Game'

              WHEN final_consolation_round = 'seventh_place_game' AND COALESCE(final_win, 0) = 1
                THEN 'Won Seventh Place Game'
              WHEN final_consolation_round = 'seventh_place_game' AND COALESCE(final_win, 0) = 0
                THEN 'Lost Seventh Place Game'

              WHEN final_consolation_round = 'ninth_place_game' AND COALESCE(final_win, 0) = 1
                THEN 'Won Ninth Place Game'
              WHEN final_consolation_round = 'ninth_place_game' AND COALESCE(final_win, 0) = 0
                THEN 'Lost Ninth Place Game'

              WHEN final_consolation_round = 'consolation_semifinal'
                THEN 'Lost in Consolation Semifinals'

              -- Didn't make playoffs
              ELSE 'Missed Playoffs'
            END AS "Final Result"
        FROM with_status
        """

        try:
            df_out = self.con.execute(q).fetchdf()
            years = sorted(df_out["Year"].unique())

            selected_year = st.selectbox(
                "Year", years, index=len(years)-1, key=f"{prefix}_year"
            )

            filtered = df_out[df_out["Year"] == selected_year]

            if "Seed" in filtered.columns:
                filtered = filtered.sort_values("Seed", ascending=True)

            # Style entire row for champion/sacko, otherwise just the Final Result cell
            def style_row(row):
                result = row['Final Result']
                n_cols = len(row)

                # Champion - highlight entire row gold
                if result == 'Won Championship':
                    return [f'background-color: {colors["gold"]}; color: {colors["text_dark"]}; font-weight: bold;'] * n_cols

                # Sacko - highlight entire row red
                elif result == 'Sacko':
                    return [f'background-color: {colors["sacko"]}; color: {colors["text_light"]}; font-weight: bold;'] * n_cols

                # For all other results, only style the Final Result column
                else:
                    styles = [''] * n_cols
                    result_idx = list(row.index).index('Final Result')

                    if result == 'Lost Championship':
                        styles[result_idx] = f'background-color: {colors["silver"]}; color: {colors["text_dark"]};'
                    elif result == 'Lost Semifinals':
                        styles[result_idx] = f'background-color: {colors["bronze"]}; color: {colors["text_light"]};'
                    elif result == 'Lost Quarterfinals':
                        styles[result_idx] = f'background-color: {colors["quarterfinal"]}; color: {colors["text_dark"] if not is_dark else colors["text_light"]};'
                    elif 'Won Third Place' in result:
                        styles[result_idx] = f'background-color: {colors["green_1"]}; color: {colors["text_dark"]};'
                    elif 'Won Fifth Place' in result:
                        styles[result_idx] = f'background-color: {colors["green_2"]}; color: {colors["text_dark"]};'
                    elif 'Won Seventh Place' in result:
                        styles[result_idx] = f'background-color: {colors["green_3"]}; color: {colors["text_dark"]};'
                    elif 'Won Ninth Place' in result:
                        styles[result_idx] = f'background-color: {colors["green_4"]}; color: {colors["text_dark"]};'
                    elif 'Lost Third Place' in result:
                        styles[result_idx] = f'background-color: {colors["gray_1"]}; color: {colors["text_dark"] if not is_dark else colors["text_light"]};'
                    elif 'Lost Fifth Place' in result:
                        styles[result_idx] = f'background-color: {colors["gray_2"]}; color: {colors["text_dark"] if not is_dark else colors["text_light"]};'
                    elif 'Lost Seventh Place' in result or 'Lost Ninth Place' in result:
                        styles[result_idx] = f'background-color: {colors["gray_3"]}; color: {colors["text_dark"] if not is_dark else colors["text_light"]};'
                    elif 'Consolation' in result:
                        styles[result_idx] = f'background-color: {colors["consol_semi"]}; color: {colors["text_dark"] if not is_dark else colors["text_light"]};'
                    elif result == 'Season in Progress':
                        styles[result_idx] = f'background-color: {colors["in_progress"]}; color: {colors["text_light"]};'
                    elif result == 'Missed Playoffs':
                        styles[result_idx] = f'background-color: {colors["missed"]}; color: {colors["text_light"]};'

                    return styles

            # Apply row styling
            styled_df = filtered.style.apply(
                style_row, axis=1
            ).format({
                'PF': '{:.1f}',
                'PA': '{:.1f}',
                'W': '{:.1f}' if per_game else '{:.0f}',
                'L': '{:.1f}' if per_game else '{:.0f}'
            })

            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Compact legend
            with st.expander("Legend", expanded=False):
                st.markdown(f"""
                    <div style='display: flex; flex-wrap: wrap; gap: 6px; font-size: 0.8rem;'>
                        <span style='background: {colors['gold']}; color: {colors['text_dark']}; padding: 2px 8px; border-radius: 4px;'>🏆 Champ</span>
                        <span style='background: {colors['silver']}; color: {colors['text_dark']}; padding: 2px 8px; border-radius: 4px;'>2nd</span>
                        <span style='background: {colors['bronze']}; color: {colors['text_light']}; padding: 2px 8px; border-radius: 4px;'>3rd/4th</span>
                        <span style='background: {colors['green_1']}; color: {colors['text_dark']}; padding: 2px 8px; border-radius: 4px;'>Consolation W</span>
                        <span style='background: {colors['gray_1']}; color: {colors['text_light']}; padding: 2px 8px; border-radius: 4px;'>Consolation L</span>
                        <span style='background: {colors['sacko']}; color: {colors['text_light']}; padding: 2px 8px; border-radius: 4px;'>Sacko</span>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error("Failed to build standings table from DuckDB.")
            st.exception(e)

@st.fragment
def display_season_standings(df, prefix: str = ""):
    viewer = SeasonStandingsViewer(df)
    viewer.display(prefix)
