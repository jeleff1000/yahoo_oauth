import streamlit as st
import pandas as pd
from ...shared.modern_styles import apply_modern_styles
from ..shared.theme import apply_theme_styles


class WeeklyMatchupStatsViewer:
    def __init__(self, df):
        self.df = df

    @st.fragment
    def display(self, prefix=""):
        """Display enhanced weekly matchup stats with improved UX."""
        apply_modern_styles()
        apply_theme_styles()

        # Header with description
        st.markdown(
            """
        <div class="tab-header">
        <h2>📊 Weekly Matchup Stats</h2>
        <p>Complete game-by-game results with scores, outcomes, and performance metrics</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if "win" not in self.df.columns:
            st.error("❌ The required column 'win' is not available in the data.")
            return

        # Prepare data
        display_df = self._prepare_display_data()

        if display_df.empty:
            st.info("No matchup data available with current filters")
            return

        # === ENHANCED TABLE DISPLAY ===
        self._render_enhanced_table(display_df, prefix)

        # === QUICK STATS SECTION (Below Table) ===
        st.markdown("---")
        stats = self._calculate_stats(display_df)
        self._render_quick_stats(stats)

        # === DOWNLOAD SECTION ===
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**💾 Export Data**")
        with col2:
            csv = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"weekly_matchup_stats_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
                use_container_width=True,
            )
        with col3:
            # Excel download would go here if needed
            pass

        # === INSIGHTS SECTION ===
        if len(display_df) > 0:
            self._render_insights(display_df)

    @staticmethod
    def _format_playoff_result(row) -> str:
        """Format playoff/consolation result combining round and outcome."""
        playoff_round = row["playoff_round"] if "playoff_round" in row.index else ""
        consolation_round = (
            row["consolation_round"] if "consolation_round" in row.index else ""
        )
        win = row["win"] if "win" in row.index else False

        # Determine which round to use
        round_name = ""
        if (
            playoff_round
            and pd.notna(playoff_round)
            and str(playoff_round).strip()
            and str(playoff_round).lower() != "none"
        ):
            round_name = str(playoff_round).strip()
        elif (
            consolation_round
            and pd.notna(consolation_round)
            and str(consolation_round).strip()
            and str(consolation_round).lower() != "none"
        ):
            round_name = str(consolation_round).strip()

        # If no playoff/consolation round, return empty
        if not round_name:
            return ""

        # Format the round name
        # Convert from snake_case to Title Case (e.g., "fifth_place_game" -> "Fifth Place Game")
        round_name = round_name.replace("_", " ").title()

        # Combine with outcome
        outcome = "Won" if win else "Lost"
        return f"{outcome} {round_name}"

    def _prepare_display_data(self) -> pd.DataFrame:
        """Prepare and format data for display."""
        df = self.df.copy()

        # Convert boolean columns
        df["win"] = df["win"] == 1
        df["is_playoffs"] = (
            df["is_playoffs"] == 1 if "is_playoffs" in df.columns else False
        )

        # Create combined playoff result column
        df["playoff_result"] = df.apply(self._format_playoff_result, axis=1)

        # Select columns to display
        columns_to_show = [
            "year",
            "week",
            "manager",
            "team_name",
            "opponent",
            "team_points",
            "opponent_points",
            "margin",
            "win",
            "playoff_result",  # Combined playoff/consolation round with outcome
            "weekly_rank",  # Weekly ranking (1-8)
        ]

        # Add optional columns if they exist
        optional_cols = {
            "league_weekly_median": "league_weekly_median",
            "above_league_median": "above_league_median",
        }

        for col_name, col_key in optional_cols.items():
            if col_key in df.columns:
                columns_to_show.append(col_key)

        # Filter to available columns
        available_cols = [col for col in columns_to_show if col in df.columns]
        display_df = df[available_cols].copy()

        # Ensure team_name exists
        if "team_name" not in display_df.columns:
            display_df["team_name"] = ""

        # Format and rename columns
        display_df = display_df.rename(
            columns={
                "year": "Year",
                "week": "Week",
                "manager": "Manager",
                "team_name": "Team",
                "opponent": "Opponent",
                "team_points": "PF",  # Points For
                "opponent_points": "PA",  # Points Against
                "margin": "Margin",
                "win": "Result",
                "playoff_result": "Playoff Result",
                "weekly_rank": "Rank",
                "league_weekly_median": "League Avg",
                "above_league_median": "Above Avg",
            }
        )

        # Format numeric columns
        numeric_cols = ["PF", "PA", "Margin", "League Avg"]
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce")

        # Sort by most recent first
        display_df["Year"] = display_df["Year"].astype(int)
        display_df["Week"] = display_df["Week"].astype(int)
        display_df = display_df.sort_values(
            by=["Year", "Week"], ascending=[False, False]
        ).reset_index(drop=True)

        return display_df

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        total_games = len(df)
        if total_games == 0:
            return {}

        wins = df["Result"].sum()
        losses = total_games - wins
        win_pct = (wins / total_games * 100) if total_games > 0 else 0

        avg_pf = df["PF"].mean()
        avg_pa = df["PA"].mean()
        avg_margin = df["Margin"].mean()

        max_score = df["PF"].max()
        min_score = df["PF"].min()

        # Winning/Losing streaks
        current_streak = self._calculate_current_streak(df)

        # Best/worst performances
        best_win = df[df["Result"]].nlargest(1, "Margin") if wins > 0 else None
        worst_loss = (
            df[not df["Result"]].nsmallest(1, "Margin") if losses > 0 else None
        )

        return {
            "total_games": total_games,
            "wins": int(wins),
            "losses": int(losses),
            "win_pct": win_pct,
            "avg_pf": avg_pf,
            "avg_pa": avg_pa,
            "avg_margin": avg_margin,
            "max_score": max_score,
            "min_score": min_score,
            "current_streak": current_streak,
            "best_win": best_win,
            "worst_loss": worst_loss,
        }

    def _calculate_current_streak(self, df: pd.DataFrame) -> dict:
        """Calculate current winning/losing streak."""
        if df.empty:
            return {"type": None, "count": 0}

        # Sort by most recent
        sorted_df = df.sort_values(["Year", "Week"], ascending=[False, False])

        streak_type = "W" if sorted_df.iloc[0]["Result"] else "L"
        streak_count = 1

        for i in range(1, len(sorted_df)):
            current_result = sorted_df.iloc[i]["Result"]
            if (streak_type == "W" and current_result) or (
                streak_type == "L" and not current_result
            ):
                streak_count += 1
            else:
                break

        return {"type": streak_type, "count": streak_count}

    def _render_quick_stats(self, stats: dict):
        """Render quick statistics cards at the top."""
        if not stats:
            return

        st.markdown("### 📈 Quick Stats")

        # Row 1: Core metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Games",
                f"{stats['total_games']:,}",
                help="Total number of matchups",
            )

        with col2:
            record_str = f"{stats['wins']}-{stats['losses']}"
            "normal" if stats["win_pct"] >= 50 else "inverse"
            st.metric(
                "Record",
                record_str,
                delta=f"{stats['win_pct']:.1f}%",
                help="Win-Loss record and win percentage",
            )

        with col3:
            margin_delta = "+" if stats["avg_margin"] > 0 else "-"
            st.metric(
                "Avg Margin",
                f"{stats['avg_margin']:.2f}",
                delta=margin_delta,
                help="Average point differential per game",
            )

        with col4:
            streak = stats.get("current_streak", {})
            if streak.get("type"):
                streak_emoji = "🔥" if streak["type"] == "W" else "❄️"
                streak_text = f"{streak['count']}{streak['type']}"
                st.metric(
                    "Current Streak",
                    streak_text,
                    delta=streak_emoji,
                    help=f"Current {'winning' if streak['type'] == 'W' else 'losing'} streak",
                )
            else:
                st.metric("Current Streak", "N/A")

        # Row 2: Scoring metrics
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "Avg PF",
                f"{stats['avg_pf']:.2f}",
                help="Average points scored per game",
            )

        with col6:
            st.metric(
                "Avg PA",
                f"{stats['avg_pa']:.2f}",
                help="Average points allowed per game",
            )

        with col7:
            st.metric(
                "High Score",
                f"{stats['max_score']:.2f}",
                help="Highest single-game score",
            )

        with col8:
            st.metric(
                "Low Score",
                f"{stats['min_score']:.2f}",
                help="Lowest single-game score",
            )

    def _render_enhanced_table(self, df: pd.DataFrame, prefix: str):
        """Render enhanced table with column configuration."""

        # Create display dataframe with formatted values
        display_df = df.copy()

        # Add visual indicators for Result column - using quieter indicators
        display_df["Result"] = display_df["Result"].apply(lambda x: "W" if x else "L")

        # Format playoff result column with emoji
        if "Playoff Result" in display_df.columns:
            display_df["Playoff Result"] = display_df["Playoff Result"].apply(
                lambda x: f"🏆 {x}" if x else ""
            )

        # Configure column display
        column_config = {
            "Year": st.column_config.NumberColumn(
                "Year", help="Season year", format="%d", width="small"
            ),
            "Week": st.column_config.NumberColumn(
                "Week", help="Week number", format="%d", width="small"
            ),
            "Manager": st.column_config.TextColumn(
                "Manager", help="Manager name", width="medium"
            ),
            "Team": st.column_config.TextColumn(
                "Team", help="Team name", width="medium"
            ),
            "Opponent": st.column_config.TextColumn(
                "Opponent", help="Opponent name", width="medium"
            ),
            "PF": st.column_config.NumberColumn(
                "PF", help="Points For (scored)", format="%.2f", width="small"
            ),
            "PA": st.column_config.NumberColumn(
                "PA", help="Points Against (allowed)", format="%.2f", width="small"
            ),
            "Margin": st.column_config.NumberColumn(
                "Margin",
                help="Point differential (PF - PA)",
                format="%.2f",
                width="small",
            ),
            "Result": st.column_config.TextColumn(
                "Result", help="Game outcome", width="small"
            ),
            "Playoff Result": st.column_config.TextColumn(
                "Playoff Result",
                help='Playoff/consolation game result (e.g., "Won Championship", "Lost 7th Place Game")',
                width="medium",
            ),
            "Rank": st.column_config.NumberColumn(
                "Rank",
                help="Weekly league ranking (1 = best)",
                format="%d",
                width="small",
            ),
        }

        # Add configurations for optional columns

        if "League Avg" in display_df.columns:
            column_config["League Avg"] = st.column_config.NumberColumn(
                "League Avg",
                help="League average score this week",
                format="%.2f",
                width="small",
            )

        if "Above Avg" in display_df.columns:
            display_df["Above Avg"] = display_df["Above Avg"].apply(
                lambda x: "Yes" if x else "No"
            )
            column_config["Above Avg"] = st.column_config.TextColumn(
                "Above Avg", help="Scored above league average", width="small"
            )

        # Display the enhanced dataframe
        st.dataframe(
            display_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            height=500,  # Fixed height for better scrolling
        )

    def _render_insights(self, df: pd.DataFrame):
        """Render data insights and highlights."""
        st.markdown("---")
        st.markdown("### 💡 Insights & Highlights")

        col1, col2 = st.columns(2)

        with col1:
            # Best performance
            if len(df[df["Result"] == "W"]) > 0:
                wins_df = df[df["Result"] == "W"].copy()
                wins_df["Margin_num"] = df[df["Result"] == "W"]["Margin"]
                best_win = wins_df.nlargest(1, "Margin_num").iloc[0]

                st.markdown(
                    """
                <div class="theme-success">
                <strong>Best Win</strong><br>
                Week {week}, {year}: <strong>{manager}</strong> defeated {opponent}<br>
                Score: {pf:.2f} - {pa:.2f} (Margin: +{margin:.2f})
                </div>
                """.format(
                        week=int(best_win["Week"]),
                        year=int(best_win["Year"]),
                        manager=best_win["Manager"],
                        opponent=best_win["Opponent"],
                        pf=best_win["PF"],
                        pa=best_win["PA"],
                        margin=best_win["Margin_num"],
                    ),
                    unsafe_allow_html=True,
                )

        with col2:
            # Worst performance
            if len(df[df["Result"] == "L"]) > 0:
                losses_df = df[df["Result"] == "L"].copy()
                losses_df["Margin_num"] = df[df["Result"] == "L"]["Margin"]
                worst_loss = losses_df.nsmallest(1, "Margin_num").iloc[0]

                st.markdown(
                    """
                <div class="theme-warning">
                <strong>Toughest Loss</strong><br>
                Week {week}, {year}: <strong>{manager}</strong> lost to {opponent}<br>
                Score: {pf:.2f} - {pa:.2f} (Margin: {margin:.2f})
                </div>
                """.format(
                        week=int(worst_loss["Week"]),
                        year=int(worst_loss["Year"]),
                        manager=worst_loss["Manager"],
                        opponent=worst_loss["Opponent"],
                        pf=worst_loss["PF"],
                        pa=worst_loss["PA"],
                        margin=worst_loss["Margin_num"],
                    ),
                    unsafe_allow_html=True,
                )

        # Additional insights
        st.markdown("<br>", unsafe_allow_html=True)

        insight_col1, insight_col2, insight_col3 = st.columns(3)

        with insight_col1:
            # Playoff record
            if "Playoff Result" in df.columns:
                playoff_games = df[df["Playoff Result"].str.len() > 0]
                if len(playoff_games) > 0:
                    playoff_df_wins = playoff_games[playoff_games["Result"] == "W"]
                    playoff_wins = len(playoff_df_wins)
                    playoff_total = len(playoff_games)
                    st.info(
                        f"**Playoff Record:** {playoff_wins}-{playoff_total - playoff_wins}"
                    )

        with insight_col2:
            # Close games (within 10 points)
            close_games = df[abs(df["Margin"]) <= 10]
            if len(close_games) > 0:
                close_wins = len(close_games[close_games["Result"] == "W"])
                st.info(
                    f"**Close Games (<10 pts):** {close_wins}-{len(close_games) - close_wins}"
                )

        with insight_col3:
            # High scoring games (>120 points)
            if df["PF"].max() > 0:
                high_score_threshold = df["PF"].quantile(0.75)
                high_scoring = len(df[df["PF"] >= high_score_threshold])
                st.info(f"**Top 25% Scores:** {high_scoring} games")
