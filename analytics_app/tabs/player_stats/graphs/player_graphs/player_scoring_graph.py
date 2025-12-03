#!/usr/bin/env python3
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from md.core import list_player_seasons
from md.tab_data_access.players import load_players_season_data


@st.fragment
def display_player_scoring_graphs(prefix=""):
    """
    Player scoring trends over time - optimized to use data_access layer.
    Shows weekly points within a season or yearly averages across seasons.
    """
    st.header("📈 Player Scoring Trends")

    st.markdown(
        """
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="margin: 0; color: #31333F; font-size: 0.9rem;">
    <strong>Track player performance:</strong> Search for players to see their scoring trends. 
    Single season view shows weekly points + cumulative average. Multi-season view shows yearly averages.
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Year selection
    available_years = list_player_seasons()  # FIXED: was list_player_season_data()
    if not available_years:
        st.error("No player data found.")
        return

    min_year, max_year = min(available_years), max(available_years)

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "Start Year",
            min_value=min_year,
            max_value=max_year,
            value=max_year,
            key=f"{prefix}_start_year",
        )
    with col2:
        end_year = st.number_input(
            "End Year",
            min_value=min_year,
            max_value=max_year,
            value=max_year,
            key=f"{prefix}_end_year",
        )

    # Position filter
    position_order = ["QB", "RB", "WR", "TE", "K", "DEF"]
    with st.expander("Filter by Position (Optional)", expanded=False):
        selected_positions = st.multiselect(
            "Select positions to include",
            options=position_order,
            default=position_order,
            key=f"{prefix}_positions",
        )

    # Roster filter option
    roster_filter = st.checkbox(
        "Show only rostered players (limits to manager history starting 2014)",
        value=False,
        key=f"{prefix}_rostered_only",
        help="Unchecked = All players back to 1999. Checked = Only players with managers (2014+)",
    )

    # Player search
    player_search = st.text_input(
        "🔍 Enter player names (comma separated):",
        value="",
        placeholder="e.g., Patrick Mahomes, Justin Jefferson",
        key=f"{prefix}_player_search",
    ).strip()

    if not player_search:
        st.info("💡 Enter player name(s) above to display scoring trends.")
        return

    # Parse player names
    search_names = [name.strip() for name in player_search.split(",") if name.strip()]

    with st.spinner("Loading player data..."):
        # Determine which years to load
        years_to_load = [y for y in range(int(start_year), int(end_year) + 1)]

        # Load data efficiently - load all needed years at once
        df = load_players_season_data(
            year=years_to_load,
            rostered_only=roster_filter,  # Use the checkbox value
            sort_column="points",
            sort_direction="DESC",
        )

        if df.empty:
            st.warning("No data found for the selected years.")
            return

        player_data = df

        # Filter by position if selected
        if selected_positions and "nfl_position" in player_data.columns:
            player_data = player_data[
                player_data["nfl_position"].isin(selected_positions)
            ]

        # Filter by player names (case insensitive partial match)
        player_data["player_lower"] = player_data["player"].str.lower()
        search_lower = [n.lower() for n in search_names]

        filtered = player_data[
            player_data["player_lower"].apply(
                lambda x: any(search in x for search in search_lower)
            )
        ]
        filtered = filtered.drop(columns=["player_lower"])

        if filtered.empty:
            st.warning(f"No players found matching: {', '.join(search_names)}")
            st.info("Try different spellings or check the year range.")
            return

    # Display results
    st.success(f"✅ Found {len(filtered['player'].unique())} player(s)")

    # Create visualization
    fig = go.Figure()

    if start_year == end_year:
        # Single season - show weekly breakdown if available
        st.subheader(f"Weekly Performance ({start_year})")
        st.caption("Shows points per game and cumulative season average")

        for player_name in filtered["player"].unique():
            player_df = filtered[filtered["player"] == player_name].copy()

            # For season view, we have aggregated data - show PPG
            if (
                "season_ppg" in player_df.columns
                and not player_df["season_ppg"].isna().all()
            ):
                ppg = player_df["season_ppg"].iloc[0]
                games = (
                    player_df["fantasy_games"].iloc[0]
                    if "fantasy_games" in player_df.columns
                    else 0
                )

                # Create a simple display
                st.metric(
                    label=player_name, value=f"{ppg:.2f} PPG", delta=f"{games} games"
                )

        st.info(
            "💡 Weekly breakdown requires weekly data access. Showing season aggregates."
        )

    else:
        # Multi-season view
        st.subheader("Year-over-Year Performance")

        for player_name in filtered["player"].unique():
            player_df = filtered[filtered["player"] == player_name].copy()
            player_df = player_df.sort_values("year")

            # Calculate cumulative average
            player_df["cumulative_avg"] = player_df["season_ppg"].expanding().mean()

            # Add yearly PPG
            fig.add_trace(
                go.Scatter(
                    x=player_df["year"],
                    y=player_df["season_ppg"],
                    mode="markers+lines",
                    name=f"{player_name} PPG",
                    marker=dict(size=10),
                    line=dict(width=2),
                    hovertemplate=f"<b>{player_name}</b><br>Year: %{{x}}<br>PPG: %{{y:.2f}}<br><extra></extra>",
                )
            )

            # Add cumulative average
            fig.add_trace(
                go.Scatter(
                    x=player_df["year"],
                    y=player_df["cumulative_avg"],
                    mode="lines",
                    name=f"{player_name} Cumulative Avg",
                    line=dict(dash="dash", width=2),
                    hovertemplate=f"<b>{player_name}</b><br>Year: %{{x}}<br>Career Avg: %{{y:.2f}}<br><extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="Season",
            yaxis_title="Points Per Game",
            hovermode="x unified",
            showlegend=True,
            height=500,
            template="plotly_white",
        )
        fig.update_xaxes(tickmode="linear", dtick=1, showgrid=True)
        fig.update_yaxes(showgrid=True)

        st.plotly_chart(
            fig, use_container_width=True, key=f"{prefix}_player_scoring_multi_year"
        )

        # Summary table
        with st.expander("📊 Detailed Stats", expanded=False):
            summary_cols = [
                "player",
                "year",
                "season_ppg",
                "points",
                "fantasy_games",
                "games_started",
            ]
            available_cols = [c for c in summary_cols if c in filtered.columns]
            summary_df = filtered[available_cols].sort_values(
                ["player", "year"], ascending=[True, False]
            )
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
