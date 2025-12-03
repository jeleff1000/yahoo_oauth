"""
Playoff Brackets Viewer - Modern, Responsive Design

Automatically adapts to any playoff structure.
Features seed-based coloring, responsive layout, and enhanced readability.
"""

import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple


class PlayoffBracketsViewer:
    """Modern playoff bracket visualization with seed display and responsive design."""

    # Round display names
    ROUND_NAMES = {
        "wildcard": "Wild Card",
        "quarterfinal": "Quarterfinals",
        "semifinal": "Semifinals",
        "championship": "Championship",
        "final": "Final",
    }

    # Seed-based color scheme (gradient from dark green to light)
    SEED_COLORS = {
        1: "#1B5E20",  # Dark green (top seed)
        2: "#2E7D32",
        3: "#388E3C",
        4: "#4CAF50",
        5: "#66BB6A",
        6: "#81C784",
        7: "#A5D6A7",
        8: "#C8E6C9",
    }

    # Visual configuration
    COLORS = {
        "winner_bg": "rgba(76, 175, 80, 0.15)",
        "loser_bg": "rgba(250, 250, 250, 0.95)",
        "bye_bg": "rgba(255, 193, 7, 0.12)",
        "line": "#424242",
        "bye_line": "#FFA726",
        "champion": "#FFD700",
        "background": "#FAFAFA",
        "text_primary": "#212121",
        "text_secondary": "#757575",
        "seed_bg": "rgba(255, 255, 255, 0.95)",
        "border": "#E0E0E0",
    }

    SPACING = {
        "x_gap": 6,  # Increased for more space between rounds
        "box_width": 2.8,
        "box_height": 1.1,
        "vertical_gap": 3.5,  # Increased vertical spacing
        "bye_offset": -3.5,  # Moved bye indicators further left
        "seed_radius": 0.35,
        "champion_offset": 1.5,  # Space between final match and champion box
    }

    def __init__(self, df: pd.DataFrame):
        """
        Initialize bracket viewer.

        Args:
            df: DataFrame with matchup data including playoff_round and final_playoff_seed
        """
        self.df = df

    def _get_seed_color(self, seed: Optional[int]) -> str:
        """Get color for a seed number."""
        if seed is None or pd.isna(seed):
            return self.COLORS["text_secondary"]
        seed_int = int(seed)
        return self.SEED_COLORS.get(seed_int, self.COLORS["text_secondary"])

    def _extract_matches_from_round(self, round_df: pd.DataFrame) -> List[Dict]:
        """
        Extract unique matchups from round data including seeds.

        Args:
            round_df: DataFrame filtered to single playoff round

        Returns:
            List of match dicts with team info, scores, seeds, and winner
        """
        matches = []
        seen_matchups = set()

        for _, row in round_df.iterrows():
            manager = str(row.get("manager", ""))
            opponent = str(row.get("opponent", ""))

            matchup_key = tuple(sorted([manager, opponent]))

            if matchup_key in seen_matchups:
                continue

            seen_matchups.add(matchup_key)

            # Get both teams' data
            manager_row = round_df[round_df["manager"] == manager].iloc[0]
            opponent_rows = round_df[round_df["manager"] == opponent]
            opponent_row = opponent_rows.iloc[0] if not opponent_rows.empty else None

            pts_a = (
                float(manager_row.get("team_points", 0))
                if pd.notna(manager_row.get("team_points"))
                else 0.0
            )
            pts_b = (
                float(opponent_row.get("team_points", 0))
                if opponent_row is not None
                and pd.notna(opponent_row.get("team_points"))
                else float(row.get("opponent_points", 0))
            )

            seed_a = manager_row.get("final_playoff_seed")
            seed_b = (
                opponent_row.get("final_playoff_seed")
                if opponent_row is not None
                else None
            )

            # Convert seeds to int
            seed_a = int(seed_a) if pd.notna(seed_a) else None
            seed_b = int(seed_b) if pd.notna(seed_b) else None

            # Determine winner
            winner = manager if pts_a > pts_b else opponent
            winner_pts = pts_a if pts_a > pts_b else pts_b

            matches.append(
                {
                    "teamA": manager,
                    "teamB": opponent,
                    "ptsA": pts_a,
                    "ptsB": pts_b,
                    "seedA": seed_a,
                    "seedB": seed_b,
                    "winner": winner,
                    "winner_pts": winner_pts,
                }
            )

        # Sort by seed (lower seed = better position)
        matches.sort(
            key=lambda m: (
                m["seedA"] if m["seedA"] else 99,
                m["seedB"] if m["seedB"] else 99,
            )
        )

        return matches

    def _detect_bye_teams(
        self, rounds_data: List[Tuple[str, List[Dict]]]
    ) -> Dict[str, int]:
        """
        Detect teams that received first-round byes with their seeds.

        Args:
            rounds_data: List of (round_name, matches) tuples

        Returns:
            Dict mapping team name to seed
        """
        if len(rounds_data) < 2:
            return {}

        # Get teams from first two rounds
        first_round_teams = set()
        for match in rounds_data[0][1]:
            first_round_teams.add(match["teamA"])
            first_round_teams.add(match["teamB"])

        # Get bye teams with their seeds
        bye_teams = {}
        for match in rounds_data[1][1]:
            for team_key in ["teamA", "teamB"]:
                team = match[team_key]
                if team not in first_round_teams:
                    seed_key = "seedA" if team_key == "teamA" else "seedB"
                    bye_teams[team] = match[seed_key]

        return bye_teams

    def _build_bracket_structure(
        self, rounds_data: List[Tuple[str, List[Dict]]]
    ) -> Dict:
        """Build bracket structure tracking match flows."""
        structure = {"flows": {}}

        for r_idx in range(len(rounds_data) - 1):
            current_matches = rounds_data[r_idx][1]
            next_matches = rounds_data[r_idx + 1][1]

            flows = {}

            for curr_idx, curr_match in enumerate(current_matches):
                winner = curr_match["winner"]

                for next_idx, next_match in enumerate(next_matches):
                    if winner in (next_match["teamA"], next_match["teamB"]):
                        flows[curr_idx] = next_idx
                        break

            structure["flows"][r_idx] = flows

        return structure

    def _calculate_y_positions(
        self, rounds_data: List[Tuple[str, List[Dict]]], bracket_structure: Dict
    ) -> List[List[float]]:
        """Calculate Y positions for all matches."""
        y_positions = []
        v_gap = self.SPACING["vertical_gap"]

        # First round: evenly space matches
        first_count = len(rounds_data[0][1])
        first_y = [(i - (first_count - 1) / 2) * v_gap * 2 for i in range(first_count)]
        y_positions.append(first_y)

        # Subsequent rounds: position based on feeding matches
        for r_idx in range(1, len(rounds_data)):
            current_matches = rounds_data[r_idx][1]
            current_y = []

            if r_idx - 1 in bracket_structure["flows"]:
                flows = bracket_structure["flows"][r_idx - 1]
                prev_y = y_positions[r_idx - 1]

                for next_idx in range(len(current_matches)):
                    feeding_matches = [
                        curr_idx
                        for curr_idx, target_idx in flows.items()
                        if target_idx == next_idx
                    ]

                    if feeding_matches:
                        feeding_y_values = [
                            prev_y[idx] for idx in feeding_matches if idx < len(prev_y)
                        ]
                        current_y.append(sum(feeding_y_values) / len(feeding_y_values))
                    else:
                        current_y.append(
                            (next_idx - (len(current_matches) - 1) / 2) * v_gap * 2
                        )
            else:
                for i in range(len(current_matches)):
                    current_y.append((i - (len(current_matches) - 1) / 2) * v_gap * 2)

            y_positions.append(current_y)

        return y_positions

    def _draw_seed_badge(self, fig: go.Figure, x: float, y: float, seed: Optional[int]):
        """
        Draw a circular seed badge.

        Args:
            fig: Plotly figure
            x: X position
            y: Y position
            seed: Seed number
        """
        if seed is None:
            return

        radius = self.SPACING["seed_radius"]
        seed_color = self._get_seed_color(seed)

        # Draw circle
        fig.add_shape(
            type="circle",
            x0=x - radius,
            y0=y - radius,
            x1=x + radius,
            y1=y + radius,
            fillcolor=seed_color,
            line=dict(color="white", width=2),
        )

        # Draw seed number
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{seed}</b>",
            showarrow=False,
            font=dict(size=13, color="white", family="Arial Black"),
        )

    def _draw_match_box(self, fig: go.Figure, x: float, y_center: float, match: Dict):
        """
        Draw a modern match box with seeds and enhanced styling.

        Args:
            fig: Plotly figure
            x: X position (center)
            y_center: Y position (center)
            match: Match dict with team data
        """
        box_w = self.SPACING["box_width"]
        box_h = self.SPACING["box_height"]

        left = x - box_w / 2
        right = x + box_w / 2
        top = y_center + box_h
        bottom = y_center - box_h
        mid = y_center

        team_a, team_b = match["teamA"], match["teamB"]
        pts_a, pts_b = match["ptsA"], match["ptsB"]
        seed_a, seed_b = match["seedA"], match["seedB"]
        winner = match["winner"]

        # Determine colors based on winner
        a_bg = self.COLORS["winner_bg"] if winner == team_a else self.COLORS["loser_bg"]
        b_bg = self.COLORS["winner_bg"] if winner == team_b else self.COLORS["loser_bg"]

        # Draw team boxes with rounded corners (via border)
        for team_bg, y0, y1 in [(a_bg, mid, top), (b_bg, bottom, mid)]:
            fig.add_shape(
                type="rect",
                x0=left,
                y0=y0,
                x1=right,
                y1=y1,
                line=dict(color=self.COLORS["border"], width=2),
                fillcolor=team_bg,
            )

        # Draw seed badges
        seed_x = left + 0.4
        self._draw_seed_badge(fig, seed_x, (mid + top) / 2, seed_a)
        self._draw_seed_badge(fig, seed_x, (bottom + mid) / 2, seed_b)

        # Format team names and scores
        pts_a_str = f"{pts_a:.1f}" if pts_a else "0.0"
        pts_b_str = f"{pts_b:.1f}" if pts_b else "0.0"

        # Add team names (with emphasis on winner)
        team_a_style = (
            "font-weight: 700; font-size: 13px;"
            if winner == team_a
            else "font-size: 12px;"
        )
        team_b_style = (
            "font-weight: 700; font-size: 13px;"
            if winner == team_b
            else "font-size: 12px;"
        )

        fig.add_annotation(
            x=seed_x + 0.5,
            y=(mid + top) / 2,
            text=f"<span style='{team_a_style}'>{team_a}</span>",
            showarrow=False,
            xanchor="left",
            font=dict(color=self.COLORS["text_primary"]),
        )

        fig.add_annotation(
            x=seed_x + 0.5,
            y=(bottom + mid) / 2,
            text=f"<span style='{team_b_style}'>{team_b}</span>",
            showarrow=False,
            xanchor="left",
            font=dict(color=self.COLORS["text_primary"]),
        )

        # Add scores (right-aligned)
        score_x = right - 0.15
        score_a_style = (
            "font-weight: 700; font-size: 13px;"
            if winner == team_a
            else "font-size: 12px; color: #9E9E9E;"
        )
        score_b_style = (
            "font-weight: 700; font-size: 13px;"
            if winner == team_b
            else "font-size: 12px; color: #9E9E9E;"
        )

        fig.add_annotation(
            x=score_x,
            y=(mid + top) / 2,
            text=f"<span style='{score_a_style}'>{pts_a_str}</span>",
            showarrow=False,
            xanchor="right",
            font=dict(color=self.COLORS["text_primary"]),
        )

        fig.add_annotation(
            x=score_x,
            y=(bottom + mid) / 2,
            text=f"<span style='{score_b_style}'>{pts_b_str}</span>",
            showarrow=False,
            xanchor="right",
            font=dict(color=self.COLORS["text_primary"]),
        )

    def _draw_connector_lines(
        self,
        fig: go.Figure,
        x: float,
        y: float,
        next_x: float,
        next_y: float,
        paired_y: Optional[float] = None,
    ):
        """Draw smooth connector lines between rounds."""
        box_w = self.SPACING["box_width"]
        line_width = 2.5

        # Horizontal from current match
        fig.add_shape(
            type="line",
            x0=x + box_w / 2,
            y0=y,
            x1=x + box_w / 2 + 0.6,
            y1=y,
            line=dict(color=self.COLORS["line"], width=line_width),
        )

        # Vertical connector if paired
        if paired_y is not None:
            fig.add_shape(
                type="line",
                x0=x + box_w / 2 + 0.6,
                y0=y,
                x1=x + box_w / 2 + 0.6,
                y1=paired_y,
                line=dict(color=self.COLORS["line"], width=line_width),
            )

        # Horizontal to next round
        fig.add_shape(
            type="line",
            x0=x + box_w / 2 + 0.6,
            y0=next_y,
            x1=next_x - box_w / 2,
            y1=next_y,
            line=dict(color=self.COLORS["line"], width=line_width),
        )

    def create_bracket(self, year: int) -> go.Figure:
        """
        Create modern playoff bracket visualization.

        Args:
            year: Season year

        Returns:
            Plotly Figure with enhanced bracket
        """
        # Filter to playoff games
        year_df = self.df[
            (self.df["year"] == year)
            & (self.df["is_playoffs"] == 1)
            & (self.df["is_consolation"] == 0)
        ].copy()

        if year_df.empty or "playoff_round" not in year_df.columns:
            return go.Figure()

        # Extract rounds
        round_order = (
            year_df.groupby("playoff_round")["week"].min().sort_values().index.tolist()
        )

        # Build matches by round
        rounds_data = []

        for round_name in round_order:
            round_df = year_df[year_df["playoff_round"] == round_name]
            matches = self._extract_matches_from_round(round_df)

            if matches:
                display_name = self.ROUND_NAMES.get(round_name, round_name.title())
                rounds_data.append((display_name, matches))

        if not rounds_data:
            return go.Figure()

        # Detect bye teams
        bye_teams = self._detect_bye_teams(rounds_data)

        # Build structure and positions
        bracket_structure = self._build_bracket_structure(rounds_data)
        y_positions = self._calculate_y_positions(rounds_data, bracket_structure)
        x_positions = [i * self.SPACING["x_gap"] for i in range(len(rounds_data))]

        # Create figure
        fig = go.Figure()

        # Draw bye indicators
        if bye_teams and len(rounds_data) >= 2:
            for bye_team, bye_seed in bye_teams.items():
                for match_idx, match in enumerate(rounds_data[1][1]):
                    if bye_team in (match["teamA"], match["teamB"]):
                        x_bye = x_positions[0] + self.SPACING["bye_offset"]
                        y_bye = y_positions[1][match_idx]

                        # Bye box with better sizing
                        fig.add_shape(
                            type="rect",
                            x0=x_bye - 1.2,
                            y0=y_bye - 0.6,
                            x1=x_bye + 1.2,
                            y1=y_bye + 0.6,
                            fillcolor=self.COLORS["bye_bg"],
                            line=dict(color=self.COLORS["bye_line"], width=2),
                        )

                        # Seed badge for bye team (moved left)
                        self._draw_seed_badge(fig, x_bye - 0.8, y_bye, bye_seed)

                        # Team name only (no "First Round Bye" text to avoid overlap)
                        fig.add_annotation(
                            x=x_bye - 0.3,
                            y=y_bye + 0.15,
                            text=f"<b>{bye_team}</b>",
                            showarrow=False,
                            font=dict(size=12, color="#E65100"),
                            xanchor="left",
                        )

                        # "BYE" label below
                        fig.add_annotation(
                            x=x_bye,
                            y=y_bye - 0.25,
                            text="<span style='font-size:10px; color: #F57C00;'>Bye</span>",
                            showarrow=False,
                            font=dict(size=10, color="#F57C00"),
                            xanchor="center",
                        )

                        # Dashed line to semifinal
                        fig.add_shape(
                            type="line",
                            x0=x_bye + 1.2,
                            y0=y_bye,
                            x1=x_positions[1] - self.SPACING["box_width"] / 2,
                            y1=y_bye,
                            line=dict(
                                color=self.COLORS["bye_line"], width=2.5, dash="dot"
                            ),
                        )

        # Draw all rounds
        for r_idx, ((round_name, matches), x, ys) in enumerate(
            zip(rounds_data, x_positions, y_positions)
        ):
            # Round label
            fig.add_annotation(
                x=x,
                y=max(ys) + self.SPACING["vertical_gap"] * 1.5,
                text=f"<b style='font-size: 18px;'>{round_name}</b>",
                showarrow=False,
                font=dict(
                    size=18, color=self.COLORS["text_primary"], family="Arial Black"
                ),
            )

            # Draw each match
            for match_idx, (match, y) in enumerate(zip(matches, ys)):
                self._draw_match_box(fig, x, y, match)

                # Draw connector lines to next round
                if r_idx < len(rounds_data) - 1:
                    flows = bracket_structure["flows"].get(r_idx, {})

                    if match_idx in flows:
                        next_match_idx = flows[match_idx]
                        next_x = x_positions[r_idx + 1]
                        next_y = y_positions[r_idx + 1][next_match_idx]

                        # Find paired match
                        paired_y = None
                        for other_idx, other_target in flows.items():
                            if (
                                other_target == next_match_idx
                                and other_idx != match_idx
                            ):
                                if other_idx < len(ys):
                                    paired_y = ys[other_idx]
                                break

                        self._draw_connector_lines(fig, x, y, next_x, next_y, paired_y)

        # Champion annotation (moved further right to avoid overlap)
        if rounds_data:
            final_match = rounds_data[-1][1][0]
            champion = final_match["winner"]
            champ_pts = final_match["winner_pts"]
            champ_seed = (
                final_match["seedA"]
                if champion == final_match["teamA"]
                else final_match["seedB"]
            )

            final_x = x_positions[-1]
            final_y = y_positions[-1][0]

            # Champion box position (moved further right)
            champ_x_start = (
                final_x
                + self.SPACING["box_width"] / 2
                + self.SPACING["champion_offset"]
            )

            # Champion trophy box
            fig.add_shape(
                type="rect",
                x0=champ_x_start,
                y0=final_y - 0.9,
                x1=champ_x_start + 2.8,
                y1=final_y + 0.9,
                fillcolor="rgba(255, 215, 0, 0.15)",
                line=dict(color=self.COLORS["champion"], width=3),
            )

            # Champion seed badge
            self._draw_seed_badge(fig, champ_x_start + 0.5, final_y, champ_seed)

            # Champion text
            fig.add_annotation(
                x=champ_x_start + 1.4,
                y=final_y + 0.25,
                text=f"<b style='font-size: 16px;'>🏆 {champion}</b>",
                showarrow=False,
                font=dict(size=16, color="#F57F17", family="Arial Black"),
            )

            fig.add_annotation(
                x=champ_x_start + 1.4,
                y=final_y - 0.35,
                text=f"<span style='font-size: 13px;'>{champ_pts:.1f} points</span>",
                showarrow=False,
                font=dict(size=13, color=self.COLORS["text_secondary"]),
            )

        # Configure layout
        all_y = [y for round_y in y_positions for y in round_y]
        y_range = (
            [
                min(all_y) - self.SPACING["vertical_gap"] * 2,
                max(all_y) + self.SPACING["vertical_gap"] * 2.5,
            ]
            if all_y
            else [-10, 10]
        )

        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[
                    -5,
                    x_positions[-1] + 6,
                ],  # Extended range for bye indicators and champion box
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=y_range,
                scaleanchor=None,
            ),
            plot_bgcolor=self.COLORS["background"],
            showlegend=False,
            margin=dict(l=20, r=20, t=80, b=20),
            height=(
                max(
                    700,
                    int(
                        (max(all_y) - min(all_y) + self.SPACING["vertical_gap"] * 5)
                        * 55
                    ),
                )
                if all_y
                else 700
            ),
            width=None,  # Auto-width for responsiveness
            font=dict(family="Arial, sans-serif"),
        )

        return fig

    @st.fragment
    def display(self):
        """Display playoff brackets with modern UI and mobile responsiveness."""
        st.markdown(
            """
            <div class='hof-gradient-header hof-header-gold'>
                <h2>🏆 Playoff Brackets</h2>
                <p>Interactive tournament bracket with playoff seeding</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if self.df is None or self.df.empty:
            st.info("📊 No playoff data available")
            return

        # Get playoff years
        playoff_df = self.df[
            (self.df["is_playoffs"] == 1) & (self.df["is_consolation"] == 0)
        ]

        if playoff_df.empty:
            st.warning("No playoff data found in dataset")
            return

        years = sorted(playoff_df["year"].unique(), reverse=True)

        # Year selector
        selected_year = st.selectbox(
            "Select Season",
            years,
            key="bracket_year",
            help="Choose a playoff season to view the bracket",
        )

        playoff_df[playoff_df["year"] == selected_year]

        # Create bracket
        fig = self.create_bracket(selected_year)

        if not fig.data and not fig.layout.shapes:
            st.warning(f"No playoff bracket data available for {selected_year}")
            return

        # Mobile-responsive scrollable container with hint
        st.markdown(
            """
            <div class='bracket-scroll-hint'>
                <span>👆 Swipe or scroll horizontally to see full bracket</span>
            </div>
            <div class='bracket-container'>
        """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("</div>", unsafe_allow_html=True)

        # Seed legend (simplified)
        with st.expander("🎯 Seed Legend", expanded=False):
            st.markdown(
                """
                Seeds indicate regular season finish - lower = better.
            """
            )
            legend_cols = st.columns(4)
            for i, col in enumerate(legend_cols, 1):
                if i <= 4:
                    color = PlayoffBracketsViewer.SEED_COLORS.get(i, "#9E9E9E")
                    col.markdown(
                        f"""
                        <div style='background: {color}; color: white;
                                    border-radius: 8px; text-align: center;
                                    font-weight: bold; padding: 0.5rem;'>
                            #{i}
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
