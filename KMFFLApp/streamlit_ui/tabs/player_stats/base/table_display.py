"""
Enhanced table display with better performance and UX.
"""
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict


class EnhancedTableDisplay:
    """
    High-performance table display with virtual scrolling, column selection,
    and better UX without filtering data.
    """

    def __init__(self, key_prefix: str):
        self.key_prefix = key_prefix
        self.rows_key = f"{key_prefix}_displayed_rows"
        self.columns_key = f"{key_prefix}_selected_columns"
        self.initial_load = 15000  # Increased to 15k - no performance issues observed
        self.load_increment = 5000  # Load 5k at a time when "Load More" clicked

        # Initialize session state
        if self.rows_key not in st.session_state:
            st.session_state[self.rows_key] = self.initial_load

    def get_position_specific_columns(self, df: pd.DataFrame, position: str = None) -> List[str]:
        """
        Get position-specific default columns for skinnier, more relevant displays.

        Args:
            df: The dataframe to get columns from
            position: The position to get defaults for (QB, RB, WR, TE, K, DEF, or None for All)

        Returns:
            List of column names that are most relevant for that position
        """
        # MATCHUP STATS: Show matchup-specific columns
        if 'matchup' in self.key_prefix:
            matchup_cols = [
                'Player', 'Pts', 'Pos', 'Team', 'GP', 'PPG',
                'Manager', 'Unique Mgrs', 'Opponent',
                'My Team', 'Opp Team', 'Margin',
                'Wins', 'Losses',  # Career has Losses
                'Playoffs', 'Playoff Apps',  # Career uses Playoff Apps
                'Year', 'Week'  # Week only in weekly/season
            ]
            return [col for col in matchup_cols if col in df.columns]

        # Core columns always shown (using renamed friendly names)
        core_cols = ['Player', 'Team', 'Week', 'Year', 'Manager', 'Points']

        # Position-specific relevant columns (using renamed friendly names)
        position_cols = {
            'QB': ['Position', 'Pass Yds', 'Pass TD', 'Pass INT', 'Comp', 'Pass Att',
                   'Rush Yds', 'Rush TD', 'Pass EPA', 'CPOE', 'PACR'],

            'RB': ['Position', 'Rush Yds', 'Rush TD', 'Att', 'YPC',
                   'Rec', 'Rec Yds', 'Rec TD', 'Tgt', 'Catch%',
                   'Rush EPA', 'Rec EPA', 'Tgt %'],

            'WR': ['Position', 'Rec', 'Rec Yds', 'Rec TD', 'Tgt', 'Catch%', 'YPR',
                   'Tgt %', 'Rec Air Yds', 'Rec YAC', 'Air Yds %',
                   'Rec EPA', 'WOPR', 'RACR'],

            'TE': ['Position', 'Rec', 'Rec Yds', 'Rec TD', 'Tgt', 'Catch%', 'YPR',
                   'Tgt %', 'Rec Air Yds', 'Rec YAC', 'Rec EPA'],

            'K': ['Position', 'FGM', 'FGA', 'FG%', 'FG 0-19', 'FG 20-29', 'FG 30-39',
                  'FG 40-49', 'FG 50+', 'XPM', 'XPA'],

            'DEF': ['Position', 'Sacks', 'INT', 'PA', 'TD', 'FF', 'Fum Rec',
                    'Total Tkl', 'TFL', 'PD', 'Total Yds Allow'],
        }

        # Get position-specific columns or default to common stats
        if position and position in position_cols:
            additional_cols = position_cols[position]
        else:
            # All positions / mixed - show general stats
            additional_cols = ['Position', 'Roster Slot']

        # Combine core + position-specific, filter to only existing columns
        all_cols = core_cols + additional_cols
        return [col for col in all_cols if col in df.columns]

    def display_column_selector(self, df: pd.DataFrame, default_columns: List[str] = None,
                                active_position: str = None):
        """
        Display a column selector to reduce table width and improve performance.
        Now supports position-specific defaults that auto-update.

        Args:
            df: DataFrame to select columns from
            default_columns: Manual default columns (overrides position-specific)
            active_position: Current position filter (QB, RB, etc.) for smart defaults
        """
        if df.empty:
            return default_columns or df.columns.tolist()

        # CRITICAL: Remove duplicate columns to prevent React errors
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        all_columns = df.columns.tolist()

        # Smart defaults based on position or provided defaults
        if default_columns is None:
            if active_position:
                default_columns = self.get_position_specific_columns(df, active_position)
            else:
                # General defaults when no position selected
                priority_cols = ['Player', 'Points', 'Team', 'Week', 'Year', 'Manager',
                               'Position', 'Roster Slot']
                default_columns = [col for col in priority_cols if col in all_columns]
                if not default_columns:
                    default_columns = all_columns[:8]

        # Initialize or update selected columns based on position change
        position_key = f"{self.columns_key}_last_position"
        manual_override_key = f"{self.columns_key}_manual_override"

        if position_key not in st.session_state:
            st.session_state[position_key] = active_position
        if manual_override_key not in st.session_state:
            st.session_state[manual_override_key] = False

        # If position changed AND user hasn't manually customized, update default columns
        # This prevents infinite loops while still providing smart defaults
        if st.session_state[position_key] != active_position and not st.session_state[manual_override_key]:
            st.session_state[position_key] = active_position
            st.session_state[self.columns_key] = default_columns

        # Initialize selected columns in session state
        if self.columns_key not in st.session_state:
            st.session_state[self.columns_key] = default_columns

        with st.expander("📊 Customize Columns", expanded=False):
            # Show position hint if active
            if active_position:
                st.info(f"💡 Showing **{active_position}**-specific columns. Change position to see different stats!")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown("**Select columns to display:**")

            with col2:
                if st.button("Select All", key=f"{self.key_prefix}_select_all"):
                    st.session_state[self.columns_key] = all_columns
                    st.session_state[manual_override_key] = True  # Mark as manually overridden
                    st.rerun()

            with col3:
                if st.button("Reset Default", key=f"{self.key_prefix}_reset_cols"):
                    st.session_state[self.columns_key] = default_columns
                    st.session_state[manual_override_key] = False  # Reset manual override flag
                    st.rerun()

            # Group columns by category for better organization
            column_groups = self._group_columns(all_columns)

            for group_name, columns in column_groups.items():
                if columns:
                    st.markdown(f"**{group_name}**")
                    cols_per_row = 4
                    for i in range(0, len(columns), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(columns[i:i+cols_per_row]):
                            with cols[j]:
                                is_selected = col in st.session_state[self.columns_key]
                                # Use unique key with group name and index to avoid duplicates
                                unique_key = f"{self.key_prefix}_col_{group_name}_{i+j}_{col}"
                                if st.checkbox(col, value=is_selected, key=unique_key):
                                    if col not in st.session_state[self.columns_key]:
                                        st.session_state[self.columns_key].append(col)
                                        st.session_state[manual_override_key] = True  # Mark as manually changed
                                else:
                                    if col in st.session_state[self.columns_key]:
                                        st.session_state[self.columns_key].remove(col)
                                        st.session_state[manual_override_key] = True  # Mark as manually changed

        return st.session_state[self.columns_key]

    def _group_columns(self, columns: List[str]) -> Dict[str, List[str]]:
        """Group columns by category for better organization."""
        groups = {
            "Core Info": [],
            "Passing": [],
            "Rushing": [],
            "Receiving": [],
            "Defense": [],
            "Kicking": [],
            "Advanced": [],
            "Other": []
        }

        for col in columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['player', 'team', 'week', 'year', 'manager',
                                            'position', 'points', 'opponent']):
                groups["Core Info"].append(col)
            elif any(x in col_lower for x in ['pass', 'completion', 'int', 'sack']):
                groups["Passing"].append(col)
            elif any(x in col_lower for x in ['rush', 'carry', 'attempt']):
                groups["Rushing"].append(col)
            elif any(x in col_lower for x in ['rec', 'target', 'catch', 'air_yards']):
                groups["Receiving"].append(col)
            elif any(x in col_lower for x in ['def', 'tackle', 'sack', 'interception',
                                              'fumble_rec', 'allow']):
                groups["Defense"].append(col)
            elif any(x in col_lower for x in ['fg', 'pat', 'field_goal', 'extra_point']):
                groups["Kicking"].append(col)
            elif any(x in col_lower for x in ['epa', 'cpoe', 'pacr', 'wopr', 'racr',
                                              'share', 'target_share']):
                groups["Advanced"].append(col)
            else:
                groups["Other"].append(col)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def display_table_with_load_more(
        self,
        df: pd.DataFrame,
        total_available: int,
        on_load_more_callback=None,
        height: int = 600
    ):
        """
        Display table with 'Load More' functionality instead of pagination.
        More intuitive than pagination and doesn't break user's scroll position.
        For season and career stats, show all rows since performance is acceptable.
        """
        if df.empty:
            st.warning("No data available.")
            return

        # For season and career stats, show all rows at once
        is_season_or_career = self.key_prefix.startswith(("season_", "career_"))
        if is_season_or_career:
            rows_shown = len(df)
        else:
            current_rows = st.session_state.get(self.rows_key, self.initial_load)
            rows_shown = min(len(df), current_rows)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Showing", f"{rows_shown:,}")
        with col2:
            # Handle None total_available (lazy count optimization)
            if total_available is not None:
                st.metric("Total Available", f"{total_available:,}")
            else:
                # Check if there's more data available
                has_more = getattr(df, 'attrs', {}).get('has_more', True)
                st.metric("Total Available", "~" + f"{rows_shown:,}+" if has_more else f"{rows_shown:,}")
        with col3:
            if total_available is not None:
                remaining = max(0, total_available - rows_shown)
                st.metric("Remaining", f"{remaining:,}")
            else:
                has_more = getattr(df, 'attrs', {}).get('has_more', True)
                st.metric("More Available", "Yes" if has_more else "No")

        # Display the table with fixed height for consistent UX
        display_df = df.head(rows_shown).copy()

        # CRITICAL: Ensure all column names are strings FIRST
        display_df.columns = [str(col) for col in display_df.columns]

        # CRITICAL: Remove duplicate columns before displaying to prevent React error #185
        # Check both exact and case-insensitive duplicates
        if display_df.columns.duplicated().any():
            display_df = display_df.loc[:, ~display_df.columns.duplicated(keep='first')]

        # Also check for case-insensitive duplicates (e.g., "Player" and "player")
        lower_cols = [c.lower() for c in display_df.columns]
        if len(lower_cols) != len(set(lower_cols)):
            # Find and remove case-insensitive duplicates
            seen = set()
            cols_to_keep = []
            for i, col in enumerate(display_df.columns):
                col_lower = col.lower()
                if col_lower not in seen:
                    seen.add(col_lower)
                    cols_to_keep.append(i)
            display_df = display_df.iloc[:, cols_to_keep]

        # Final safety check - log any remaining duplicates
        if display_df.columns.duplicated().any():
            dup_cols = display_df.columns[display_df.columns.duplicated()].tolist()
            st.warning(f"Warning: Duplicate columns detected: {dup_cols}")
            display_df = display_df.loc[:, ~display_df.columns.duplicated(keep='first')]

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=height
        )

        # Load more controls only for non-season/career views
        has_more = getattr(df, 'attrs', {}).get('has_more', True)
        show_load_more = not is_season_or_career and (
            (total_available is not None and rows_shown < total_available) or
            (total_available is None and has_more)
        )

        if show_load_more:
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button(
                    f"⬇️ Load {self.load_increment} More",
                    key=f"{self.key_prefix}_load_more",
                    use_container_width=True
                ):
                    st.session_state[self.rows_key] = current_rows + self.load_increment
                    if on_load_more_callback:
                        on_load_more_callback(st.session_state[self.rows_key])
                    st.rerun()

            with col2:
                if st.button(
                    "⬇️⬇️ Load All",
                    key=f"{self.key_prefix}_load_all",
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state[self.rows_key] = total_available
                    if on_load_more_callback:
                        on_load_more_callback(st.session_state[self.rows_key])
                    st.rerun()

            with col3:
                st.caption(f"💡 Tip: Loading all {total_available:,} rows may take a moment")
        elif is_season_or_career:
            st.success(f"✅ All {total_available:,} rows loaded")
        else:
            st.success(f"✅ All {total_available:,} rows loaded")

    def _get_column_config(self, df: pd.DataFrame) -> Dict:
        """
        Get optimized column configuration for better display.
        Ensures all numeric columns are properly typed for correct sorting.
        """
        config = {}

        # CRITICAL: Remove duplicate columns if they exist to prevent React errors
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # Ensure column names are strings
        df.columns = [str(col) for col in df.columns]

        # First, ensure numeric columns are actually numeric
        numeric_stat_keywords = ['points', 'yds', 'yards', 'td', 'int', 'rec', 'att', 'attempt',
                                  'fg', 'pat', 'sack', 'target', 'completion', 'carry',
                                  'fum', 'fumble', 'epa', 'cpoe', 'pacr', 'wopr', 'racr',
                                  'share', 'air_yards', 'tfl', 'def', 'allow', 'safe']

        for col in df.columns:
            try:
                col_lower = col.lower()

                # Check if column should be numeric
                is_numeric_col = any(keyword in col_lower for keyword in numeric_stat_keywords)

                # Ensure we're working with a Series, not a DataFrame
                col_data = df[col]
                if isinstance(col_data, pd.DataFrame):
                    # Skip if it's a DataFrame (duplicate column name)
                    continue

                # If it should be numeric but isn't, try to convert it
                if is_numeric_col and col_data.dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(col_data, errors='coerce')
                        col_data = df[col]
                    except:
                        pass

                # Number columns - configure with proper formatting
                if col_data.dtype in ['int64', 'float64', 'Int64', 'Float64']:
                    if 'points' in col_lower or 'yds' in col_lower or 'yards' in col_lower or 'epa' in col_lower:
                        config[col] = st.column_config.NumberColumn(
                            col,
                            format="%.1f",
                            help=f"{col} statistic"
                        )
                    elif any(x in col_lower for x in ['%', 'percentage', 'rate', 'share', 'cpoe', 'pacr', 'wopr', 'racr']):
                        config[col] = st.column_config.NumberColumn(
                            col,
                            format="%.2f",
                            help=f"{col} statistic"
                        )
                    else:
                        config[col] = st.column_config.NumberColumn(
                            col,
                            format="%d",
                            help=f"{col} statistic"
                        )

                # Player names - wider column
                elif 'player' in col_lower and 'optimal' not in col_lower:
                    config[col] = st.column_config.TextColumn(
                        col,
                        width="medium",
                        help="Player name"
                    )

                # Boolean columns
                elif col_data.dtype == 'bool':
                    config[col] = st.column_config.CheckboxColumn(
                        col,
                        help=f"{col}",
                        disabled=True
                    )
            except Exception:
                # Skip columns that cause errors
                continue

        return config

    def display_quick_export(self, df: pd.DataFrame, filename_prefix: str = "player_data"):
        """
        Quick export button for users who want the full dataset.
        """
        if df.empty:
            return

        with st.expander("💾 Export Data", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="📥 Download as CSV",
                    data=df.to_csv(index=False),
                    file_name=f"{filename_prefix}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                # For smaller datasets, offer Excel
                if len(df) < 10000:
                    try:
                        from io import BytesIO
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Data')

                        st.download_button(
                            label="📥 Download as Excel",
                            data=buffer.getvalue(),
                            file_name=f"{filename_prefix}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except ImportError:
                        st.caption("Excel export requires openpyxl")

    def reset_display(self):
        """Reset display state."""
        if self.rows_key in st.session_state:
            st.session_state[self.rows_key] = self.initial_load
