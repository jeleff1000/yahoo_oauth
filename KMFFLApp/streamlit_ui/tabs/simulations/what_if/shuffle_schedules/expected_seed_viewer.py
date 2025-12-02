import streamlit as st
import pandas as pd
import numpy as np
from .table_styles import render_modern_table


def _select_week(base_df):
    """Standardized week selection with session state buttons."""
    if base_df.empty or base_df['year'].dropna().empty:
        st.info("No valid year data available.")
        return None, None

    # Session state buttons instead of radio
    mode_key = "exp_seed_mode"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = 0

    modes = ["Today's Date", "Specific Week"]
    cols = st.columns(2)
    for idx, (col, name) in enumerate(zip(cols, modes)):
        with col:
            is_active = (st.session_state[mode_key] == idx)
            if st.button(name, key=f"exp_seed_btn_{idx}", use_container_width=True,
                        type="primary" if is_active else "secondary"):
                if not is_active:
                    st.session_state[mode_key] = idx
                    st.rerun()

    mode = modes[st.session_state[mode_key]]

    if mode == "Today's Date":
        max_year = base_df['year'].max()
        if pd.isna(max_year):
            st.info("No valid year data available.")
            return None, None
        year = int(max_year)
        week = int(base_df[base_df['year'] == year]['week'].max())
        st.caption(f"Auto-selected Year {year}, Week {week}")
    else:
        years = sorted(base_df['year'].unique())
        if not years:
            st.info("No valid year data available.")
            return None, None

        c_year, c_week = st.columns(2)
        year_choice = c_year.selectbox("Year", ["Select Year"] + [str(y) for y in years],
                                       key="exp_seed_year")
        if year_choice == "Select Year":
            return None, None
        year = int(year_choice)

        weeks = sorted(base_df[base_df['year'] == year]['week'].unique())
        if not weeks:
            st.info("No valid week data available.")
            return None, None

        week_choice = c_week.selectbox("Week", ["Select Week"] + [str(w) for w in weeks],
                                       key="exp_seed_week")
        if week_choice == "Select Week":
            return None, None
        week = int(week_choice)

    return year, week


@st.fragment
def _render_expected_seed(base_df, year, week):
    """Optimized rendering with vectorized operations."""
    # Filter once
    week_df = base_df[(base_df['year'] == year) & (base_df['week'] == week)]
    if week_df.empty:
        st.info("No rows for selected year/week.")
        return

    # Vectorized column filtering
    all_cols = week_df.columns
    seed_cols = [c for c in all_cols if c.startswith("shuffle_") and c.endswith("_seed")]

    if not seed_cols:
        st.info("No valid shuffle seed cols.")
        return

    # Extract week numbers efficiently
    week_numbers = {c: int(c.split('_')[1]) for c in seed_cols
                    if len(c.split('_')) >= 3 and c.split('_')[1].isdigit()}
    seed_cols = sorted(week_numbers.keys(), key=lambda c: week_numbers[c])

    # Build dataframe efficiently
    cols = ['manager'] + seed_cols
    df = (week_df[cols]
          .drop_duplicates(subset=['manager'])
          .set_index('manager')
          .sort_index())

    # Add actual seed if available
    if 'playoff_seed_to_date' in week_df.columns:
        actual_seed = (week_df[['manager', 'playoff_seed_to_date']]
                       .drop_duplicates(subset=['manager'])
                       .set_index('manager')['playoff_seed_to_date']
                       .rename('Actual Seed'))
        df = df.join(actual_seed)

    # Vectorized numeric conversion
    df[seed_cols] = df[seed_cols].apply(pd.to_numeric, errors='coerce')

    # Calculate percentages efficiently
    bye_source = [c for c in seed_cols if week_numbers[c] in (1, 2)]
    playoff_source = [c for c in seed_cols if week_numbers[c] <= 6]

    df['Bye%'] = df[bye_source].sum(axis=1) if bye_source else 0.0
    df['Playoff%'] = df[playoff_source].sum(axis=1) if playoff_source else 0.0

    # Rename and order columns
    rename_map = {c: str(week_numbers[c]) for c in seed_cols}
    df = df.rename(columns=rename_map)

    iteration_cols = sorted([str(v) for v in week_numbers.values()], key=int)
    ordered = iteration_cols + ['Bye%', 'Playoff%']
    if 'Actual Seed' in df.columns:
        ordered.append('Actual Seed')
        df['Actual Seed'] = pd.to_numeric(df['Actual Seed'], errors='coerce')

    df = df[ordered]

    # Format percentages
    numeric_percent_cols = iteration_cols + ['Bye%', 'Playoff%']
    df[numeric_percent_cols] = df[numeric_percent_cols].round(2)

    # Sort by Playoff% descending (best to worst)
    df = df.sort_values('Playoff%', ascending=False)

    st.subheader("📊 Expected Seeding Distribution")
    st.caption("Probability of finishing at each playoff seed based on schedule shuffles.")

    # Create format specs
    fmt = {c: '{:.1f}' for c in numeric_percent_cols}
    if 'Actual Seed' in df.columns:
        fmt['Actual Seed'] = '{:.0f}'

    # Column names - just the seed number, no % symbol
    column_names = {c: c for c in iteration_cols}  # Just the number (1, 2, 3, etc.)
    column_names['Bye%'] = 'Bye'
    column_names['Playoff%'] = 'Playoff'
    if 'Actual Seed' in df.columns:
        column_names['Actual Seed'] = 'Actual'

    # Color code ONLY seed projection columns (not Bye%, Playoff%, or Actual)
    render_modern_table(
        df,
        title="",
        color_columns=iteration_cols,  # Only the seed columns get color coding
        reverse_columns=[],
        format_specs=fmt,
        column_names=column_names,
        gradient_by_column=True
    )

    st.caption("💡 **How to read:** Each numbered column shows % chance of that seed. Bye = top 2 seeds, Playoff = top 6 seeds.")


@st.fragment
def display_expected_seed(matchup_data_df):
    """Main entry point."""
    if matchup_data_df is None or matchup_data_df.empty:
        st.write("No data available")
        return

    # Filter to regular season only - vectorized
    base_df = matchup_data_df[
        (matchup_data_df['is_playoffs'] == 0) &
        (matchup_data_df['is_consolation'] == 0)
        ].copy()

    if base_df.empty:
        st.write("No regular season data available")
        return

    # Type conversion once
    base_df['year'] = pd.to_numeric(base_df['year'], errors='coerce')
    base_df['week'] = pd.to_numeric(base_df['week'], errors='coerce')
    base_df = base_df.dropna(subset=['year', 'week'])
    base_df['year'] = base_df['year'].astype(int)
    base_df['week'] = base_df['week'].astype(int)

    year, week = _select_week(base_df)
    if year is None or week is None:
        return

    _render_expected_seed(base_df, year, week)