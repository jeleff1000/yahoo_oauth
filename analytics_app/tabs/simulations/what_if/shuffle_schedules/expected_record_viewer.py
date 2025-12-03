import streamlit as st
import pandas as pd
from .table_styles import render_modern_table
from ...shared.simulation_styles import render_section_header


def _select_week(base_df):
    """Standardized week selection with session state buttons."""
    if base_df.empty or base_df["year"].dropna().empty:
        st.info("No valid year data available.")
        return None, None

    # Session state buttons instead of radio
    mode_key = "exp_record_mode"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = 0

    modes = ["Today's Date", "Specific Week"]
    cols = st.columns(2)
    for idx, (col, name) in enumerate(zip(cols, modes)):
        with col:
            is_active = st.session_state[mode_key] == idx
            if st.button(
                name,
                key=f"exp_record_btn_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    st.session_state[mode_key] = idx
                    st.rerun()

    mode = modes[st.session_state[mode_key]]

    if mode == "Today's Date":
        max_year = base_df["year"].max()
        if pd.isna(max_year):
            st.info("No valid year data available.")
            return None, None
        year = int(max_year)
        week = int(base_df[base_df["year"] == year]["week"].max())
        st.caption(f"Auto-selected Year {year}, Week {week}")
    else:
        years = sorted(base_df["year"].unique())
        if not years:
            st.info("No valid year data available.")
            return None, None

        c_year, c_week = st.columns(2)
        year_choice = c_year.selectbox(
            "Year", ["Select Year"] + [str(y) for y in years], key="exp_record_year"
        )
        if year_choice == "Select Year":
            return None, None
        year = int(year_choice)

        weeks = sorted(base_df[base_df["year"] == year]["week"].unique())
        if not weeks:
            st.info("No valid week data available.")
            return None, None

        week_choice = c_week.selectbox(
            "Week", ["Select Week"] + [str(w) for w in weeks], key="exp_record_week"
        )
        if week_choice == "Select Week":
            return None, None
        week = int(week_choice)

    return year, week


@st.fragment
def _render_expected_record(base_df, year, week):
    """Optimized rendering with vectorized operations."""
    week_slice = base_df[(base_df["year"] == year) & (base_df["week"] == week)]
    if week_slice.empty:
        st.info("No rows for selected year/week.")
        return

    # Vectorized column filtering
    all_cols = week_slice.columns
    shuffle_cols = [
        c for c in all_cols if c.startswith("shuffle_") and c.endswith("_win")
    ]

    # Extract week numbers efficiently
    valid_cols = []
    for c in shuffle_cols:
        parts = c.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            col_week = int(parts[1])
            if col_week <= week:
                valid_cols.append((col_week, c))

    if not valid_cols:
        st.info("No shuffle win cols.")
        return

    valid_cols.sort(key=lambda x: x[0])
    shuffle_cols = [c for _, c in valid_cols]

    # Build dataframe efficiently
    needed = ["manager", "wins_to_date", "losses_to_date"] + shuffle_cols
    needed = [c for c in needed if c in week_slice.columns]

    df = (
        week_slice[needed]
        .drop_duplicates(subset=["manager"])
        .set_index("manager")
        .sort_index()
    )

    # Vectorized renaming
    rename_map = {
        c: f"{int(c.split('_')[1])}-{week - int(c.split('_')[1])}" for c in shuffle_cols
    }
    df = df.rename(columns=rename_map)

    # Create actual record efficiently
    if {"wins_to_date", "losses_to_date"}.issubset(df.columns):
        df["Actual Record"] = (
            df["wins_to_date"].fillna(0).astype(int).astype(str)
            + "-"
            + df["losses_to_date"].fillna(0).astype(int).astype(str)
        )
        df = df.drop(columns=["wins_to_date", "losses_to_date"])

    # Order columns - reversed so best records (10-0, 9-1) come first
    record_cols = [c for c in df.columns if c != "Actual Record"]
    ordered = sorted(
        record_cols, key=lambda c: int(c.split("-")[0]) if "-" in c else 0, reverse=True
    )
    if "Actual Record" in df.columns:
        ordered.append("Actual Record")

    df = df[ordered]

    # Sort by actual wins (best to worst)
    if "wins_to_date" in week_slice.columns:
        # Create a temporary wins column for sorting
        wins_series = (
            week_slice[["manager", "wins_to_date"]]
            .drop_duplicates(subset=["manager"])
            .set_index("manager")["wins_to_date"]
        )
        df = df.join(wins_series.rename("_sort_wins"))
        df = df.sort_values("_sort_wins", ascending=False).drop(columns=["_sort_wins"])

    render_section_header("Expected Record Distribution", "")
    st.caption("Probability of each win-loss record based on schedule shuffles.")

    # Identify numeric columns for gradient (all except Actual Record)
    numeric_cols = [c for c in df.columns if c != "Actual Record"]

    # Create column rename map - just the record, no "W-L" prefix
    column_names = {}
    if "Actual Record" in df.columns:
        column_names["Actual Record"] = "Actual"

    # Create format specs
    format_specs = {c: "{:.1f}" for c in numeric_cols}

    render_modern_table(
        df,
        title="",
        color_columns=numeric_cols,
        reverse_columns=[],
        format_specs=format_specs,
        column_names=column_names,
        gradient_by_column=True,
    )

    st.caption(
        "💡 **How to read:** Each column shows % chance of that record. Higher % in winning records = stronger team."
    )


@st.fragment
def display_expected_record(matchup_data_df):
    """Main entry point."""
    if matchup_data_df is None or matchup_data_df.empty:
        st.write("No data available")
        return

    # Vectorized filtering
    base_df = matchup_data_df[
        (matchup_data_df["is_playoffs"] == 0) & (matchup_data_df["is_consolation"] == 0)
    ].copy()

    if base_df.empty:
        st.write("No regular season data available")
        return

    # Type conversion once
    base_df["year"] = pd.to_numeric(base_df["year"], errors="coerce")
    base_df["week"] = pd.to_numeric(base_df["week"], errors="coerce")
    base_df = base_df.dropna(subset=["year", "week"])
    base_df["year"] = base_df["year"].astype(int)
    base_df["week"] = base_df["week"].astype(int)

    year, week = _select_week(base_df)
    if year is None or week is None:
        return

    _render_expected_record(base_df, year, week)
