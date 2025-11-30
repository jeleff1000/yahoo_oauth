import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .table_styles import render_modern_table

# Import chart theming
try:
    from streamlit_ui.shared.chart_themes import get_chart_colors, apply_chart_theme
except ImportError:
    try:
        from shared.chart_themes import get_chart_colors, apply_chart_theme
    except ImportError:
        def get_chart_colors():
            return {'positive': '#00C07F', 'negative': '#FF4B4B', 'categorical': ['#667eea', '#f093fb', '#4facfe']}
        def apply_chart_theme(fig):
            return fig

_REQUIRED_COLS = {
    "manager", "year", "week", "is_playoffs", "is_consolation",
    "wins_to_date", "losses_to_date", "shuffle_avg_wins", "wins_vs_shuffle_wins"
}


def _validate_matchup_df(df: pd.DataFrame) -> bool:
    """Fast validation check."""
    if df is None or df.empty:
        st.info("No matchup data available.")
        return False
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(missing)}")
        return False
    return True


def _filter_largest_week(df: pd.DataFrame) -> pd.DataFrame:
    """Optimized filtering using groupby idxmax."""
    mask = (df["is_playoffs"] == 0) & (df["is_consolation"] == 0)
    df_filtered = df[mask]
    idx = df_filtered.groupby(["manager", "year"])["week"].idxmax()
    return df_filtered.loc[idx].reset_index(drop=True)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized type conversion."""
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["manager"] = df["manager"].astype(str).fillna("Unknown")

    numeric_cols = ["wins_to_date", "losses_to_date", "shuffle_avg_wins", "wins_vs_shuffle_wins"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


def _style_df(display_df: pd.DataFrame):
    """Consistent styling."""
    return (
        display_df.style
        .format({
            "win": "{:.0f}",
            "loss": "{:.0f}",
            "xWins": "{:.2f}",
            "xLosses": "{:.2f}",
            "delta": "{:.2f}",
        })
        .background_gradient(subset=["win", "xWins", "delta"], cmap="RdYlGn", axis=0)
        .background_gradient(subset=["loss", "xLosses"], cmap="RdYlGn_r", axis=0)
    )


def _make_display_df(src: pd.DataFrame) -> pd.DataFrame:
    """Vectorized display dataframe creation."""
    return pd.DataFrame({
        "win": src["wins_to_date"],
        "loss": src["losses_to_date"],
        "xWins": src["shuffle_avg_wins"],
        "xLosses": src["week"] - src["shuffle_avg_wins"],
        "delta": src["wins_vs_shuffle_wins"],
    })


def _build_manager_tab(df: pd.DataFrame):
    """Manager-specific view."""
    managers = sorted(df["manager"].astype(str).unique())
    selected_manager = st.selectbox("Select Manager", managers, key="gavi_mgr_sel")

    manager_df = df[df["manager"] == selected_manager].sort_values("year").copy()
    manager_df["year_str"] = manager_df["year"].astype(str)

    display_df = _make_display_df(manager_df).set_index(manager_df["year_str"])
    display_df.index.name = "Year"
    display_df.loc["Total"] = display_df.sum(numeric_only=True)

    # Render modern table with column-based gradient
    render_modern_table(
        display_df,
        title="",
        color_columns=["win", "xWins", "delta"],
        reverse_columns=["loss", "xLosses"],
        format_specs={
            "win": "{:.0f}",
            "loss": "{:.0f}",
            "xWins": "{:.2f}",
            "xLosses": "{:.2f}",
            "delta": "{:+.2f}"
        },
        column_names={
            "win": "W",
            "loss": "L",
            "xWins": "Exp W",
            "xLosses": "Exp L",
            "delta": "Luck"
        },
        gradient_by_column=True
    )


def _create_wins_chart(display_df: pd.DataFrame):
    """Create interactive chart comparing actual vs expected wins."""
    colors = get_chart_colors()

    chart_df = display_df.reset_index()
    chart_df.columns = ['Manager', 'Wins', 'Losses', 'xWins', 'xLosses', 'Delta']

    if 'Total' in chart_df['Manager'].values:
        chart_df = chart_df[chart_df['Manager'] != 'Total']

    fig = go.Figure()

    # Actual wins
    fig.add_trace(go.Bar(
        name='Actual Wins',
        x=chart_df['Manager'],
        y=chart_df['Wins'],
        marker_color=colors['categorical'][0],
        text=chart_df['Wins'].round(1),
        textposition='auto',
    ))

    # Expected wins
    fig.add_trace(go.Bar(
        name='Expected Wins',
        x=chart_df['Manager'],
        y=chart_df['xWins'],
        marker_color=colors['categorical'][2],
        text=chart_df['xWins'].round(2),
        textposition='auto',
    ))

    fig.update_layout(
        title="Actual vs Expected Wins",
        xaxis_title="Manager",
        yaxis_title="Wins",
        barmode='group',
        height=350,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    apply_chart_theme(fig)

    return fig


def _create_delta_chart(display_df: pd.DataFrame):
    """Create chart showing luck factor (delta from expected)."""
    colors = get_chart_colors()

    chart_df = display_df.reset_index()
    chart_df.columns = ['Manager', 'Wins', 'Losses', 'xWins', 'xLosses', 'Delta']

    if 'Total' in chart_df['Manager'].values:
        chart_df = chart_df[chart_df['Manager'] != 'Total']

    # Sort by delta for better visualization
    chart_df = chart_df.sort_values('Delta', ascending=True)

    # Color based on positive/negative
    bar_colors = [colors['positive'] if x > 0 else colors['negative'] for x in chart_df['Delta']]

    fig = go.Figure(go.Bar(
        x=chart_df['Manager'],
        y=chart_df['Delta'],
        marker_color=bar_colors,
        text=[f"{v:+.1f}" for v in chart_df['Delta']],
        textposition='outside',
        hovertemplate='%{x}<br>Luck: %{y:+.2f} wins<extra></extra>'
    ))

    fig.update_layout(
        title="Schedule Luck (Wins Above/Below Expected)",
        xaxis_title="Manager",
        yaxis_title="Delta (Actual - Expected)",
        height=350,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    apply_chart_theme(fig)

    return fig


def _build_year_tab(df: pd.DataFrame):
    """Year-specific view with optimized aggregation."""
    years_numeric = sorted(df["year"].dropna().astype(int).unique().tolist())
    year_labels = [str(y) for y in years_numeric]
    year_options = ["All"] + year_labels

    default_label = year_labels[-1] if year_labels else "All"
    default_index = year_options.index(default_label)

    selected_year_label = st.selectbox("Select Year", year_options,
                                       index=default_index, key="gavi_year_sel")

    if selected_year_label == "All":
        # Optimized aggregation
        grp = df.groupby("manager", dropna=False, as_index=False).agg({
            "wins_to_date": "sum",
            "losses_to_date": "sum",
            "shuffle_avg_wins": "sum",
            "week": "sum",
            "wins_vs_shuffle_wins": "sum",
        }).set_index("manager")

        display_df = pd.DataFrame({
            "win": grp["wins_to_date"],
            "loss": grp["losses_to_date"],
            "xWins": grp["shuffle_avg_wins"],
            "xLosses": grp["week"] - grp["shuffle_avg_wins"],
            "delta": grp["wins_vs_shuffle_wins"],
        })
    else:
        year_int = int(selected_year_label)
        year_df = df[df["year"] == year_int].set_index("manager")
        display_df = _make_display_df(year_df)

    display_df.index.name = "Manager"

    # Render modern table with column-based gradient
    render_modern_table(
        display_df,
        title="",
        color_columns=["win", "xWins", "delta"],
        reverse_columns=["loss", "xLosses"],
        format_specs={
            "win": "{:.0f}",
            "loss": "{:.0f}",
            "xWins": "{:.2f}",
            "xLosses": "{:.2f}",
            "delta": "{:+.2f}"
        },
        column_names={
            "win": "W",
            "loss": "L",
            "xWins": "Exp W",
            "xLosses": "Exp L",
            "delta": "Luck"
        },
        gradient_by_column=True
    )

    # Add visualization toggle below table
    st.markdown("---")
    show_charts = st.checkbox("📊 Show Visualizations", value=True, key="show_gavi_charts")

    if show_charts:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(_create_wins_chart(display_df), use_container_width=True)
        with col2:
            st.plotly_chart(_create_delta_chart(display_df), use_container_width=True)


def _build_all_tab(df: pd.DataFrame):
    """All data view."""
    df2 = df.copy()
    df2["year_str"] = df2["year"].astype(str)
    display_df = _make_display_df(df2).set_index([df2["manager"], df2["year_str"]])
    display_df.index.set_names(["Manager", "Year"], inplace=True)

    # Render modern table with column-based gradient
    render_modern_table(
        display_df,
        title="",
        color_columns=["win", "xWins", "delta"],
        reverse_columns=["loss", "xLosses"],
        format_specs={
            "win": "{:.0f}",
            "loss": "{:.0f}",
            "xWins": "{:.2f}",
            "xLosses": "{:.2f}",
            "delta": "{:+.2f}"
        },
        column_names={
            "win": "W",
            "loss": "L",
            "xWins": "Exp W",
            "xLosses": "Exp L",
            "delta": "Luck"
        },
        gradient_by_column=True
    )


class GaviStatViewer:
    def __init__(self, matchup_data_df: pd.DataFrame):
        self.df = matchup_data_df.copy() if matchup_data_df is not None else None

    @st.fragment
    def display(self):
        st.subheader("📊 Expected Wins with Shuffled Schedules")
        st.caption("What would your record be against randomized opponents? Based on 100K simulations.")

        if not _validate_matchup_df(self.df):
            return

        df = _coerce_types(self.df)
        df = _filter_largest_week(df)

        tab_year, tab_manager, tab_all = st.tabs(["📅 By Year", "👤 By Manager", "📋 All Data"])
        with tab_year:
            _build_year_tab(df)
        with tab_manager:
            _build_manager_tab(df)
        with tab_all:
            _build_all_tab(df)