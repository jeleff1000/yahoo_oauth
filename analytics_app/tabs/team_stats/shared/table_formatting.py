"""
Table formatting utilities for team stats.

Provides functions for:
- Value formatting (numbers, percentages, decimals)
- Conditional formatting (color coding, icons)
- Derived metric calculations
- Visual indicators (arrows, badges, emojis)
- Column styling helpers
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List


# ============================================================================
# VALUE FORMATTING
# ============================================================================


def format_number(
    value: float, decimals: int = 1, prefix: str = "", suffix: str = ""
) -> str:
    """
    Format a number with specified decimals and affixes.

    Args:
        value: Number to format
        decimals: Number of decimal places
        prefix: String to prepend (e.g., "$")
        suffix: String to append (e.g., "%")

    Returns:
        Formatted string
    """
    if pd.isna(value) or value is None:
        return "-"

    format_str = f"{{:.{decimals}f}}"
    return f"{prefix}{format_str.format(value)}{suffix}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    if pd.isna(value) or value is None:
        return "-"
    return f"{value:.{decimals}f}%"


def format_integer(value: float) -> str:
    """Format value as integer."""
    if pd.isna(value) or value is None:
        return "-"
    return f"{int(value)}"


def format_ratio(numerator: float, denominator: float, decimals: int = 1) -> str:
    """Format ratio with proper handling of division by zero."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return "-"
    ratio = numerator / denominator
    return f"{ratio:.{decimals}f}"


# ============================================================================
# VISUAL INDICATORS
# ============================================================================


def get_trend_arrow(
    current: float, previous: float, neutral_threshold: float = 0.5
) -> str:
    """
    Get trend arrow based on change between values.

    Args:
        current: Current value
        previous: Previous value
        neutral_threshold: Minimum change to show arrow

    Returns:
        Arrow emoji (↑, →, ↓)
    """
    if pd.isna(current) or pd.isna(previous):
        return ""

    diff = current - previous

    if abs(diff) < neutral_threshold:
        return "→"
    elif diff > 0:
        return "↑"
    else:
        return "↓"


def get_performance_emoji(percentile: float) -> str:
    """
    Get emoji based on percentile performance.

    Args:
        percentile: Percentile value (0-100)

    Returns:
        Performance emoji
    """
    if pd.isna(percentile):
        return ""

    if percentile >= 90:
        return "🔥"  # Elite
    elif percentile >= 75:
        return "⭐"  # Great
    elif percentile >= 50:
        return "✓"  # Good
    elif percentile >= 25:
        return "○"  # Average
    else:
        return "▽"  # Below average


def get_rank_badge(rank: int, total: int) -> str:
    """
    Get badge for ranking.

    Args:
        rank: Rank position (1 = best)
        total: Total number of items

    Returns:
        Rank badge string
    """
    if pd.isna(rank) or pd.isna(total):
        return "-"

    rank_int = int(rank)

    # Medal emojis for top 3
    if rank_int == 1:
        return "🥇 #1"
    elif rank_int == 2:
        return "🥈 #2"
    elif rank_int == 3:
        return "🥉 #3"
    else:
        return f"#{rank_int}"


# ============================================================================
# CONDITIONAL FORMATTING
# ============================================================================


def get_percentile_color(percentile: float, reverse: bool = False) -> str:
    """
    Get color based on percentile (higher is better by default).

    Args:
        percentile: Percentile value (0-100)
        reverse: If True, lower is better

    Returns:
        CSS color string
    """
    if pd.isna(percentile):
        return "#9ca3af"  # Gray for missing

    # Reverse percentile if lower is better (e.g., turnovers)
    pct = (100 - percentile) if reverse else percentile

    if pct >= 90:
        return "#10b981"  # Green - Elite
    elif pct >= 75:
        return "#84cc16"  # Lime - Great
    elif pct >= 50:
        return "#fbbf24"  # Yellow - Good
    elif pct >= 25:
        return "#fb923c"  # Orange - Below average
    else:
        return "#ef4444"  # Red - Poor


def get_value_color(
    value: float, thresholds: Dict[str, float], reverse: bool = False
) -> str:
    """
    Get color based on value thresholds.

    Args:
        value: Value to evaluate
        thresholds: Dict with keys 'elite', 'good', 'average', 'below'
        reverse: If True, lower is better

    Returns:
        CSS color string
    """
    if pd.isna(value):
        return "#9ca3af"

    val = value if not reverse else -value

    if val >= thresholds.get("elite", 100):
        return "#10b981"
    elif val >= thresholds.get("good", 75):
        return "#84cc16"
    elif val >= thresholds.get("average", 50):
        return "#fbbf24"
    elif val >= thresholds.get("below", 25):
        return "#fb923c"
    else:
        return "#ef4444"


# ============================================================================
# DERIVED METRICS
# ============================================================================


def calculate_completion_percentage(completions: float, attempts: float) -> float:
    """Calculate completion percentage."""
    if pd.isna(completions) or pd.isna(attempts) or attempts == 0:
        return np.nan
    return (completions / attempts) * 100


def calculate_yards_per_carry(yards: float, carries: float) -> float:
    """Calculate yards per carry."""
    if pd.isna(yards) or pd.isna(carries) or carries == 0:
        return np.nan
    return yards / carries


def calculate_yards_per_reception(yards: float, receptions: float) -> float:
    """Calculate yards per reception."""
    if pd.isna(yards) or pd.isna(receptions) or receptions == 0:
        return np.nan
    return yards / receptions


def calculate_yards_per_target(yards: float, targets: float) -> float:
    """Calculate yards per target."""
    if pd.isna(yards) or pd.isna(targets) or targets == 0:
        return np.nan
    return yards / targets


def calculate_catch_rate(receptions: float, targets: float) -> float:
    """Calculate catch rate percentage."""
    if pd.isna(receptions) or pd.isna(targets) or targets == 0:
        return np.nan
    return (receptions / targets) * 100


def calculate_td_percentage(tds: float, attempts: float) -> float:
    """Calculate touchdown percentage."""
    if pd.isna(tds) or pd.isna(attempts) or attempts == 0:
        return np.nan
    return (tds / attempts) * 100


def calculate_points_per_game(points: float, games: float) -> float:
    """Calculate points per game."""
    if pd.isna(points) or pd.isna(games) or games == 0:
        return np.nan
    return points / games


# ============================================================================
# DATAFRAME ENHANCEMENT
# ============================================================================


def add_derived_metrics(df: pd.DataFrame, position: str = "All") -> pd.DataFrame:
    """
    Add derived metrics to dataframe based on position.

    Args:
        df: Input dataframe
        position: Position type (QB, RB, WR, TE, K, DEF, All)

    Returns:
        DataFrame with added derived metrics
    """
    df = df.copy()

    # QB metrics
    if position in ["QB", "All"]:
        if "Comp" in df.columns and "Pass Att" in df.columns:
            df["Comp%"] = df.apply(
                lambda x: calculate_completion_percentage(x["Comp"], x["Pass Att"]),
                axis=1,
            )

        if "Pass Yds" in df.columns and "Pass Att" in df.columns:
            df["YPA"] = df.apply(
                lambda x: calculate_yards_per_carry(x["Pass Yds"], x["Pass Att"]),
                axis=1,
            )

        if "Pass TD" in df.columns and "Pass Att" in df.columns:
            df["TD%"] = df.apply(
                lambda x: calculate_td_percentage(x["Pass TD"], x["Pass Att"]), axis=1
            )

        if "INT" in df.columns and "Pass Att" in df.columns:
            df["INT%"] = df.apply(
                lambda x: calculate_td_percentage(x["INT"], x["Pass Att"]), axis=1
            )

    # RB/WR/TE metrics
    if position in ["RB", "WR", "TE", "All"]:
        if "Rush Yds" in df.columns and "Rush Att" in df.columns:
            df["YPC"] = df.apply(
                lambda x: calculate_yards_per_carry(x["Rush Yds"], x["Rush Att"]),
                axis=1,
            )

        if "Rec Yds" in df.columns and "Rec" in df.columns:
            df["YPR"] = df.apply(
                lambda x: calculate_yards_per_reception(x["Rec Yds"], x["Rec"]), axis=1
            )

        if "Rec Yds" in df.columns and "Targets" in df.columns:
            df["YPRT"] = df.apply(
                lambda x: calculate_yards_per_target(x["Rec Yds"], x["Targets"]), axis=1
            )

        if "Rec" in df.columns and "Targets" in df.columns:
            df["Catch%"] = df.apply(
                lambda x: calculate_catch_rate(x["Rec"], x["Targets"]), axis=1
            )

    # K metrics
    if position in ["K", "All"]:
        if "FGM" in df.columns and "FGA" in df.columns:
            # FG% might already exist, but recalculate for consistency
            df["FG%_Calc"] = df.apply(
                lambda x: calculate_completion_percentage(x["FGM"], x["FGA"]), axis=1
            )

        # Calculate 40+ FG success rate
        if "FG 40-49" in df.columns and "FG 50+" in df.columns:
            df["FG 40+"] = df["FG 40-49"].fillna(0) + df["FG 50+"].fillna(0)

    return df


def add_formatting_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add formatted display columns to dataframe.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with formatted columns
    """
    df = df.copy()

    # Format percentages if they exist
    for col in df.columns:
        if col.endswith("%") or col in ["Comp%", "Catch%", "FG%", "TD%", "INT%"]:
            df[f"{col}_Display"] = df[col].apply(
                lambda x: format_percentage(x, decimals=1) if pd.notna(x) else "-"
            )

    # Format point values
    if "Points" in df.columns:
        df["Points_Display"] = df["Points"].apply(
            lambda x: format_number(x, decimals=1) if pd.notna(x) else "-"
        )

    # Format ratios (YPC, YPR, etc.)
    for col in ["YPC", "YPR", "YPA", "YPRT"]:
        if col in df.columns:
            df[f"{col}_Display"] = df[col].apply(
                lambda x: format_number(x, decimals=1) if pd.notna(x) else "-"
            )

    return df


# ============================================================================
# STYLING FUNCTIONS
# ============================================================================


def style_dataframe(
    df: pd.DataFrame,
    highlight_columns: Optional[List[str]] = None,
    percentile_columns: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """
    Apply styling to dataframe using pandas Styler.

    Args:
        df: Input dataframe
        highlight_columns: Columns to apply gradient highlighting
        percentile_columns: Dict of column: reverse (True if lower is better)

    Returns:
        Styled dataframe
    """
    # Note: Streamlit doesn't support pandas Styler in st.dataframe
    # This function prepares data but actual styling must be done via column_config
    return df


def create_summary_stats(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Create summary statistics for specified columns.

    Args:
        df: Input dataframe
        columns: Columns to summarize

    Returns:
        Summary dataframe with mean, median, min, max
    """
    summary_data = {}

    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            summary_data[col] = {
                "Mean": df[col].mean(),
                "Median": df[col].median(),
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Std": df[col].std(),
            }

    if not summary_data:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_data).T
    summary_df.index.name = "Metric"

    return summary_df


# ============================================================================
# COMPARISON HELPERS
# ============================================================================


def add_league_comparison(
    df: pd.DataFrame, value_column: str, comparison_column_name: str = "vs League Avg"
) -> pd.DataFrame:
    """
    Add comparison to league average column.

    Args:
        df: Input dataframe
        value_column: Column to compare
        comparison_column_name: Name for new column

    Returns:
        DataFrame with comparison column
    """
    df = df.copy()

    if value_column not in df.columns:
        return df

    league_avg = df[value_column].mean()
    df[comparison_column_name] = df[value_column] - league_avg

    return df


def add_rank_column(
    df: pd.DataFrame,
    value_column: str,
    rank_column_name: str = "Rank",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Add ranking column.

    Args:
        df: Input dataframe
        value_column: Column to rank by
        rank_column_name: Name for rank column
        ascending: If True, lower values get better rank

    Returns:
        DataFrame with rank column
    """
    df = df.copy()

    if value_column not in df.columns:
        return df

    df[rank_column_name] = df[value_column].rank(ascending=ascending, method="min")

    return df


# ============================================================================
# TABLE ENHANCEMENT WRAPPER
# ============================================================================


def enhance_table_data(
    df: pd.DataFrame,
    position: str = "All",
    add_derived: bool = True,
    add_comparisons: bool = True,
    add_ranks: bool = False,
) -> pd.DataFrame:
    """
    Master function to enhance table data with all improvements.

    Args:
        df: Input dataframe
        position: Position type
        add_derived: Whether to add derived metrics
        add_comparisons: Whether to add league comparisons
        add_ranks: Whether to add ranking columns

    Returns:
        Enhanced dataframe
    """
    df = df.copy()

    # Add derived metrics
    if add_derived:
        df = add_derived_metrics(df, position)

    # Add comparisons
    if add_comparisons and "Points" in df.columns:
        df = add_league_comparison(df, "Points", "vs Avg")

    # Add ranks
    if add_ranks and "Points" in df.columns:
        df = add_rank_column(df, "Points", "Points Rank", ascending=False)

    return df
