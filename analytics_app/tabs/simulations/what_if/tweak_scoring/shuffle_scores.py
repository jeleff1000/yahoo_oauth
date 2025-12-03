"""
Optimized score shuffling with better performance.
Drop-in replacement for shuffle_scores.py
"""

import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(ttl=3600)
def calculate_std_dev(df, selected_year, show_regular_season, show_postseason):
    """
    Optimized standard deviation calculation with caching.
    """
    # Create filter mask
    mask = pd.Series(True, index=df.index)

    if selected_year != "All Years":
        mask &= df["year"] == selected_year

    if not show_regular_season:
        mask &= df["is_playoffs"]

    if not show_postseason:
        mask &= not df["is_playoffs"]

    filtered = df[mask]

    # Vectorized groupby aggregation
    std_dev_df = (
        filtered.groupby("manager", as_index=False)["team_points"]
        .std(ddof=1)
        .rename(columns={"team_points": "StdDev_TeamPoints"})
    )

    return std_dev_df


def tweak_scores(df, std_dev_df):
    """
    Optimized score tweaking with vectorized operations.
    Significantly faster than original implementation.
    """
    # Merge standard deviations once
    df = df.merge(std_dev_df, on="manager", how="left")

    # Vectorized random tweaking
    random_factors = np.random.uniform(-1 / 3, 1 / 3, size=len(df))
    df["tweaked_team_points"] = (
        df["team_points"] + random_factors * df["StdDev_TeamPoints"]
    )

    # Use existing opponent_points column (not opponent_points)
    # Vectorized win/loss calculation
    df["Sim_Wins"] = (df["tweaked_team_points"] > df["opponent_points"]).astype(int)
    df["Sim_Losses"] = (df["tweaked_team_points"] < df["opponent_points"]).astype(int)

    # Clean up temporary columns
    df = df.drop(columns=["StdDev_TeamPoints"])

    return df


def calculate_playoff_seed(df):
    """
    Optimized playoff seed calculation.
    Uses efficient aggregation and sorting.
    """
    # Vectorized aggregation
    agg_df = (
        df.groupby("manager", as_index=False)
        .agg({"Sim_Wins": "sum", "tweaked_team_points": "sum"})
        .rename(columns={"tweaked_team_points": "Total_Tweaked_Points"})
    )

    # Sort and assign seeds
    agg_df = agg_df.sort_values(
        by=["Sim_Wins", "Total_Tweaked_Points"], ascending=[False, False]
    ).reset_index(drop=True)

    agg_df["Sim_Playoff_Seed"] = range(1, len(agg_df) + 1)

    # Merge back to original dataframe
    df = df.merge(agg_df[["manager", "Sim_Playoff_Seed"]], on="manager", how="left")

    return df


def main(df, selected_year, show_regular_season, show_postseason, tweak_scores_flag):
    """
    Main orchestration function with optional caching.
    """
    # Calculate standard deviations (cached)
    std_dev_df = calculate_std_dev(
        df, selected_year, show_regular_season, show_postseason
    )

    # Apply score tweaking if requested
    if tweak_scores_flag:
        df = tweak_scores(df, std_dev_df)
    else:
        # If not tweaking, use original scores
        df["tweaked_team_points"] = df["team_points"]

    # Calculate playoff seeds
    df = calculate_playoff_seed(df)

    return df


# Additional utility functions for analysis


def batch_simulate(df, n_simulations=1000, **kwargs):
    """
    Run multiple simulations for statistical analysis.
    Returns aggregated results.
    """
    results = []

    for i in range(n_simulations):
        sim_df = main(df.copy(), **kwargs)

        # Extract key metrics
        summary = (
            sim_df.groupby("manager")
            .agg({"Sim_Wins": "sum", "Sim_Playoff_Seed": "first"})
            .reset_index()
        )
        summary["simulation"] = i
        results.append(summary)

    # Combine all simulations
    all_results = pd.concat(results, ignore_index=True)

    # Calculate statistics
    stats = (
        all_results.groupby("manager")
        .agg(
            {
                "Sim_Wins": ["mean", "std", "min", "max"],
                "Sim_Playoff_Seed": ["mean", "std"],
            }
        )
        .reset_index()
    )

    stats.columns = [
        "manager",
        "avg_wins",
        "std_wins",
        "min_wins",
        "max_wins",
        "avg_seed",
        "std_seed",
    ]

    return stats


@st.cache_data(ttl=3600)
def calculate_win_probabilities(
    df, selected_year, show_regular_season, show_postseason
):
    """
    Calculate win probability distributions using multiple simulations.
    Cached for performance.
    """
    std_dev_df = calculate_std_dev(
        df, selected_year, show_regular_season, show_postseason
    )

    # Run 1000 simulations
    sim_results = []

    for _ in range(1000):
        sim_df = tweak_scores(df.copy(), std_dev_df)
        wins = sim_df.groupby("manager")["Sim_Wins"].sum().to_dict()
        sim_results.append(wins)

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(sim_results)

    # Calculate percentiles
    percentiles = results_df.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).T
    percentiles.columns = ["p10", "p25", "median", "p75", "p90"]
    percentiles["mean"] = results_df.mean()
    percentiles["std"] = results_df.std()

    return percentiles.reset_index().rename(columns={"index": "manager"})
