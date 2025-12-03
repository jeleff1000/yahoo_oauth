"""
Fun facts generation for matchup data.
Provides data-driven insights at different aggregation levels.
"""

import pandas as pd
from typing import List


def get_weekly_fun_facts(matchup_df: pd.DataFrame) -> List[str]:
    """
    Generate fun facts based on weekly matchup data.

    Args:
        matchup_df: DataFrame with weekly matchup data

    Returns:
        List of fun fact strings
    """
    facts = []

    try:
        # Highest single-week score
        if (
            "team_points" in matchup_df.columns
            and "manager" in matchup_df.columns
            and "year" in matchup_df.columns
            and "week" in matchup_df.columns
        ):
            max_score_row = matchup_df.loc[matchup_df["team_points"].idxmax()]
            facts.append(
                f"The highest single-week team score was {max_score_row['team_points']:.2f} points "
                f"by {max_score_row['manager']} in Week {int(max_score_row['week'])}, {int(max_score_row['year'])}"
            )

        # Closest margin
        if (
            "margin" in matchup_df.columns
            and "week" in matchup_df.columns
            and "year" in matchup_df.columns
        ):
            matchup_df["abs_margin"] = matchup_df["margin"].abs()
            closest_row = matchup_df.loc[matchup_df["abs_margin"].idxmin()]
            facts.append(
                f"{closest_row['manager']} won by just {abs(closest_row['margin']):.2f} points "
                f"in Week {int(closest_row['week'])}, {int(closest_row['year'])} - one of the closest margins ever"
            )

        # Biggest blowout
        if (
            "margin" in matchup_df.columns
            and "week" in matchup_df.columns
            and "year" in matchup_df.columns
        ):
            biggest_win_row = matchup_df.loc[matchup_df["margin"].idxmax()]
            facts.append(
                f"The biggest blowout was {biggest_win_row['manager']} crushing {biggest_win_row['opponent']} "
                f"by {biggest_win_row['margin']:.2f} points in Week {int(biggest_win_row['week'])}, {int(biggest_win_row['year'])}"
            )

        # Most bench points in a win
        if (
            "optimal_points" in matchup_df.columns
            and "team_points" in matchup_df.columns
            and "win" in matchup_df.columns
            and "week" in matchup_df.columns
        ):
            wins_df = matchup_df[matchup_df["win"]].copy()
            if not wins_df.empty:
                wins_df["bench_pts"] = (
                    wins_df["optimal_points"] - wins_df["team_points"]
                )
                max_bench_row = wins_df.loc[wins_df["bench_pts"].idxmax()]
                facts.append(
                    f"{max_bench_row['manager']} left {max_bench_row['bench_pts']:.2f} points on the bench "
                    f"but still won in Week {int(max_bench_row['week'])}, {int(max_bench_row['year'])}"
                )

        # Highest scoring loss
        if (
            "team_points" in matchup_df.columns
            and "loss" in matchup_df.columns
            and "week" in matchup_df.columns
        ):
            losses_df = matchup_df[matchup_df["loss"]]
            if not losses_df.empty:
                highest_loss_row = losses_df.loc[losses_df["team_points"].idxmax()]
                facts.append(
                    f"{highest_loss_row['manager']} lost despite scoring {highest_loss_row['team_points']:.2f} points "
                    f"in Week {int(highest_loss_row['week'])}, {int(highest_loss_row['year'])}"
                )

        # Lowest scoring win
        if (
            "team_points" in matchup_df.columns
            and "win" in matchup_df.columns
            and "week" in matchup_df.columns
        ):
            wins_df = matchup_df[matchup_df["win"]]
            if not wins_df.empty:
                lowest_win_row = wins_df.loc[wins_df["team_points"].idxmin()]
                facts.append(
                    f"{lowest_win_row['manager']} won with only {lowest_win_row['team_points']:.2f} points "
                    f"in Week {int(lowest_win_row['week'])}, {int(lowest_win_row['year'])}"
                )

        # Biggest comeback (lost on projections but won actual)
        if (
            "manager_proj_score" in matchup_df.columns
            and "opponent_proj_score" in matchup_df.columns
            and "win" in matchup_df.columns
        ):
            matchup_df["proj_underdog"] = (
                matchup_df["manager_proj_score"] < matchup_df["opponent_proj_score"]
            )
            matchup_df["proj_diff"] = (
                matchup_df["opponent_proj_score"] - matchup_df["manager_proj_score"]
            )
            underdogs = matchup_df[
                (matchup_df["proj_underdog"]) & (matchup_df["win"])
            ]
            if not underdogs.empty:
                biggest_upset = underdogs.loc[underdogs["proj_diff"].idxmax()]
                facts.append(
                    f"{biggest_upset['manager']} pulled off a {biggest_upset['proj_diff']:.2f} point upset "
                    f"over {biggest_upset['opponent']} in Week {int(biggest_upset['week'])}, {int(biggest_upset['year'])}"
                )

        # Perfect optimal lineup that won
        if (
            "optimal_points" in matchup_df.columns
            and "team_points" in matchup_df.columns
            and "win" in matchup_df.columns
        ):
            matchup_df["perfect_lineup"] = (
                matchup_df["optimal_points"] - matchup_df["team_points"]
            ).abs() < 0.5
            perfect_wins = matchup_df[
                (matchup_df["perfect_lineup"]) & (matchup_df["win"])
            ]
            if not perfect_wins.empty:
                perfect_row = perfect_wins.iloc[0]
                facts.append(
                    f"{perfect_row['manager']} set a perfect lineup (optimal = actual) "
                    f"in Week {int(perfect_row['week'])}, {int(perfect_row['year'])}"
                )

        # Highest combined score in a matchup
        if "total_matchup_score" in matchup_df.columns and "week" in matchup_df.columns:
            max_combined_row = matchup_df.loc[
                matchup_df["total_matchup_score"].idxmax()
            ]
            facts.append(
                f"{max_combined_row['manager']} and {max_combined_row['opponent']} combined for "
                f"{max_combined_row['total_matchup_score']:.2f} points in Week {int(max_combined_row['week'])}, "
                f"{int(max_combined_row['year'])} - the highest scoring matchup"
            )

        # Most dominant week
        if (
            "teams_beat_this_week" in matchup_df.columns
            and "margin" in matchup_df.columns
            and "week" in matchup_df.columns
        ):
            matchup_df["dominance_score"] = (
                matchup_df["teams_beat_this_week"] * matchup_df["margin"]
            )
            if matchup_df["dominance_score"].max() > 0:
                max_dominance_row = matchup_df.loc[
                    matchup_df["dominance_score"].idxmax()
                ]
                facts.append(
                    f"{max_dominance_row['manager']} dominated Week {int(max_dominance_row['week'])}, "
                    f"{int(max_dominance_row['year'])} - beating {int(max_dominance_row['teams_beat_this_week'])} teams "
                    f"with a margin of {max_dominance_row['margin']:.2f} points"
                )

        # Average points per manager (all-time stat)
        if "manager" in matchup_df.columns and "team_points" in matchup_df.columns:
            avg_by_manager = (
                matchup_df.groupby("manager")["team_points"]
                .mean()
                .sort_values(ascending=False)
            )
            if not avg_by_manager.empty:
                top_manager = avg_by_manager.index[0]
                top_avg = avg_by_manager.iloc[0]
                facts.append(
                    f"{top_manager} has the highest average weekly score at {top_avg:.2f} points per game across all matchups"
                )

        # Most consistent scorer (lowest std dev)
        if "manager" in matchup_df.columns and "team_points" in matchup_df.columns:
            std_by_manager = (
                matchup_df.groupby("manager")["team_points"].std().sort_values()
            )
            if not std_by_manager.empty:
                most_consistent = std_by_manager.index[0]
                consistency = std_by_manager.iloc[0]
                facts.append(
                    f"{most_consistent} is the most consistent scorer with a standard deviation of only {consistency:.2f} points"
                )

        # Longest winning streak
        if (
            "manager" in matchup_df.columns
            and "win" in matchup_df.columns
            and "year" in matchup_df.columns
            and "week" in matchup_df.columns
        ):
            sorted_df = matchup_df.sort_values(["manager", "year", "week"])
            sorted_df["streak"] = sorted_df.groupby("manager")["win"].apply(
                lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
            )
            max_streak_row = sorted_df.loc[sorted_df["streak"].idxmax()]
            if max_streak_row["win"]:
                facts.append(
                    f"{max_streak_row['manager']} had a {int(max_streak_row['streak'])}-game winning streak "
                    f"ending in Week {int(max_streak_row['week'])}, {int(max_streak_row['year'])}"
                )

    except Exception:
        # If any calculation fails, continue with what we have
        pass

    # Add generic facts if we don't have enough
    if len(facts) < 5:
        facts.extend(
            [
                "Weekly matchups can swing on a single player's performance",
                "Optimal lineups show what could have been with perfect decisions",
                "Track your performance against projections to improve your strategy",
                "Power ratings help predict future matchup outcomes",
            ]
        )

    return facts[:15]


def get_season_fun_facts(matchup_df: pd.DataFrame) -> List[str]:
    """
    Generate fun facts based on season-aggregated data.

    Args:
        matchup_df: DataFrame with matchup data (will be aggregated by season)

    Returns:
        List of fun fact strings
    """
    facts = []

    try:
        # Aggregate to season level first
        season_df = (
            matchup_df.groupby(["manager", "year"])
            .agg(
                {
                    "team_points": "sum",
                    "opponent_points": "sum",
                    "win": "sum",
                    "margin": "sum",
                }
            )
            .reset_index()
        )

        # Highest single-season total points
        if len(season_df) > 0:
            max_score_row = season_df.loc[season_df["team_points"].idxmax()]
            facts.append(
                f"The highest single-season total was {max_score_row['team_points']:.2f} points "
                f"by {max_score_row['manager']} in {int(max_score_row['year'])}"
            )

        # Best season record
        if len(season_df) > 0:
            best_record_row = season_df.loc[season_df["win"].idxmax()]
            facts.append(
                f"{best_record_row['manager']} had the best season record with {int(best_record_row['win'])} wins "
                f"in {int(best_record_row['year'])}"
            )

        # Highest season margin
        if len(season_df) > 0:
            best_margin_row = season_df.loc[season_df["margin"].idxmax()]
            facts.append(
                f"The biggest season point differential was {best_margin_row['margin']:.2f} "
                f"by {best_margin_row['manager']} in {int(best_margin_row['year'])}"
            )

        # Worst season margin
        if len(season_df) > 0:
            worst_margin_row = season_df.loc[season_df["margin"].idxmin()]
            facts.append(
                f"{worst_margin_row['manager']} had the toughest season with a {worst_margin_row['margin']:.2f} "
                f"point differential in {int(worst_margin_row['year'])}"
            )

        # Most consistent manager (lowest std dev across seasons)
        if "manager" in season_df.columns and "team_points" in season_df.columns:
            manager_seasons = season_df.groupby("manager")["team_points"].agg(
                ["mean", "std", "count"]
            )
            qualified = manager_seasons[
                manager_seasons["count"] >= 3
            ]  # At least 3 seasons
            if not qualified.empty:
                most_consistent = qualified["std"].idxmin()
                consistency = qualified.loc[most_consistent, "std"]
                facts.append(
                    f"{most_consistent} is the most consistent across seasons with a standard deviation of {consistency:.2f} points"
                )

        # Highest average season points
        if "manager" in season_df.columns:
            avg_by_manager = (
                season_df.groupby("manager")["team_points"]
                .mean()
                .sort_values(ascending=False)
            )
            if not avg_by_manager.empty:
                top_manager = avg_by_manager.index[0]
                top_avg = avg_by_manager.iloc[0]
                facts.append(
                    f"{top_manager} has the highest average season total at {top_avg:.2f} points per season"
                )

        # Most wins in a single season
        if "win" in season_df.columns:
            max_wins = season_df["win"].max()
            if max_wins > 0:
                facts.append(f"The most wins in a single season is {int(max_wins)}")

    except Exception:
        # If any calculation fails, continue with what we have
        pass

    # Add generic facts if we don't have enough
    if len(facts) < 5:
        facts.extend(
            [
                "Season totals reveal which managers dominated specific years",
                "Comparing seasons shows the evolution of team performance",
                "Championship seasons often correlate with high point totals",
                "Season-long consistency separates contenders from pretenders",
            ]
        )

    return facts[:15]


def get_career_fun_facts(matchup_df: pd.DataFrame) -> List[str]:
    """
    Generate fun facts based on all-time career data.

    Args:
        matchup_df: DataFrame with all matchup data (will be aggregated by manager)

    Returns:
        List of fun fact strings
    """
    facts = []

    try:
        # Aggregate to career/all-time level first
        career_df = (
            matchup_df.groupby("manager")
            .agg(
                {
                    "team_points": "sum",
                    "opponent_points": "sum",
                    "win": "sum",
                    "loss": "sum",
                    "margin": "sum",
                }
            )
            .reset_index()
        )

        # Add games played
        career_df["games"] = career_df["win"] + career_df["loss"]
        career_df["win_pct"] = career_df["win"] / career_df["games"]
        career_df["avg_points"] = career_df["team_points"] / career_df["games"]

        # All-time points leader
        if len(career_df) > 0:
            points_leader = career_df.loc[career_df["team_points"].idxmax()]
            facts.append(
                f"{points_leader['manager']} is the all-time points leader with {points_leader['team_points']:.2f} "
                f"total points across {int(points_leader['games'])} games"
            )

        # All-time wins leader
        if len(career_df) > 0:
            wins_leader = career_df.loc[career_df["win"].idxmax()]
            facts.append(
                f"{wins_leader['manager']} has the most all-time wins with {int(wins_leader['win'])} victories"
            )

        # Best win percentage (minimum 50 games)
        if len(career_df) > 0:
            qualified = career_df[career_df["games"] >= 50]
            if not qualified.empty:
                best_pct = qualified.loc[qualified["win_pct"].idxmax()]
                facts.append(
                    f"{best_pct['manager']} has the best all-time win percentage at {best_pct['win_pct']:.1%} (min 50 games)"
                )

        # Highest average points per game
        if len(career_df) > 0:
            qualified = career_df[career_df["games"] >= 50]
            if not qualified.empty:
                best_avg = qualified.loc[qualified["avg_points"].idxmax()]
                facts.append(
                    f"{best_avg['manager']} averages the most points per game all-time at {best_avg['avg_points']:.2f} PPG"
                )

        # Biggest all-time margin
        if len(career_df) > 0:
            best_margin = career_df.loc[career_df["margin"].idxmax()]
            facts.append(
                f"{best_margin['manager']} has the best all-time point differential at +{best_margin['margin']:.2f}"
            )

        # Most games played
        if len(career_df) > 0:
            most_games = career_df.loc[career_df["games"].idxmax()]
            facts.append(
                f"{most_games['manager']} is the all-time games played leader with {int(most_games['games'])} games"
            )

        # Total league points scored
        total_points = career_df["team_points"].sum()
        facts.append(
            f"The league has scored {total_points:,.0f} total points across all-time matchups"
        )

    except Exception:
        # If any calculation fails, continue with what we have
        pass

    # Add generic facts if we don't have enough
    if len(facts) < 5:
        facts.extend(
            [
                "Career stats reveal the true powerhouses of the league",
                "All-time records show sustained excellence over many seasons",
                "Long-term trends separate lucky years from dynasty builders",
                "Head-to-head records tell the story of epic rivalries",
            ]
        )

    return facts[:15]
