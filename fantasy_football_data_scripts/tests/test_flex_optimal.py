import sys
from pathlib import Path

import polars as pl

# Add scripts path so we can import the function
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "player_stats"
sys.path.append(str(SCRIPT_DIR))

from cumulative_player_stats import compute_league_wide_optimal_players  # type: ignore


def run_case(df: pl.DataFrame, roster_by_year: dict, expect_count: int | None):
    out = compute_league_wide_optimal_players(
        df,
        roster_by_year,
        year_col="year",
        week_col="week",
        position_col="position",
        points_col="points",
    )
    assert "league_wide_optimal_player" in out.columns
    if expect_count is not None:
        cnt = out.filter(pl.col("league_wide_optimal_player") == 1).height
        print("flagged:", cnt, "/", out.height)
        assert cnt == expect_count, (cnt, expect_count, out.sort(["position","points"], descending=[False, True]).to_pandas().to_string())


# Case 1: Regular positions + one FLEX W/R/T
players = [
    ("QB1", "QB", 25.0), ("QB2", "QB", 18.0),
    ("RB1", "RB", 22.0), ("RB2", "RB", 19.0), ("RB3", "RB", 10.0),
    ("WR1", "WR", 21.0), ("WR2", "WR", 17.0), ("WR3", "WR", 11.0),
    ("TE1", "TE", 15.0), ("TE2", "TE", 6.0),
]
df1 = pl.DataFrame({
    "player": [p for p, _, _ in players],
    "position": [pos for _, pos, _ in players],
    "points": [pts for _, _, pts in players],
    "year": [2024] * len(players),
    "week": [1] * len(players),
})
roster1 = {
    2024: {
        "roster_positions": [
            {"position": "QB", "count": 1},
            {"position": "RB", "count": 2},
            {"position": "WR", "count": 2},
            {"position": "TE", "count": 1},
            {"position": "W/R/T", "count": 1},
        ]
    }
}
# Expect QB1, RB1, RB2, WR1, WR2, TE1, and best remaining from WR/RB/TE -> WR3 (21,17 used; RB3 10; TE2 6) => WR3 has 11 -> but RB3 has 10 -> so WR3 (11). Total 7.
run_case(df1, roster1, expect_count=7)

# Case 2: Only regular positions, no flex
roster2 = {
    2024: {
        "roster_positions": [
            {"position": "QB", "count": 1},
            {"position": "RB", "count": 2},
            {"position": "WR", "count": 2},
            {"position": "TE", "count": 1},
        ]
    }
}
run_case(df1, roster2, expect_count=1 + 2 + 2 + 1)

# Case 3: FLEX exists but no eligible players (dataset only K/DEF)
df3 = pl.DataFrame({
    "player": ["K1", "DEF1"],
    "position": ["K", "DEF"],
    "points": [8.0, 12.0],
    "year": [2024, 2024],
    "week": [1, 1],
})
roster3 = {
    2024: {
        "roster_positions": [
            {"position": "K", "count": 1},
            {"position": "DEF", "count": 1},
            {"position": "W/R/T", "count": 1},
        ]
    }
}
# Expect just K and DEF starters -> 2
run_case(df3, roster3, expect_count=2)

# Case 4: UTIL flex keyword path
roster4 = {
    2024: {
        "roster_positions": [
            {"position": "QB", "count": 1},
            {"position": "UTIL", "count": 1},
        ]
    }
}
# With UTIL we default to WR/RB/TE; none exist, so only QB starter -> 1
run_case(df1.filter(pl.col("position") == "QB"), roster4, expect_count=1)

print("All tests passed for compute_league_wide_optimal_players edge cases.")

