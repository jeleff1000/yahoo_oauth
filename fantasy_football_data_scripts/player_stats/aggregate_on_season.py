from pathlib import Path
import duckdb
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent / "fantasy_football_data"
INPUT_WEEKLY = BASE_DIR / "player.parquet"
OUTPUT_SEASON_PARQUET = BASE_DIR / "players_by_year.parquet"
OUTPUT_SEASON_CSV = BASE_DIR / "players_by_year.csv"

# -----------------------------------------------------------------------------
# Columns to drop (combined list)
# -----------------------------------------------------------------------------
DROP_COLS = [
    # previously removed
    "points_original", "points_dst_from_yahoo_settings",
    "fantasy_points_zero_ppr", "fantasy_points_ppr", "fantasy_points_half_ppr",
    "player_key", "player_last_name_key", "position_key",
    "points_key", "year_key", "week_key",
    "team_1", "team_2",
    "rolling_point_total", "rolling_optimal_points",
    "dummy", "max_week",
    "player_last_name", "bye", "manager_year", "opponent_year",
    "fg_made_list", "fg_missed_list", "lineup_position",
    "manager_player_all_time_history_percentile",
    "manager_player_season_history_percentile",
    "manager_position_all_time_history_percentile",
    "manager_position_season_history_percentile",
    "matchup_name", "optimal_lineup_position",
    "player_personal_all_time_history_percentile",
    "player_personal_season_history_percentile",
    "position_all_time_history_percentile",
    "position_season_history_percentile",
    "computed_points", "manager_week", "player_week", "opponent_week",
    "player_all_time_history_percentile",
    "player_season_history_percentile",
    "optimal_points", "optimal_ppg", "league_wide_optimal_position",
    "keeper_price", "kept_next_year", "avg_cost_next_year",
    "total_points_next_year"
]

# -----------------------------------------------------------------------------
# Script logic
# -----------------------------------------------------------------------------
OVERWRITE_INPUT = False

def main():
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{INPUT_WEEKLY.as_posix()}')").fetchdf()
    before = len(df)

    # Filter: remove rows where points == 0 (or NaN) AND yahoo_player_id is blank
    if {"points", "yahoo_player_id"}.issubset(df.columns):
        p = pd.to_numeric(df["points"], errors="coerce").fillna(0)
        y_raw = df["yahoo_player_id"]
        y_present = y_raw.notna() & (y_raw.astype(str).str.strip() != "")
        bad = (p == 0) & (~y_present)
        df = df.loc[~bad].copy()

    # Drop all unnecessary columns that exist in df
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        print(f"Dropped columns ({len(existing_drop)}): {', '.join(existing_drop)}")

    after = len(df)
    print(f"Rows before: {before:,} | after: {after:,} | dropped: {before - after:,}")

    if OVERWRITE_INPUT:
        df.to_parquet(INPUT_WEEKLY, index=False)
        print(f"✅ Overwrote {INPUT_WEEKLY}")
    else:
        df.to_parquet(OUTPUT_SEASON_PARQUET, index=False)
        df.to_csv(OUTPUT_SEASON_CSV, index=False)
        print(f"✅ Wrote {OUTPUT_SEASON_PARQUET}")
        print(f"✅ Wrote {OUTPUT_SEASON_CSV}")

if __name__ == "__main__":
    main()
