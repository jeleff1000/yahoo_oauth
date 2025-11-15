#!/usr/bin/env python3
"""
Quick script to run the full pipeline with defensive stats and save to KMFFL directory.
"""
import shutil
from pathlib import Path
import subprocess
import sys

# KMFFL context-aware directory (where files should be saved)
KMFFL_DIR = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data\KMFFL\player_data")
CONTEXT_FILE = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data\league_context.json")

# Default directory (where merge script looks without context)
DEFAULT_DIR = Path(r"C:\Users\joeye\OneDrive\Desktop\fantasy_football_data_downloads\fantasy_football_data_scripts\fantasy_football_data\player_data")

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"✗ {description} FAILED")
        sys.exit(result.returncode)
    print(f"✓ {description} completed successfully")
    return result

def main():
    year = 2014
    week = 0

    KMFFL_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"RUNNING FRESH PIPELINE FOR {year}")
    print("="*70)
    print("All data will be downloaded fresh (not using cached files)")

    # Step 1: Fetch Yahoo fantasy data
    print("\n" + "="*70)
    print("STEP 1: Fetching Yahoo Fantasy Data")
    print("="*70)
    yahoo_script = Path(__file__).parent / "yahoo_fantasy_data.py"
    run_command(
        [sys.executable, str(yahoo_script), "--year", str(year), "--week", str(week),
         "--context", str(CONTEXT_FILE)],
        "Fetch Yahoo fantasy data"
    )

    # Step 2: Fetch NFL offense stats
    print("\n" + "="*70)
    print("STEP 2: Fetching NFL Offense Stats")
    print("="*70)
    nfl_offense_script = Path(__file__).parent / "nfl_offense_stats.py"
    run_command(
        [sys.executable, str(nfl_offense_script), "--year", str(year), "--week", str(week),
         "--context", str(CONTEXT_FILE)],
        "Fetch NFL offense stats"
    )

    # Step 3: Fetch team defensive stats from NFLverse
    print("\n" + "="*70)
    print("STEP 3: Fetching Team Defensive Stats from NFLverse")
    print("="*70)
    defense_script = Path(__file__).parent / "defense_stats.py"
    run_command(
        [sys.executable, str(defense_script), "--year", str(year), "--week", str(week),
         "--context", str(CONTEXT_FILE)],
        "Fetch team defensive stats"
    )

    # Step 4: Manually combine offense and defense stats
    print("\n" + "="*70)
    print("STEP 4: Combining Offense and Defense Stats")
    print("="*70)

    import pandas as pd

    offense_file = KMFFL_DIR / f"nfl_offense_stats_{year}_all_weeks.parquet"
    defense_file = KMFFL_DIR / f"defense_stats_{year}_all_weeks.parquet"
    nfl_file_kmffl = KMFFL_DIR / f"nfl_stats_merged_{year}_all_weeks.parquet"

    if not offense_file.exists():
        print(f"✗ ERROR: NFL offense stats not found at: {offense_file}")
        print("   The nfl_offense_stats.py script may have failed or saved to a different location.")
        sys.exit(1)

    if not defense_file.exists():
        print(f"✗ ERROR: Defense stats not found at: {defense_file}")
        print("   The defense_stats.py script may have failed.")
        sys.exit(1)

    print(f"Loading offense stats: {offense_file}")
    offense_df = pd.read_parquet(offense_file)
    print(f"  Loaded {len(offense_df):,} offense records")

    print(f"Loading defense stats: {defense_file}")
    defense_df = pd.read_parquet(defense_file)
    print(f"  Loaded {len(defense_df):,} defense records")

    # Combine the dataframes
    print("Combining offense and defense stats...")
    combined_df = pd.concat([offense_df, defense_df], ignore_index=True, sort=False)
    print(f"  Combined: {len(combined_df):,} total records")

    # Save combined file
    combined_df.to_parquet(nfl_file_kmffl, index=False)
    print(f"✓ Saved combined NFL stats to: {nfl_file_kmffl}")

    # Step 4.5: Clean player and team names before merge
    print("\n" + "="*70)
    print("STEP 4.5: Cleaning Player and Team Names")
    print("="*70)
    print("Normalizing names to improve merge matching...")
    print("  - Removing punctuation (T.Y. → ty, C.J. → cj)")
    print("  - Removing suffixes (Jr, Sr, III)")
    print("  - Removing apostrophes (Le'Veon → leveon)")
    print("  - Normalizing team names (Los Angeles → LA)")

    clean_script = Path(__file__).parent / "clean_names.py"
    run_command(
        [sys.executable, str(clean_script)],
        "Clean player and team names"
    )

    # Step 5: Copy files to default directory for merge script
    print("\n" + "="*70)
    print("STEP 5: Preparing files for merge")
    print("="*70)

    # Copy Yahoo data
    yahoo_file_kmffl = KMFFL_DIR / f"yahoo_player_stats_{year}_all_weeks.parquet"
    yahoo_file_default = DEFAULT_DIR / f"yahoo_player_stats_{year}_all_weeks.parquet"

    if not yahoo_file_kmffl.exists():
        print(f"✗ ERROR: Yahoo data not found at: {yahoo_file_kmffl}")
        sys.exit(1)

    print(f"Copying Yahoo data to default directory...")
    shutil.copy2(yahoo_file_kmffl, yahoo_file_default)
    print(f"  ✓ {yahoo_file_default}")

    # Copy NFL combined stats
    nfl_file_default = DEFAULT_DIR / f"nfl_stats_merged_{year}_all_weeks.parquet"
    print(f"Copying NFL stats to default directory...")
    shutil.copy2(nfl_file_kmffl, nfl_file_default)
    print(f"  ✓ {nfl_file_default}")

    # Step 6: Run the Yahoo-NFL merge
    print("\n" + "="*70)
    print("STEP 6: Running Yahoo-NFL Merge")
    print("="*70)
    merge_script = Path(__file__).parent / "yahoo_nfl_merge.py"
    run_command(
        [sys.executable, str(merge_script), "--year", str(year), "--week", str(week)],
        "Yahoo-NFL merge"
    )

    # Step 7: Copy merged results back to KMFFL directory
    print("\n" + "="*70)
    print("STEP 7: Copying merged results to KMFFL directory")
    print("="*70)
    week_suffix = f"week_{week}" if week > 0 else "all_weeks"
    merged_file_default = DEFAULT_DIR / f"yahoo_nfl_merged_{year}_{week_suffix}.parquet"
    merged_file_kmffl = KMFFL_DIR / f"yahoo_nfl_merged_{year}_{week_suffix}.parquet"
    merged_csv_default = DEFAULT_DIR / f"yahoo_nfl_merged_{year}_{week_suffix}.csv"
    merged_csv_kmffl = KMFFL_DIR / f"yahoo_nfl_merged_{year}_{week_suffix}.csv"

    if merged_file_default.exists():
        print(f"Copying merged parquet to KMFFL directory...")
        shutil.copy2(merged_file_default, merged_file_kmffl)
        print(f"  ✓ Saved to: {merged_file_kmffl}")
    else:
        print(f"✗ WARNING: Merged parquet file not found at {merged_file_default}")

    if merged_csv_default.exists():
        print(f"Copying merged CSV to KMFFL directory...")
        shutil.copy2(merged_csv_default, merged_csv_kmffl)
        print(f"  ✓ Saved to: {merged_csv_kmffl}")
    else:
        print(f"✗ WARNING: Merged CSV file not found at {merged_csv_default}")

    # Step 8: Show summary of merged data
    print("\n" + "="*70)
    print("MERGED DATA SUMMARY")
    print("="*70)

    if not merged_file_kmffl.exists():
        print(f"✗ ERROR: Final merged file not found at {merged_file_kmffl}")
        print("The merge may have failed. Check the merge script output above.")
        sys.exit(1)

    try:
        import pandas as pd
        df = pd.read_parquet(merged_file_kmffl)
        print(f"Total rows: {len(df):,}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Unique players: {df['player'].nunique()}")
        print(f"Weeks covered: {sorted(df['week'].unique().tolist())}")
        print(f"\nSample data (first 5 rows):")
        print(df[['player', 'position', 'week', 'points', 'nfl_team']].head(5).to_string(index=False))
    except Exception as e:
        print(f"✗ ERROR loading merged data for summary: {e}")
        sys.exit(1)

    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nFinal merged files saved to:")
    print(f"  Parquet: {merged_file_kmffl}")
    print(f"  CSV:     {merged_csv_kmffl}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
