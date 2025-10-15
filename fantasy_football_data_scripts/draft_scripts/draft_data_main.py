import os
from datetime import datetime
import pandas as pd
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
from fetch_data import fetch_draft_data, fetch_team_and_player_mappings, fetch_draft_analysis
from write_data import write_draft_results_to_csv, write_draft_analysis_to_csv
from merge_data import merge_csvs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "fantasy_football_data", "draft_data")
)

def fetch_and_write_year(oauth, gm, year, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    league_ids = gm.league_ids(year=year)
    if not league_ids:
        print(f"No league IDs found for year {year}. Exiting.")
        return None, None, None
    league_id = league_ids[-1]
    team_key_to_manager, player_id_to_name, player_id_to_team, statMapping = fetch_team_and_player_mappings(oauth, league_id)
    draft_data = fetch_draft_data(oauth, league_id)
    for data in draft_data:
        data.year = year
    draft_analysis = fetch_draft_analysis(oauth, league_id, year)
    draft_data_file = os.path.join(output_dir, f'draft_data_{year}.csv')
    draft_analysis_file = os.path.join(output_dir, f'draft_analysis_{year}.csv')
    merged_file = os.path.join(output_dir, f'merged_draft_data_{year}.csv')
    write_draft_results_to_csv(draft_data, draft_data_file, team_key_to_manager, player_id_to_team, player_id_to_name)
    print(f"Wrote draft data to {draft_data_file}")
    write_draft_analysis_to_csv(draft_analysis, draft_analysis_file)
    print(f"Wrote draft analysis to {draft_analysis_file}")
    merge_csvs(draft_data_file, draft_analysis_file, merged_file)
    print(f"Merged file written to {merged_file}")
    return draft_data_file, draft_analysis_file, merged_file

def main():
    oauth_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "oauth", "Oauth.json"
    )
    oauth_path = os.path.abspath(oauth_path)
    oauth = OAuth2(None, None, from_file=oauth_path)
    if not oauth.token_is_valid():
        oauth.refresh_access_token()
    gm = yfa.Game(oauth, 'nfl')
    year = int(input("Select the year to get data for (type 0 for all years available): "))
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    if year == 0:
        years = []
        merged_files = []
        current_year = datetime.now().year
        while True:
            files = fetch_and_write_year(oauth, gm, current_year, output_dir)
            if files == (None, None, None):
                break
            _, _, merged_file = files
            merged_files.append(merged_file)
            years.append(current_year)
            current_year -= 1
        # Combine all merged CSVs into one DataFrame
        dfs = [pd.read_csv(f) for f in merged_files if f and os.path.exists(f)]
        if dfs:
            all_years_df = pd.concat(dfs, ignore_index=True)
            merged_all_csv = os.path.join(output_dir, "merged_draft_data_all_years.csv")
            merged_all_parquet = os.path.join(output_dir, "merged_draft_data_all_years.parquet")
            all_years_df.to_csv(merged_all_csv, index=False)
            all_years_df.to_parquet(merged_all_parquet, index=False)
            print(f"All years merged CSV written to {merged_all_csv}")
            print(f"All years merged Parquet written to {merged_all_parquet}")
        print("Draft, analysis, and merged CSVs written for all years.")
    else:
        fetch_and_write_year(oauth, gm, year, output_dir)
        print(f"Draft, analysis, and merged CSVs written for {year}.")

if __name__ == "__main__":
    main()