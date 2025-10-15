import csv
import pandas as pd
from imports_and_utils import rename_players

NEW_HEADERS = [
    "year", "pick", "round", "team_key", "manager", "yahoo_player_id", "cost", "player",
    "yahoo_position", "avg_pick", "avg_round", "avg_cost", "percent_drafted",
    "is_keeper_status", "is_keeper_cost", "savings", "player_year", "manager_year", "nfl_team",
    "cost_bucket"
]

# map raw fields -> normalized output columns
COLUMN_MAP = {
    "year": "year",
    "pick": "pick",
    "round": "round",
    "team_key": "team_key",
    "player_id": "yahoo_player_id",       # <— was player_id
    "cost": "cost",
    "name_full": "player",                # <— was player_name
    "primary_position": "yahoo_position", # <— was primary_position
    "average_pick": "avg_pick",
    "average_round": "avg_round",
    "average_cost": "avg_cost",
    "percent_drafted": "percent_drafted",
    "is_keeper_status": "is_keeper_status",
    "is_keeper_cost": "is_keeper_cost",
    "editorial_team_abbr": "nfl_team",
}

def write_draft_results_to_csv(draft_data, draft_data_file, team_key_to_manager, player_id_to_team, player_id_to_name):
    # convert list[DraftResult] -> DataFrame
    draft_results_df = pd.DataFrame([result.__dict__ for result in draft_data])
    draft_results_df = rename_players(draft_results_df)
    draft_results_df = draft_results_df.rename(columns=COLUMN_MAP)

    # manager & team enrichments
    draft_results_df["manager"] = draft_results_df["team_key"].map(team_key_to_manager).fillna("N/A")
    draft_results_df["nfl_team"] = draft_results_df["yahoo_player_id"].map(player_id_to_team).fillna("N/A")

    # backfill missing player with ID->name map
    draft_results_df["player"] = draft_results_df.apply(
        lambda row: player_id_to_name.get(str(row["yahoo_player_id"]), row["player"])
        if row["player"] in ["", "N/A", None, pd.NA] else row["player"],
        axis=1
    )

    # composite keys
    draft_results_df["player_year"] = draft_results_df["player"].str.replace(" ", "", regex=False) + draft_results_df["year"].astype(str)
    draft_results_df["manager_year"] = draft_results_df["manager"].str.replace(" ", "", regex=False) + draft_results_df["year"].astype(str)

    # required columns & defaults
    draft_results_df["savings"] = draft_results_df.get("savings", "N/A")
    draft_results_df["cost_bucket"] = draft_results_df.get("cost_bucket", "")

    for col in NEW_HEADERS:
        if col not in draft_results_df.columns:
            draft_results_df[col] = ""

    draft_results_df = draft_results_df[NEW_HEADERS]
    draft_results_df.to_csv(draft_data_file, index=False)

def write_draft_analysis_to_csv(draft_analysis, file_path):
    draft_analysis_df = pd.DataFrame(draft_analysis)
    draft_analysis_df = rename_players(draft_analysis_df)

    COLUMN_MAP = {
        "year": "year",
        "player_id": "yahoo_player_id",      # <— was player_id
        "name_full": "player",               # <— was player_name
        "primary_position": "yahoo_position",# <— was primary_position
        "average_pick": "avg_pick",
        "average_round": "avg_round",
        "average_cost": "avg_cost",
        "percent_drafted": "percent_drafted",
        "is_keeper_status": "is_keeper_status",
        "is_keeper_cost": "is_keeper_cost",
        "editorial_team_abbr": "nfl_team",
    }

    draft_analysis_df = draft_analysis_df.rename(columns=COLUMN_MAP)

    for col in NEW_HEADERS:
        if col not in draft_analysis_df.columns:
            draft_analysis_df[col] = ""

    draft_analysis_df = draft_analysis_df[NEW_HEADERS]
    draft_analysis_df.to_csv(file_path, index=False)
