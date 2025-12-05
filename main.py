#!/usr/bin/env python3
"""
Fantasy Football Analytics - Main Streamlit Application

This is the main entry point for the Streamlit app. It handles:
- Page routing (landing, analytics, register)
- OAuth callbacks
- Session state management

Most functionality is now modularized into streamlit_helpers/:
- auth.py: OAuth authentication
- yahoo_api.py: Yahoo Fantasy API calls
- database.py: MotherDuck operations
- import_flow.py: Import management
- ui_components.py: Reusable UI components
"""
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import uuid
from typing import Optional
import subprocess
import sys
import re

# Import LeagueContext for creating context files
try:
    from fantasy_football_data_scripts.multi_league.core.league_context import LeagueContext
except (ImportError, KeyError, ModuleNotFoundError):
    LeagueContext = None

try:
    import streamlit as st
except ImportError as e:
    raise ImportError("Missing dependency: streamlit. Install with `pip install streamlit`") from e

try:
    import requests
except ImportError as e:
    raise ImportError("Missing dependency: requests. Install with `pip install requests`") from e

try:
    import duckdb
except ImportError as e:
    raise ImportError("Missing dependency: duckdb. Install with `pip install duckdb`") from e

# Import modularized helpers
from streamlit_helpers.auth import (
    build_authorize_url,
    exchange_code_for_tokens,
    save_oauth_token,
    save_token_to_motherduck,
    CLIENT_ID,
    CLIENT_SECRET,
    OAUTH_DIR,
)
from streamlit_helpers.yahoo_api import (
    yahoo_api_call,
    get_user_games,
    get_user_football_leagues,
    get_league_teams,
    get_league_teams_all_years,
    find_hidden_managers,
    extract_football_games,
    seasons_for_league_name,
    get_all_leagues_with_years,
)
from streamlit_helpers.database import (
    format_league_display_name,
    sanitize_league_name_for_db,
    get_existing_league_databases,
    get_private_leagues,
    validate_league_database,
    create_import_job_in_motherduck,
    get_job_status,
    get_motherduck_progress,
    upload_to_motherduck,
)
from streamlit_helpers.import_flow import (
    can_start_import as _can_start_import,
    mark_import_started as _mark_import_started,
    get_data_templates,
    get_column_aliases,
    detect_data_type_from_filename,
    extract_year_from_filename,
    normalize_columns,
    clean_dataframe,
    validate_dataframe,
    check_year_conflicts,
    check_duplicate_uploads,
)
from streamlit_helpers.ui_components import (
    load_custom_css,
    render_hero,
    render_feature_card,
    render_status_badge,
    render_job_card,
    render_timeline,
)
from streamlit_helpers.keeper_rules_ui import render_keeper_rules_ui


# =========================
# Import Protection (wrappers for session state)
# =========================
def can_start_import(league_name: str = None) -> tuple[bool, str]:
    """Check if a new import can be started. Wrapper for session state."""
    return _can_start_import(st.session_state, league_name=league_name)


def mark_import_started():
    """Mark that an import has been started. Wrapper for session state."""
    _mark_import_started(st.session_state)


# =========================
# Constants and Paths
# =========================
MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN") or st.secrets.get("MOTHERDUCK_TOKEN", "")
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "fantasy_football_data"
SCRIPTS_DIR = ROOT_DIR / "fantasy_football_data_scripts"
INITIAL_IMPORT_SCRIPT = SCRIPTS_DIR / "initial_import_v2.py"


# =========================
# External Data Files Configuration UI
# =========================
def render_external_data_ui(yahoo_years: list[int] = None) -> Optional[dict]:
    """
    Render the external data files configuration UI.
    Returns a dict with uploaded file data if any files uploaded.

    Args:
        yahoo_years: List of years that Yahoo API will fetch data for.
                     External uploads for these years will be blocked.

    Supports:
    - CSV, Excel (.xlsx), Parquet files for tabular data
    - JSON files for league settings
    - Auto-detection of data type from filename
    - Comprehensive column name aliases (owner->manager, season->year, etc.)
    - Data validation and cleaning
    - Duplicate detection
    - Year conflict blocking (can't upload years Yahoo covers)
    """
    import pandas as pd
    import io

    yahoo_years = yahoo_years or []

    st.markdown("### Import Historical Data Files")
    st.caption("Upload data from years BEFORE your Yahoo league history. Files are merged with Yahoo data.")

    # Show what years are blocked
    if yahoo_years:
        st.info(f"Yahoo will fetch data for: **{min(yahoo_years)}-{max(yahoo_years)}**. "
                f"You can only upload data for years before {min(yahoo_years)}.")

    templates = get_data_templates()
    column_aliases = get_column_aliases()

    # Initialize session state for uploaded files
    if "external_uploaded_files" not in st.session_state:
        st.session_state.external_uploaded_files = {}

    uploaded_files = st.session_state.external_uploaded_files

    # Template download section
    with st.expander("Download Templates", expanded=False):
        st.markdown("Download CSV templates for each data type:")

        # Filter out settings (JSON-only) from CSV templates
        csv_templates = {k: v for k, v in templates.items() if k != "settings"}
        cols = st.columns(len(csv_templates))
        for i, (data_type, template) in enumerate(csv_templates.items()):
            with cols[i]:
                # Create template CSV
                all_cols = template["required_columns"] + template["optional_columns"]
                template_df = pd.DataFrame([template["example_row"]])
                # Ensure all columns exist
                for col in all_cols:
                    if col not in template_df.columns:
                        template_df[col] = ""
                template_df = template_df[all_cols]

                csv_buffer = io.StringIO()
                template_df.to_csv(csv_buffer, index=False)

                st.download_button(
                    f"{data_type.title()}",
                    csv_buffer.getvalue(),
                    file_name=f"{data_type}_template.csv",
                    mime="text/csv",
                    key=f"download_{data_type}_template",
                    use_container_width=True
                )
                st.caption(template["description"])

    # Show accepted column aliases
    with st.expander("Column Name Aliases", expanded=False):
        st.markdown("Your files can use any of these column names (they'll be auto-converted):")
        st.markdown("""
        | Canonical Name | Accepted Aliases |
        |---------------|------------------|
        | `manager` | owner, Owner, manager_name, team_owner, user, username |
        | `player` | player_name, Player, Player Name, athlete, name |
        | `year` | season, Season, Year, fantasy_year, league_year |
        | `week` | Week, week_num, week_number, wk, gameweek, scoring_period |
        | `points` | Points, fantasy_points, fpts, pts, score, actual |
        | `team_name` | team, Team, Team Name, fantasy_team, franchise |
        | `team_points` | team_score, my_points, points_for, PF, your_score |
        | `opponent` | Opponent, opp, opponent_name, vs, against |
        | `opponent_points` | opp_points, opponent_score, points_against, PA |
        | `win` | Win, W, won, victory |
        | `loss` | Loss, L, lost, defeat |
        | `round` | Round, draft_round, rd, rnd |
        | `pick` | Pick, draft_pick, overall_pick, selection, slot |
        | `cost` | price, auction_price, salary, bid, amount, value |
        | `transaction_type` | type, action, move, activity |
        """)

    st.markdown("---")

    st.markdown("#### Upload Your Data Files")
    st.caption("Supports CSV, Excel (.xlsx/.xls), Parquet, and JSON files. Files are auto-categorized by name pattern.")

    # Smart upload - auto-detect file type
    with st.expander("Upload Files", expanded=True):
        st.markdown("Upload any files and we'll auto-detect the data type from the filename.")
        st.caption("Examples: `league_settings_2014.json`, `matchup_2013.csv`, `draft_results.xlsx`")

        smart_files = st.file_uploader(
            "Upload files",
            type=["csv", "xlsx", "xls", "parquet", "json"],
            accept_multiple_files=True,
            key="smart_upload",
            label_visibility="collapsed"
        )

        # Track files that need user input
        files_needing_type = []
        files_needing_year = []

        if smart_files:
            for file in smart_files:
                # Skip if already processed
                file_key = f"{file.name}_{file.size}"
                already_processed = any(
                    f.get("_file_key") == file_key
                    for files in uploaded_files.values()
                    for f in files
                )
                if already_processed:
                    continue

                try:
                    # Auto-detect data type from filename
                    detected_type = detect_data_type_from_filename(file.name)

                    # Try to extract year from filename
                    filename_year = extract_year_from_filename(file.name)

                    if file.name.endswith('.json'):
                        # Handle JSON files (league settings)
                        file_content = file.read()
                        file.seek(0)  # Reset for potential re-read
                        json_data = json.loads(file_content)

                        # Default to settings for JSON files
                        data_type = detected_type or "settings"

                        # Extract year from JSON or filename
                        year = json_data.get("year") or json_data.get("season") or filename_year

                        if year:
                            year = int(year)
                            # Check if year conflicts with Yahoo data
                            if year in yahoo_years:
                                st.error(f"{file.name}: Year {year} is already covered by Yahoo data. Cannot upload.")
                                continue

                        if not year:
                            # Store file for user input
                            files_needing_year.append({
                                "file": file,
                                "filename": file.name,
                                "data": json_data,
                                "data_type": data_type,
                                "file_type": "json",
                                "_file_key": file_key
                            })
                            continue

                        if data_type not in uploaded_files:
                            uploaded_files[data_type] = []

                        uploaded_files[data_type].append({
                            "filename": file.name,
                            "data": json_data,
                            "file_type": "json",
                            "year": year,
                            "row_count": 1,
                            "_file_key": file_key
                        })

                        st.success(f"{file.name}: League settings for {year} loaded")

                    else:
                        # Handle tabular files (CSV, Excel, Parquet)
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        elif file.name.endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(file)
                        elif file.name.endswith('.parquet'):
                            df = pd.read_parquet(file)
                        else:
                            st.error(f"Unsupported file type: {file.name}")
                            continue

                        # Clean and normalize
                        df = clean_dataframe(df)
                        df = normalize_columns(df, column_aliases)

                        # Determine data type
                        data_type = detected_type

                        if not data_type:
                            # Try to infer from columns
                            cols_set = set(df.columns)
                            if "transaction_type" in cols_set:
                                data_type = "transactions"
                            elif "round" in cols_set and "pick" in cols_set:
                                data_type = "draft"
                            elif "team_points" in cols_set and "opponent_points" in cols_set:
                                data_type = "matchup"
                            elif "player" in cols_set and "points" in cols_set:
                                data_type = "player"
                            elif "opponent" in cols_set:
                                data_type = "schedule"

                        if not data_type:
                            # Store file for user input
                            files_needing_type.append({
                                "file": file,
                                "filename": file.name,
                                "df": df,
                                "file_type": "tabular",
                                "filename_year": filename_year,
                                "_file_key": file_key
                            })
                            continue

                        # Check if year is in data or filename
                        has_year_column = "year" in df.columns
                        year = filename_year

                        if not has_year_column and not year:
                            # Store file for user input
                            files_needing_year.append({
                                "file": file,
                                "filename": file.name,
                                "df": df,
                                "data_type": data_type,
                                "file_type": "tabular",
                                "_file_key": file_key
                            })
                            continue

                        # Add year column if missing but extracted from filename
                        if not has_year_column and year:
                            df["year"] = year

                        # Check for year conflicts with Yahoo data
                        if "year" in df.columns and yahoo_years:
                            df_years = df["year"].dropna().unique().tolist()
                            try:
                                df_years = [int(y) for y in df_years]
                            except (ValueError, TypeError):
                                pass

                            conflict_check = check_year_conflicts(df_years, yahoo_years)
                            if conflict_check["has_conflicts"]:
                                blocked = conflict_check["blocked"]
                                st.error(f"{file.name}: Contains data for years {blocked} which are already covered by Yahoo. "
                                        f"Please remove those years from your file.")
                                continue

                        # Validate the data
                        validation = validate_dataframe(df, data_type, templates)

                        if not validation["valid"]:
                            for err in validation["errors"]:
                                st.error(f"{file.name}: {err}")
                            continue

                        # Show warnings
                        for warn in validation["warnings"]:
                            st.warning(f"{file.name}: {warn}")

                        if data_type not in uploaded_files:
                            uploaded_files[data_type] = []

                        uploaded_files[data_type].append({
                            "filename": file.name,
                            "data": df.to_dict(orient="records"),
                            "columns": list(df.columns),
                            "file_type": "tabular",
                            "row_count": len(df),
                            "stats": validation["stats"],
                            "_file_key": file_key
                        })

                        st.success(f"{file.name}: {len(df):,} rows loaded as **{data_type}** data")

                        # Show preview with stats
                        with st.expander(f"Preview: {file.name} ({data_type})"):
                            stats = validation["stats"]
                            if "years" in stats:
                                st.caption(f"Years: {stats['years']}")
                            if "managers" in stats:
                                st.caption(f"Managers: {', '.join(str(m) for m in stats['managers'][:5])}{'...' if len(stats['managers']) > 5 else ''}")
                            st.dataframe(df.head(10), use_container_width=True)

                except Exception as e:
                    st.error(f"Error reading {file.name}: {e}")

        # Handle files that need data type selection
        if files_needing_type:
            st.markdown("---")
            st.markdown("#### Specify Data Type")
            st.caption("We couldn't auto-detect the data type for these files. Please select:")

            data_type_options = list(templates.keys())

            for i, file_info in enumerate(files_needing_type):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**{file_info['filename']}**")
                    st.caption(f"Columns: {', '.join(list(file_info['df'].columns)[:5])}...")
                with col2:
                    selected_type = st.selectbox(
                        "Data type",
                        options=data_type_options,
                        key=f"type_select_{i}_{file_info['filename']}",
                        label_visibility="collapsed"
                    )
                with col3:
                    if st.button("Add", key=f"add_type_{i}_{file_info['filename']}"):
                        df = file_info['df']
                        data_type = selected_type

                        # Check if year is needed
                        has_year = "year" in df.columns or file_info.get('filename_year')
                        if not has_year:
                            st.warning(f"Please also specify the year for {file_info['filename']} below")
                            files_needing_year.append({
                                "filename": file_info['filename'],
                                "df": df,
                                "data_type": data_type,
                                "file_type": "tabular",
                                "_file_key": file_info["_file_key"]
                            })
                        else:
                            if "year" not in df.columns and file_info.get('filename_year'):
                                df["year"] = file_info['filename_year']

                            if data_type not in uploaded_files:
                                uploaded_files[data_type] = []
                            uploaded_files[data_type].append({
                                "filename": file_info['filename'],
                                "data": df.to_dict(orient="records"),
                                "columns": list(df.columns),
                                "file_type": "tabular",
                                "row_count": len(df),
                                "_file_key": file_info["_file_key"]
                            })
                            st.success(f"Added {file_info['filename']} as {data_type}")
                            st.rerun()

        # Handle files that need year specification
        if files_needing_year:
            st.markdown("---")
            st.markdown("#### Specify Year")
            st.caption("We couldn't find the year for these files. Please specify:")

            # Generate year options - only years NOT covered by Yahoo
            current_year = datetime.now().year
            all_years = list(range(current_year + 1, 1999, -1))
            if yahoo_years:
                min_yahoo = min(yahoo_years)
                year_options = [y for y in all_years if y < min_yahoo]
                st.caption(f"Only years before {min_yahoo} are available (Yahoo covers {min_yahoo}+)")
            else:
                year_options = all_years

            for i, file_info in enumerate(files_needing_year):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**{file_info['filename']}**")
                    st.caption(f"Type: {file_info['data_type']}")
                with col2:
                    selected_year = st.selectbox(
                        "Year",
                        options=year_options,
                        key=f"year_select_{i}_{file_info['filename']}",
                        label_visibility="collapsed"
                    )
                with col3:
                    if st.button("Add", key=f"add_year_{i}_{file_info['filename']}"):
                        data_type = file_info['data_type']

                        if data_type not in uploaded_files:
                            uploaded_files[data_type] = []

                        if file_info['file_type'] == 'json':
                            uploaded_files[data_type].append({
                                "filename": file_info['filename'],
                                "data": file_info['data'],
                                "file_type": "json",
                                "year": selected_year,
                                "row_count": 1,
                                "_file_key": file_info.get("_file_key")
                            })
                        else:
                            df = file_info['df']
                            df["year"] = selected_year
                            uploaded_files[data_type].append({
                                "filename": file_info['filename'],
                                "data": df.to_dict(orient="records"),
                                "columns": list(df.columns),
                                "file_type": "tabular",
                                "row_count": len(df),
                                "_file_key": file_info.get("_file_key")
                            })

                        st.success(f"Added {file_info['filename']} for year {selected_year}")
                        st.rerun()

    # Show currently uploaded files with remove buttons
    if uploaded_files:
        st.markdown("---")
        st.markdown("#### Uploaded Files")

        # Check for duplicates
        dup_check = check_duplicate_uploads(uploaded_files)
        if dup_check["has_duplicates"]:
            for warn in dup_check["warnings"]:
                st.warning(f"Duplicate: {warn}")

        for data_type, files in uploaded_files.items():
            if not files:
                continue

            st.markdown(f"**{data_type.title()}**")

            for idx, file_info in enumerate(files):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    if data_type == "settings":
                        st.caption(f"{file_info['filename']} - Year {file_info.get('year', '?')}")
                    else:
                        stats = file_info.get("stats", {})
                        years_str = ""
                        if "years" in stats:
                            years_str = f" (Years: {min(stats['years'])}-{max(stats['years'])})"
                        elif "year" in file_info:
                            years_str = f" (Year: {file_info['year']})"
                        st.caption(f"{file_info['filename']} - {file_info.get('row_count', 0):,} rows{years_str}")
                with col2:
                    # Preview button
                    if st.button("Preview", key=f"preview_{data_type}_{idx}"):
                        st.session_state[f"show_preview_{data_type}_{idx}"] = True
                with col3:
                    # Remove button
                    if st.button("Remove", key=f"remove_{data_type}_{idx}"):
                        uploaded_files[data_type].pop(idx)
                        if not uploaded_files[data_type]:
                            del uploaded_files[data_type]
                        st.rerun()

                # Show preview if requested
                if st.session_state.get(f"show_preview_{data_type}_{idx}"):
                    if file_info["file_type"] == "json":
                        st.json(file_info["data"].get("metadata", file_info["data"]))
                    else:
                        preview_df = pd.DataFrame(file_info["data"][:10])
                        st.dataframe(preview_df, use_container_width=True)
                    if st.button("Hide", key=f"hide_preview_{data_type}_{idx}"):
                        st.session_state[f"show_preview_{data_type}_{idx}"] = False
                        st.rerun()

        # Summary
        st.markdown("---")
        st.markdown("#### Summary")
        total_files = sum(len(files) for files in uploaded_files.values())
        total_rows = sum(
            f.get("row_count", 0)
            for files in uploaded_files.values()
            for f in files
        )
        st.markdown(f"**{total_files} file(s)** uploaded, **{total_rows:,} total rows**")

        for data_type, files in uploaded_files.items():
            if data_type == "settings":
                years = sorted([f.get("year", "?") for f in files])
                st.caption(f"- {data_type.title()}: {len(files)} file(s) for years {years}")
            else:
                type_rows = sum(f.get("row_count", 0) for f in files)
                st.caption(f"- {data_type.title()}: {len(files)} file(s), {type_rows:,} rows")

        # Return data (clean up internal keys)
        clean_files = {}
        for data_type, files in uploaded_files.items():
            clean_files[data_type] = []
            for f in files:
                clean_f = {k: v for k, v in f.items() if not k.startswith("_")}
                clean_files[data_type].append(clean_f)

        return {"external_data": clean_files}

    return None


# =========================
# Hidden Manager UI (must stay in main.py due to session state)
# =========================
def render_hidden_manager_ui(hidden_teams: list[dict], all_teams: list[dict]) -> dict:
    """
    Render UI to identify hidden managers by their team names.

    Returns dict mapping team names (title case) to real manager names.
    This allows the fetchers to identify hidden managers by their team name
    since multiple managers can be --hidden-- but have unique team names.
    """
    # Group by unique team name to show years together
    unique_hidden = {}
    for team in hidden_teams:
        team_name = team.get("team_name", "Unknown")
        if team_name not in unique_hidden:
            unique_hidden[team_name] = []
        if team.get("year"):
            unique_hidden[team_name].append(team["year"])

    st.warning(f"Found {len(unique_hidden)} hidden manager(s). Please identify by team name:")

    overrides = {}

    for i, (team_name, years) in enumerate(unique_hidden.items()):
        years_sorted = sorted(set(years), reverse=True) if years else []
        years_str = ", ".join(str(y) for y in years_sorted) if years_sorted else "?"

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"**{years_str}** - {team_name}")

        with col2:
            new_name = st.text_input(
                "Manager name:",
                value="",
                key=f"hidden_mgr_{i}",
                placeholder="Enter name",
                label_visibility="collapsed"
            )
            if new_name.strip():
                # Use team name (title case) as the key since that's what
                # the fetcher falls back to for --hidden-- managers
                # This maps the team name to the real manager name
                team_name_key = str(team_name).strip().title()
                overrides[team_name_key] = new_name.strip()

    if overrides:
        st.success(f"Will rename {len(overrides)} manager(s)")
        st.caption("Mapping team names to manager names: " + ", ".join(f"{k} → {v}" for k, v in overrides.items()))

    return overrides


# =========================
# Streamlit-Aware File Collection (with session state handling)
# =========================
def collect_parquet_files(base_dir: Optional[Path] = None) -> list[Path]:
    """
    Collect parquet files in priority order:
    1. Canonical files at base_dir root (schedule.parquet, matchup.parquet, etc.)
    2. Any other parquet files in base_dir subdirectories

    Args:
        base_dir: Directory to collect from. If None, uses session state league_data_dir or DATA_DIR.
    """
    # Determine which directory to use
    if base_dir is None:
        # Try league-specific directory from session state first
        if "league_data_dir" in st.session_state and st.session_state.league_data_dir:
            base_dir = Path(st.session_state.league_data_dir)
        else:
            base_dir = DATA_DIR

    files = []
    seen = set()

    # Priority 1: Canonical files at root
    canonical_names = ["schedule.parquet", "matchup.parquet", "transactions.parquet",
                       "player.parquet", "players_by_year.parquet", "draft.parquet"]

    for name in canonical_names:
        p = base_dir / name
        if p.exists() and p.is_file():
            files.append(p)
            seen.add(p.resolve())

    # Priority 2: Subdirectories (schedule_data, matchup_data, etc.)
    if base_dir.exists():
        for subdir in ["schedule_data", "matchup_data", "transaction_data", "player_data", "draft_data"]:
            sub_path = base_dir / subdir
            if sub_path.exists() and sub_path.is_dir():
                for p in sub_path.glob("*.parquet"):
                    resolved = p.resolve()
                    if resolved not in seen:
                        files.append(p)
                        seen.add(resolved)

    # Priority 3: Any other parquet files in base_dir (non-recursive, to avoid noise)
    if base_dir.exists():
        for p in base_dir.glob("*.parquet"):
            resolved = p.resolve()
            if resolved not in seen:
                files.append(p)
                seen.add(resolved)

    return files


# =========================
# Import Runner
# =========================
def run_initial_import() -> bool:
    """Run the initial data import script"""
    if not INITIAL_IMPORT_SCRIPT.exists():
        st.error(f"❌ Initial import script not found at: {INITIAL_IMPORT_SCRIPT}")
        return False

    try:
        st.info("🚀 Starting initial data import... This may take several minutes.")

        log_placeholder = st.empty()
        status_placeholder = st.empty()

        IMPORT_LOG_DIR = DATA_DIR / "import_logs"
        IMPORT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        import_log_path = IMPORT_LOG_DIR / f"initial_import_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if MOTHERDUCK_TOKEN:
            env["MOTHERDUCK_TOKEN"] = MOTHERDUCK_TOKEN
        env["AUTO_CONFIRM"] = "1"
        env["EXPORT_DATA_DIR"] = str(DATA_DIR.resolve())

        # Prefer a per-league oauth file if we have league_info in the session; fall back to global Oauth.json
        oauth_file = OAUTH_DIR / "Oauth.json"
        try:
            if "league_info" in st.session_state and st.session_state.league_info:
                league_info = st.session_state.league_info
                league_key = league_info.get("league_key") or league_info.get("league_id") or "unknown"
                safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", str(league_key))
                per_file = OAUTH_DIR / f"Oauth_{safe_key}.json"
                if per_file.exists():
                    env["OAUTH_PATH"] = str(per_file.resolve())
                elif oauth_file.exists():
                    env["OAUTH_PATH"] = str(oauth_file.resolve())
            else:
                if oauth_file.exists():
                    env["OAUTH_PATH"] = str(oauth_file.resolve())
        except Exception:
            # If anything goes wrong, don't block the import; initial_import.py may still attempt other auth flows
            if oauth_file.exists():
                env["OAUTH_PATH"] = str(oauth_file.resolve())

        # Surface which oauth file we'll use (helps debug which token the import picks up)
        try:
            used_oauth = env.get("OAUTH_PATH")
            if used_oauth:
                status_placeholder.info(f"Using OAuth file: {used_oauth}")
            else:
                status_placeholder.info(
                    "No OAuth file set; initial_import may use other auth flows or environment variables.")
        except Exception:
            # status_placeholder may not be available in some failure branches; ignore
            pass

        # Create league context file for the import script
        context_file_path = None
        if "league_info" in st.session_state:
            league_info = st.session_state.league_info
            env["LEAGUE_NAME"] = league_info.get("name", "Unknown League")
            env["LEAGUE_KEY"] = league_info.get("league_key", "unknown")
            env["LEAGUE_SEASON"] = str(league_info.get("season", ""))
            env["LEAGUE_NUM_TEAMS"] = str(league_info.get("num_teams", ""))

            # Create LeagueContext file for initial_import_v2.py
            if LeagueContext is not None:
                try:
                    oauth_file = env.get("OAUTH_PATH", str(OAUTH_DIR / "Oauth.json"))
                    league_key = league_info.get("league_key", "unknown")
                    league_name = league_info.get("name", "Unknown League")
                    season = league_info.get("season")
                    num_teams = league_info.get("num_teams")

                    # Discover all years this league has existed
                    all_seasons = []
                    if "access_token" in st.session_state and "games_data" in st.session_state:
                        try:
                            all_games = extract_football_games(st.session_state.games_data)
                            all_seasons = seasons_for_league_name(
                                st.session_state.access_token,
                                all_games,
                                league_name
                            )
                            if all_seasons:
                                status_placeholder.info(f"Found {len(all_seasons)} seasons for '{league_name}': {', '.join(all_seasons)}")
                        except Exception as e:
                            status_placeholder.warning(f"Could not discover all seasons: {e}")

                    # Determine year range - all years the league has existed
                    if all_seasons:
                        start_year = int(min(all_seasons))
                        end_year = int(max(all_seasons))
                    else:
                        # Fallback to single season if discovery fails
                        start_year = int(season) if season else 2014
                        end_year = start_year

                    # Create league-specific data directory for isolation
                    # Sanitize league name for filesystem
                    safe_league_name = re.sub(r"[^a-zA-Z0-9_-]", "_", league_name).strip("_")
                    league_data_dir = DATA_DIR / safe_league_name

                    # Create context with league-isolated data directory
                    # Get keeper rules from session state (if configured via advanced settings)
                    keeper_rules = None
                    if "league_info" in st.session_state:
                        keeper_rules = st.session_state.league_info.get("keeper_rules")

                    ctx = LeagueContext(
                        league_id=league_key,
                        league_name=league_name,
                        oauth_file_path=oauth_file,
                        start_year=start_year,
                        end_year=end_year,  # Full history for this league
                        num_teams=int(num_teams) if num_teams else None,
                        data_directory=league_data_dir,
                        keeper_rules=keeper_rules,
                    )

                    # Save context file in the league-specific directory
                    context_file_path = league_data_dir / "league_context.json"
                    ctx.save(context_file_path)
                    status_placeholder.info(f"Created league context: {context_file_path}")
                    status_placeholder.info(f"Importing years {start_year}-{end_year} for '{league_name}'")

                    # Store league data directory in session for file collection later
                    st.session_state.league_data_dir = league_data_dir

                except Exception as e:
                    st.warning(f"Could not create league context file: {e}")
                    st.warning("The import may fail without a context file.")

        # Build command - include --context if we have a context file
        if context_file_path and context_file_path.exists():
            cmd = [sys.executable, str(INITIAL_IMPORT_SCRIPT), "--context", str(context_file_path)]
        else:
            # Fallback - try without context (will likely fail)
            cmd = [sys.executable, str(INITIAL_IMPORT_SCRIPT)]
            st.warning("Running without context file - import may fail.")

        with st.spinner("Importing league data..."):
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(ROOT_DIR)
            )

            output_lines = []
            with open(import_log_path, 'a', encoding='utf-8') as lf:
                for line in process.stdout:
                    stripped = line.rstrip('\n')
                    output_lines.append(stripped)
                    lf.write(stripped + "\n")
                    lf.flush()
                    status_placeholder.info(stripped)
                    # Show more lines in the log window (50 instead of 10)
                    log_placeholder.code('\n'.join(output_lines[-50:]))

            process.wait()

            if process.returncode == 0:
                status_placeholder.success("✅ Import finished successfully.")
                st.success("✅ Data import completed successfully!")

                # Show full log in expander for debugging
                with st.expander("📋 View Full Import Log"):
                    st.code('\n'.join(output_lines))

                return True
            else:
                status_placeholder.error(f"❌ Import failed (exit code {process.returncode}).")
                st.error(f"❌ Import failed with exit code {process.returncode}")
                st.code('\n'.join(output_lines[-100:]))  # Show more error context
                return False

    except Exception as e:
        st.error(f"❌ Error running import: {e}")
        return False


# =========================
# UI Components (session state aware - not in ui_components.py)
# =========================
def render_import_progress():
    """
    Render the import progress tracker.
    Polls MotherDuck for detailed progress, falls back to GitHub API for workflow status.
    Shows a progress bar and step-by-step status.
    """
    job_id = st.session_state.get("import_job_id")
    run_id = st.session_state.get("workflow_run_id")
    github_token = st.session_state.get("github_token_for_status")
    league_name = st.session_state.get("import_league_name", "League")

    if not job_id:
        st.info("Progress tracking unavailable. Check the workflow link above.")
        return

    # Add a refresh button and auto-refresh option
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Importing: {league_name}**")
    with col2:
        if st.button("🔄 Refresh", key="refresh_progress"):
            st.rerun()
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=False, key="auto_refresh")

    # Try to get detailed progress from MotherDuck first
    md_progress = get_motherduck_progress(job_id)

    if md_progress:
        # We have detailed progress from MotherDuck!
        status = md_progress.get('status', 'running')
        phase = md_progress.get('phase', 'starting')
        stage = md_progress.get('stage', '')
        stage_detail = md_progress.get('stage_detail', '')
        overall_pct = md_progress.get('overall_pct', 0)
        current_step = md_progress.get('current_step', 0)
        total_steps = md_progress.get('total_steps', 0)

        # Status display
        if status == 'completed':
            st.success("🎉 Import completed successfully!")
            st.session_state.import_in_progress = False
            st.session_state.pop('workflow_run_id', None)
            st.session_state.pop('import_job_id', None)
        elif status == 'failed':
            error_msg = md_progress.get('error_message', 'Unknown error')
            st.error(f"❌ Import failed: {error_msg}")
            st.session_state.import_in_progress = False
        else:
            # Show current phase and stage
            phase_display = phase.replace('_', ' ').title()
            st.info(f"⏳ **{phase_display}**: {stage_detail or stage}")

        # Progress bar with overall percentage
        st.progress(overall_pct / 100, text=f"{overall_pct:.0f}% complete")

        # Show phase breakdown
        phase_icons = {
            "settings": ("⚙️", "League Settings", 5),
            "fetchers": ("📥", "Fetching Data", 40),
            "merges": ("🔀", "Merging Data", 15),
            "transformations": ("🔧", "Transformations", 40),
            "complete": ("✅", "Complete", 0),
            "error": ("❌", "Error", 0),
        }

        phases_order = ["settings", "fetchers", "merges", "transformations"]

        with st.expander("View progress by phase", expanded=(status == 'running')):
            for p in phases_order:
                icon, label, weight = phase_icons.get(p, ("•", p, 0))

                if p == phase:
                    # Current phase
                    if current_step and total_steps:
                        st.markdown(f"⟳ **{icon} {label}** ({current_step}/{total_steps} steps)")
                    else:
                        st.markdown(f"⟳ **{icon} {label}** (in progress)")
                    if stage_detail:
                        st.caption(f"   └─ {stage_detail}")
                elif p in ["settings", "fetchers", "merges", "transformations"]:
                    # Check if phase is complete based on order
                    phase_idx = phases_order.index(p) if p in phases_order else -1
                    current_idx = phases_order.index(phase) if phase in phases_order else -1

                    if current_idx > phase_idx:
                        st.markdown(f"✓ {icon} {label}")
                    else:
                        st.markdown(f"○ {icon} {label}")

        # Auto-refresh
        if auto_refresh and status == 'running':
            import time
            time.sleep(5)  # MotherDuck updates more frequently, so poll faster
            st.rerun()

    elif run_id and github_token:
        # Fall back to GitHub API for basic status
        try:
            from streamlit_helpers.trigger_import_workflow import check_import_status

            status = check_import_status(
                user_id=job_id,
                github_token=github_token,
                run_id=run_id
            )

            if not status.get('success'):
                st.warning(f"Could not fetch status: {status.get('error', 'Unknown error')}")
                return

            run_status = status.get('status', 'unknown')
            conclusion = status.get('conclusion')

            if run_status == 'completed':
                if conclusion == 'success':
                    st.success("🎉 Import completed successfully!")
                    st.session_state.import_in_progress = False
                    st.session_state.pop('workflow_run_id', None)
                elif conclusion == 'failure':
                    st.error("❌ Import failed. Check the workflow logs for details.")
                    st.session_state.import_in_progress = False
                else:
                    st.warning(f"Import ended with status: {conclusion}")
            elif run_status == 'in_progress':
                st.info("⏳ Import in progress... (detailed progress will appear once the import script starts)")
            elif run_status == 'queued':
                st.info("📋 Import queued, waiting to start...")

            # Basic progress from GitHub (just workflow steps)
            progress_pct = status.get('progress_pct', 0)
            st.progress(progress_pct / 100, text=f"Workflow progress: {progress_pct:.0f}%")

            # Auto-refresh
            if auto_refresh and run_status in ('queued', 'in_progress'):
                import time
                time.sleep(10)
                st.rerun()

        except ImportError:
            st.warning("Progress tracking module not available.")
        except Exception as e:
            st.warning(f"Error fetching progress: {e}")
    else:
        st.info("Waiting for progress data... The import script will report progress once it starts.")


# =========================
# Main App
# =========================
def main():
    st.set_page_config(
        page_title="Fantasy Football Analytics",
        page_icon="🏈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    load_custom_css()

    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("⚠️ Service configuration error. Please contact support.")
        return

    qp = st.query_params

    # Handle OAuth callback
    if "code" in qp:
        with st.spinner("🔐 Connecting to Yahoo..."):
            try:
                token_data = exchange_code_for_tokens(qp["code"])
                st.session_state.token_data = token_data
                st.session_state.access_token = token_data.get("access_token")
                st.session_state.app_mode = "register"  # Stay in register mode after OAuth
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        return

    # Handle shareable league URLs: ?league=kmffl_2025
    league_from_url = qp.get("league")
    if league_from_url and "app_mode" not in st.session_state:
        # Validate the league exists before jumping to analytics
        existing_dbs = get_existing_league_databases()
        if league_from_url in existing_dbs:
            st.session_state.selected_league_db = league_from_url
            st.session_state.app_mode = "analytics"
        else:
            # League not found - show landing page with error
            st.warning(f"League '{format_league_display_name(league_from_url)}' not found. Please select from available leagues.")

    # Initialize app mode if not set
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "landing"

    # Landing page - choose between existing leagues or register new
    if st.session_state.app_mode == "landing":
        render_landing_page()
        return

    # Analytics mode - run the KMFFL app for selected league
    if st.session_state.app_mode == "analytics":
        run_analytics_app()
        return

    # Register mode - the current OAuth flow
    if st.session_state.app_mode == "register":
        run_register_flow()
        return


def render_open_league_card(existing_dbs: list[str]):
    """
    Render the Open Existing League card with a simple dropdown.
    Excludes private (link-only) leagues from the list.
    """
    st.markdown("**Returning User**")

    if not existing_dbs:
        st.caption("No leagues imported yet.")
        return

    # Filter out private leagues
    private_leagues = get_private_leagues()
    public_dbs = [db for db in existing_dbs if db not in private_leagues]

    if not public_dbs:
        st.caption("No public leagues available.")
        return

    # Sort by display name for alphabetical order
    existing_dbs_sorted = sorted(public_dbs, key=lambda db: format_league_display_name(db).lower())

    # Create options list with display names
    options = ["Select a league..."] + [format_league_display_name(db) for db in existing_dbs_sorted]
    db_map = {format_league_display_name(db): db for db in existing_dbs_sorted}

    selected = st.selectbox(
        "Choose league",
        options,
        label_visibility="collapsed",
        key="league_dropdown"
    )

    if selected and selected != "Select a league...":
        db_name = db_map[selected]
        if st.button("Open League", key="open_league_btn", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.selected_league_db = db_name
            st.session_state.app_mode = "analytics"
            st.query_params["league"] = db_name
            st.rerun()
    else:
        st.caption("Select a league to view your dashboard.")


def render_register_card():
    """Render the Register New League card."""
    st.markdown("**New User**")
    st.caption("Connect once — we sync your league's full history automatically.")

    # Direct OAuth redirect
    auth_url = build_authorize_url()
    st.link_button("Import From Yahoo", auth_url, type="primary", use_container_width=True)

    st.markdown("")
    st.caption("No Yahoo access? Try the demo first:")

    # Demo league button
    if st.button("👀 Preview Demo League", key="demo_league_btn", use_container_width=True):
        st.cache_data.clear()
        st.session_state.selected_league_db = "kmffl"
        st.session_state.app_mode = "analytics"
        st.query_params["league"] = "kmffl"
        st.rerun()
    st.caption("Explore the dashboard with sample data.")


def render_landing_page():
    """Render the landing page with search-based league selection and register option."""
    render_hero()

    # Load existing databases once and cache for collision detection in register flow
    with st.spinner("Loading leagues..."):
        existing_dbs = get_existing_league_databases()
        st.session_state.existing_dbs_cache = set(existing_dbs)

    # Two-column layout (stacks on mobile)
    left, right = st.columns([1, 1])

    with left:
        render_open_league_card(existing_dbs)

    with right:
        render_register_card()


def run_analytics_app():
    """Run the KMFFL analytics app for the selected league."""
    import importlib.util

    selected_db = st.session_state.get("selected_league_db")

    if not selected_db:
        st.error("No league selected. Please go back and select a league.")
        if st.button("Back to Home"):
            st.session_state.app_mode = "landing"
            st.rerun()
        return

    # Add back button in sidebar
    with st.sidebar:
        if st.button("← Back to League Selection"):
            st.session_state.app_mode = "landing"
            # Clear any cached data when switching leagues
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.markdown(f"**Current League:** {format_league_display_name(selected_db)}")

    # Set environment variable for the selected database BEFORE importing
    os.environ["SELECTED_LEAGUE_DB"] = selected_db

    # Import and run the analytics app
    try:
        # Path to the app_homepage.py file
        app_homepage_path = ROOT_DIR / "analytics_app" / "app_homepage.py"

        if not app_homepage_path.exists():
            st.error(f"Analytics app not found at: {app_homepage_path}")
            if st.button("Back to Home"):
                st.session_state.app_mode = "landing"
                st.rerun()
            return

        # Add required directories to path for the app's imports to work
        analytics_app_dir = ROOT_DIR / "analytics_app"
        if str(analytics_app_dir) not in sys.path:
            sys.path.insert(0, str(analytics_app_dir))

        # Also add the root directory for fantasy_football_data_scripts imports
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))

        # Use importlib to load the module directly from the file path
        spec = importlib.util.spec_from_file_location("app_homepage", str(app_homepage_path))
        app_module = importlib.util.module_from_spec(spec)
        sys.modules["app_homepage"] = app_module
        spec.loader.exec_module(app_module)

        # Run the main function
        app_module.main()

    except Exception as e:
        st.error(f"Error running analytics app: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        if st.button("Back to Home", key="back_home_error"):
            st.session_state.app_mode = "landing"
            st.rerun()


def run_register_flow():
    """Run the registration/OAuth flow for new leagues."""
    # Add back button
    if st.button("← Back to Home"):
        st.session_state.app_mode = "landing"
        st.session_state.import_in_progress = False  # Reset import state when going back
        st.rerun()

    # Main application (original OAuth flow)
    if "access_token" in st.session_state:
        render_hero()

        # Load games
        if "games_data" not in st.session_state:
            with st.spinner("Loading your leagues..."):
                try:
                    st.session_state.games_data = get_user_games(st.session_state.access_token)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

        football_games = extract_football_games(st.session_state.games_data)

        if not football_games:
            st.warning("No fantasy football leagues found.")
            return

        # League selection - show all leagues with year ranges
        st.markdown("### Select Your League")

        # Load all leagues with their year ranges (cached)
        if "all_leagues_with_years" not in st.session_state:
            with st.spinner("Loading your leagues..."):
                try:
                    st.session_state.all_leagues_with_years = get_all_leagues_with_years(
                        st.session_state.access_token, football_games
                    )
                except Exception as e:
                    st.error(f"Error loading leagues: {e}")
                    return

        all_leagues = st.session_state.get("all_leagues_with_years", [])

        if not all_leagues:
            st.warning("No leagues found.")
            return

        # Show leagues with year ranges like "KMFFL (2015-2025)"
        league_display_names = [l["display_name"] for l in all_leagues]
        selected_display = st.radio("", league_display_names, label_visibility="collapsed")

        if selected_display:
            # Find the selected league
            selected_idx = league_display_names.index(selected_display)
            league_data = all_leagues[selected_idx]

            # Build selected_league dict for compatibility with existing code
            selected_league = {
                "league_key": league_data["latest_league_key"],
                "name": league_data["name"],
                "num_teams": league_data["num_teams"],
                "season": league_data["latest_season"],
                "years": league_data["years"],  # All years this league exists
            }

            st.session_state.selected_league = selected_league

            # Check if import can be started (pass league name for server-side check)
            can_import, block_reason = can_start_import(league_name=selected_league.get('name'))

            if not can_import:
                # Show status instead of buttons
                st.warning(f"⏳ {block_reason}")
                if st.session_state.get("import_job_id"):
                    st.info(f"Job ID: `{st.session_state.import_job_id}`")
            else:
                # Show URL preview (check for collision using cached db list)
                db_name = sanitize_league_name_for_db(selected_league['name'])
                existing_dbs_cache = st.session_state.get("existing_dbs_cache", set())

                if db_name in existing_dbs_cache:
                    # Collision - use hashed name
                    league_id = selected_league.get("league_key", "")
                    league_id_hash = hashlib.md5(league_id.encode()).hexdigest()[:6]
                    db_name = f"{db_name}_{league_id_hash}"

                league_url = f"https://leaguehistory.streamlit.app/?league={db_name}"
                st.caption("📎 Share this link with your league:")
                st.code(league_url, language=None)

                st.markdown("##### Optional Settings")

                # Privacy setting
                is_private = st.checkbox(
                    "🔒 Make league private (direct link only)",
                    value=False,
                    key="league_private",
                )
                st.session_state.configured_is_private = is_private

                # Hidden Manager Detection
                cache_key = f"teams_all_years_{selected_league.get('name')}"
                if cache_key not in st.session_state:
                    with st.spinner("Checking for hidden managers..."):
                        teams = get_league_teams_all_years(
                            st.session_state.access_token,
                            selected_league.get("name"),
                            st.session_state.get("games_data", {})
                        )
                        st.session_state[cache_key] = teams

                teams = st.session_state.get(cache_key, [])
                hidden_teams = find_hidden_managers(teams)

                if hidden_teams:
                    unique_hidden = set(t.get("team_name") for t in hidden_teams)
                    with st.expander(f"👤 Hidden Managers ({len(unique_hidden)}) — Review", expanded=False):
                        st.caption("Some managers have hidden profiles. Match them to team names.")
                        manager_overrides = render_hidden_manager_ui(hidden_teams, teams)
                        st.session_state.configured_manager_overrides = manager_overrides
                else:
                    st.session_state.configured_manager_overrides = {}

                # Keeper Rules Tab
                with st.expander("🏆 Keeper Rules", expanded=False):
                    st.caption("Configure custom keeper pricing & max years.")
                    keeper_rules = render_keeper_rules_ui()
                    st.session_state.configured_keeper_rules = keeper_rules

                # External Data Files Tab
                with st.expander("📁 Import Historical Data", expanded=False):
                    st.caption("Add data from ESPN, older seasons, or other sources.")
                    # Clear uploaded files if league changed
                    current_league_for_uploads = st.session_state.get("_external_uploads_league")
                    if current_league_for_uploads != selected_league.get("name"):
                        st.session_state.external_uploaded_files = {}
                        st.session_state._external_uploads_league = selected_league.get("name")

                    # Use the years we already know from league selection
                    yahoo_years = selected_league.get("years", [])
                    st.session_state[f"yahoo_years_{selected_league.get('name')}"] = yahoo_years

                    external_data = render_external_data_ui(yahoo_years=yahoo_years)
                    st.session_state.configured_external_data = external_data

                st.markdown("---")

                # Import button
                st.caption("Full history import • 1-2 hours • safe to close browser")
                if st.button(f"🚀 Import {selected_league['name']}", key="start_import_btn", type="primary", use_container_width=True):
                    mark_import_started()
                    league_info = {
                        "league_key": selected_league.get("league_key"),
                        "name": selected_league.get("name"),
                        "season": selected_league.get("season"),
                        "num_teams": selected_league.get("num_teams"),
                        "keeper_rules": st.session_state.get("configured_keeper_rules"),
                        "external_data": st.session_state.get("configured_external_data"),
                        "manager_name_overrides": st.session_state.get("configured_manager_overrides", {}),
                        "is_private": st.session_state.get("configured_is_private", False),
                    }
                    perform_import_flow(league_info)
                    return  # Don't continue rendering - import flow handles display

        # Job status section - show if we have an active import
        if st.session_state.get("import_job_id") or st.session_state.get("workflow_run_id"):
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 📊 Your Import Job")

            # Show progress tracker with GitHub Actions status
            render_import_progress()

            # Link to GitHub Actions
            workflow_url = st.session_state.get("workflow_run_url")
            if workflow_url:
                st.link_button(
                    "🔗 View Full Logs on GitHub",
                    workflow_url,
                    use_container_width=True
                )

    else:
        # No access token - redirect back to landing page
        # (Users should arrive here via OAuth from landing page)
        st.warning("No Yahoo connection found. Please connect your Yahoo account first.")
        if st.button("← Back to Home", key="back_no_token"):
            st.session_state.app_mode = "landing"
            st.rerun()
        return

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Made with ❤️ for fantasy football managers | Powered by MotherDuck & GitHub Actions")


def perform_import_flow(league_info: dict):
    """Trigger GitHub Actions workflow to run the import in the cloud.
    This avoids Streamlit Cloud's timeout and resource limits.
    """
    try:
        st.session_state.league_info = league_info

        # Check if we have the GitHub token to trigger the workflow
        github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_WORKFLOW_TOKEN") or st.secrets.get("GITHUB_TOKEN")

        if not github_token:
            st.error("⚠️ GitHub token not configured. Please add `GITHUB_TOKEN` to your Streamlit secrets.")
            st.info("Go to your Streamlit Cloud app settings and add `GITHUB_TOKEN` with a valid GitHub Personal Access Token.")
            return

        # Prepare league data for GitHub Actions
        # Include token_time so the yahoo_oauth library knows when the token was obtained
        raw_token = st.session_state.get("token_data", {})
        oauth_token = {
            "access_token": raw_token.get("access_token"),
            "refresh_token": raw_token.get("refresh_token"),
            "token_type": raw_token.get("token_type", "bearer"),
            "expires_in": raw_token.get("expires_in", 3600),
            "xoauth_yahoo_guid": raw_token.get("xoauth_yahoo_guid"),
            # token_time is CRITICAL for yahoo_oauth library to check expiration
            "token_time": datetime.now(timezone.utc).timestamp(),
        }

        # Discover ALL years this league has existed
        league_name = league_info.get("name", "Unknown League")
        selected_season = league_info.get("season", 2024)
        start_year = selected_season
        end_year = selected_season

        with st.spinner(f"Discovering all seasons for '{league_name}'..."):
            try:
                if "access_token" in st.session_state and "games_data" in st.session_state:
                    all_games = extract_football_games(st.session_state.games_data)
                    all_seasons = seasons_for_league_name(
                        st.session_state.access_token,
                        all_games,
                        league_name
                    )
                    if all_seasons:
                        start_year = int(min(all_seasons))
                        end_year = int(max(all_seasons))
                        st.success(f"Found {len(all_seasons)} seasons: {min(all_seasons)} - {max(all_seasons)}")
                    else:
                        st.warning(f"Could not discover seasons, importing only {selected_season}")
            except Exception as e:
                st.warning(f"Season discovery failed: {e}. Importing only {selected_season}")

        # Check if external data extends the year range
        external_data = league_info.get("external_data")
        if external_data and isinstance(external_data, dict) and "external_data" in external_data:
            # Find min/max years from external data
            for data_type, files in external_data["external_data"].items():
                if not isinstance(files, list):
                    continue
                for file_info in files:
                    if not isinstance(file_info, dict):
                        continue
                    data_rows = file_info.get("data", [])
                    if not isinstance(data_rows, list):
                        continue
                    for row in data_rows:
                        # Skip non-dict rows (could be strings from malformed data)
                        if not isinstance(row, dict):
                            continue
                        if "year" in row and row["year"]:
                            try:
                                year = int(row["year"])
                                start_year = min(start_year, year)
                                end_year = max(end_year, year)
                            except (ValueError, TypeError):
                                pass
            st.info(f"Extended year range with external data: {start_year}-{end_year}")

        # Upload external data directly to MotherDuck (bypasses GitHub Actions size limits)
        has_external_data = False
        if external_data and isinstance(external_data, dict) and "external_data" in external_data:
            with st.spinner("Uploading external data to MotherDuck..."):
                try:
                    from streamlit_helpers.database import upload_external_data_to_staging

                    upload_result = upload_external_data_to_staging(
                        external_data=external_data["external_data"],
                        db_name=league_name
                    )

                    if upload_result.get("success"):
                        has_external_data = True
                        tables = upload_result.get("tables_uploaded", [])
                        if tables:
                            st.success(f"✅ Uploaded {len(tables)} external data tables to MotherDuck staging")
                            for tbl, cnt in tables:
                                st.caption(f"  • {tbl}: {cnt:,} rows")
                    else:
                        st.warning(f"⚠️ External data upload failed: {upload_result.get('error')}")
                        st.caption("External data will be skipped. Yahoo data will still import.")
                except Exception as e:
                    st.warning(f"⚠️ Could not upload external data: {e}")
                    st.caption("External data will be skipped. Yahoo data will still import.")

        is_private = league_info.get("is_private", False)

        league_data = {
            "league_id": league_info.get("league_key") or league_info.get("league_id"),
            "league_name": league_name,
            "season": end_year,  # Most recent year
            "start_year": start_year,  # First year the league existed
            "oauth_token": oauth_token,
            "num_teams": league_info.get("num_teams", 10),
            "playoff_teams": 6,
            "regular_season_weeks": 14,
            "keeper_rules": league_info.get("keeper_rules"),  # Pass keeper rules if configured
            "has_external_data": has_external_data,  # Flag only - data is in MotherDuck staging
            "manager_name_overrides": league_info.get("manager_name_overrides", {}),  # Pass hidden manager mappings
            "is_private": is_private,  # Privacy setting - workflow will apply after determining final db_name
        }

        st.info(f"🚀 Starting import via GitHub Actions for {start_year}-{end_year}...")
        st.caption("This runs in GitHub's cloud to avoid Streamlit timeouts. Takes 60-120 minutes.")

        # Import the trigger function
        try:
            from streamlit_helpers.trigger_import_workflow import trigger_import_workflow, get_workflow_run_id

            # Record trigger time for finding the run
            trigger_time = datetime.now(timezone.utc).isoformat()

            result = trigger_import_workflow(
                league_data=league_data,
                github_token=github_token
            )

            if result['success']:
                st.success("✅ Import Started Successfully!")
                st.session_state.import_job_id = result['user_id']
                st.session_state.import_in_progress = False  # Reset since workflow is now queued
                st.session_state.import_league_name = league_data['league_name']
                st.session_state.workflow_run_url = result['workflow_run_url']

                # Try to get the actual run ID (poll for a few seconds)
                # Pass user_id to enable MotherDuck-based lookup (more reliable than timestamp)
                with st.spinner("Finding workflow run..."):
                    run_id = get_workflow_run_id(github_token, trigger_time, user_id=result['user_id'])
                    if run_id:
                        st.session_state.workflow_run_id = run_id
                        st.session_state.github_token_for_status = github_token

                st.markdown(f"""
                ### 📊 Import Job Details
                - **Job ID**: `{result['user_id']}`
                - **League**: {league_data['league_name']} ({league_data['season']})
                - **Estimated Time**: {result.get('estimated_time', '60-120 minutes')}
                - **Track Progress**: [View Workflow]({result['workflow_run_url']})
                """)

                # Show progress tracker
                render_import_progress()

                return
            else:
                st.error(f"❌ Failed to start import: {result.get('error', 'Unknown error')}")
                st.session_state.import_in_progress = False  # Reset on failure
                if 'details' in result:
                    with st.expander("Error Details"):
                        st.code(result['details'])

        except ImportError:
            st.error("❌ Workflow trigger helper not found. Please ensure streamlit_helpers/trigger_import_workflow.py exists.")
            st.session_state.import_in_progress = False  # Reset on failure

    except Exception as e:
        st.error(f"Error starting workflow: {e}")
        st.session_state.import_in_progress = False  # Reset on failure
        import traceback
        with st.expander("Full Error"):
            st.code(traceback.format_exc())


def run_local_import_fallback(league_info: dict):
    """Fallback: run import directly in Streamlit (not recommended, will likely timeout)"""
    try:
        st.session_state.league_info = league_info

        # Save OAuth token locally
        if "token_data" in st.session_state:
            try:
                saved_path = save_oauth_token(st.session_state.token_data, st.session_state.league_info)
                st.info(f"Saved OAuth token to: {saved_path}")
            except Exception:
                st.warning("Failed to write per-league OAuth file")

        ok = run_initial_import()

        if ok:
            st.success("🎉 Import finished — collecting files and uploading (if configured)...")

            files = collect_parquet_files()
            if not files:
                st.warning("⚠️ No parquet files found after import. Check the import logs.")
            else:
                st.success(f"✅ Found {len(files)} parquet file(s)")
                # Upload to MotherDuck if token available
                if MOTHERDUCK_TOKEN:
                    league_name = st.session_state.league_info.get("name", "league")
                    all_games = extract_football_games(st.session_state.get("games_data", {}))
                    season_list = seasons_for_league_name(st.session_state.access_token, all_games, league_name)
                    selected_season = str(st.session_state.league_info.get("season", "")).strip()
                    if selected_season and selected_season not in season_list:
                        season_list.append(selected_season)

                    # Use just the league name (no year suffix) - data contains all historical years
                    db_name = league_name.lower().replace(' ', '_').replace('-', '_')

                    st.write(f"**Database:** `{db_name}`")
                    uploaded = upload_to_motherduck(files, db_name, MOTHERDUCK_TOKEN)

                    if uploaded:
                        st.success("✅ Upload complete!")
                        with st.expander("📊 Upload Summary"):
                            st.write(f"**{db_name}**")
                            for tbl, cnt in uploaded:
                                st.write(f"- `public.{tbl}` → {cnt:,} rows")

        # Reset UI state (clear query params) and rerun so UI refreshes
        st.query_params.clear()
        st.button("Continue")
        st.rerun()

    except Exception as e:
        st.error(f"Error starting import: {e}")
        st.query_params.clear()
        st.rerun()


if __name__ == "__main__":
    main()
