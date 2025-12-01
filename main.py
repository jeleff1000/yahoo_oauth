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
)
from streamlit_helpers.database import (
    format_league_display_name,
    get_existing_league_databases,
    validate_league_database,
    create_import_job_in_motherduck,
    get_job_status,
    get_motherduck_progress,
)
from streamlit_helpers.import_flow import (
    can_start_import as _can_start_import,
    mark_import_started as _mark_import_started,
    get_data_templates,
)
from streamlit_helpers.ui_components import (
    load_custom_css,
    render_hero,
    render_feature_card,
    render_status_badge,
    render_job_card,
    render_timeline,
)


# =========================
# Import Protection (wrappers for session state)
# =========================
def can_start_import() -> tuple[bool, str]:
    """Check if a new import can be started. Wrapper for session state."""
    return _can_start_import(st.session_state)


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
# Keeper Rules Configuration UI
# =========================
def render_keeper_rules_ui() -> Optional[dict]:
    """
    Render the keeper rules configuration UI.
    Returns a keeper_rules dict if configured, None if keepers are disabled.
    """
    st.markdown("### Keeper League Settings")

    # Enable keepers toggle
    keeper_enabled = st.toggle(
        "This is a Keeper League",
        value=st.session_state.get("keeper_enabled", False),
        key="keeper_toggle",
        help="Enable if managers can keep players from year to year"
    )
    st.session_state.keeper_enabled = keeper_enabled

    if not keeper_enabled:
        st.info("Keeper rules disabled. League will be analyzed as a redraft league.")
        return None

    st.markdown("---")

    # Step 1: Draft Type (determines subsequent options)
    st.markdown("#### 1. Draft Type")
    draft_type = st.radio(
        "What type of draft does your league use?",
        ["Auction (bid $$ on players)", "Snake (pick by round)"],
        horizontal=True,
        key="keeper_draft_type"
    )
    is_auction = draft_type.startswith("Auction")

    st.markdown("---")

    # Step 2: Basic Settings
    st.markdown("#### 2. Basic Settings")

    col1, col2 = st.columns(2)
    with col1:
        max_keepers = st.number_input(
            "Max keepers per team",
            min_value=1, max_value=15, value=3,
            key="max_keepers"
        )
    with col2:
        if is_auction:
            budget = st.number_input(
                "Auction budget ($)",
                min_value=50, max_value=1000, value=200,
                key="keeper_budget"
            )
        else:
            budget = 200  # Default for snake
            total_rounds = st.number_input(
                "Total draft rounds",
                min_value=5, max_value=25, value=15,
                key="total_rounds"
            )

    col1, col2 = st.columns(2)
    with col1:
        unlimited_years = st.checkbox("Players can be kept forever", value=False, key="unlimited_keeper_years")
    with col2:
        if unlimited_years:
            max_years = 99
            st.info("No year limit")
        else:
            max_years = st.number_input("Max years kept", min_value=1, max_value=20, value=3, key="max_keeper_years")

    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("Min keeper price ($)", min_value=0, max_value=50, value=1, key="min_keeper_price")
    with col2:
        no_max_price = st.checkbox("No max price", value=True, key="no_max_keeper_price")
        max_price = None if no_max_price else st.number_input("Max price ($)", min_value=1, max_value=500, value=100, key="max_keeper_price")

    round_to_integer = st.checkbox("Round to whole dollars", value=True, key="round_to_integer")

    st.markdown("---")

    # Step 3: First Year Keeper Price
    st.markdown("#### 3. First Year Keeper Price")

    base_cost_rules = {}

    if is_auction:
        st.caption("How is the keeper price determined the first time you keep a player?")

        first_year_source = st.selectbox(
            "Base price comes from:",
            [
                "Draft price only",
                "FAAB bid only",
                "Higher of: draft price OR (multiplier × FAAB bid)",
            ],
            key="first_year_source"
        )

        # Formula inputs: multiplier × value + flat
        st.markdown("**Adjust the base price (optional):**")
        col1, col2 = st.columns(2)
        with col1:
            base_multiplier = st.number_input(
                "Multiply by",
                min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                key="base_multiplier",
                help="1.0 = no change, 0.5 = half, 2.0 = double"
            )
        with col2:
            base_flat = st.number_input(
                "Then add ($)",
                min_value=-50.0, max_value=100.0, value=0.0, step=0.5,
                key="base_flat",
                help="Added after multiplying (can be negative)"
            )

        if first_year_source == "Higher of: draft price OR (multiplier × FAAB bid)":
            faab_multiplier = st.number_input(
                "FAAB multiplier (for comparison)",
                min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                key="faab_compare_mult",
                help="e.g., 0.5 means compare draft price to 50% of FAAB bid"
            )
            base_cost_rules["auction"] = {
                "source": "max_of_draft_faab",
                "faab_multiplier": faab_multiplier,
                "multiplier": base_multiplier,
                "flat": base_flat
            }
            base_cost_rules["faab_only"] = base_cost_rules["auction"].copy()
        elif first_year_source == "FAAB bid only":
            base_cost_rules["auction"] = {"source": "draft_price", "multiplier": base_multiplier, "flat": base_flat}
            base_cost_rules["faab_only"] = {"source": "faab_bid", "multiplier": base_multiplier, "flat": base_flat}
        else:  # Draft price only
            base_cost_rules["auction"] = {"source": "draft_price", "multiplier": base_multiplier, "flat": base_flat}
            base_cost_rules["faab_only"] = {"source": "faab_bid", "multiplier": base_multiplier, "flat": base_flat}

        base_cost_rules["free_agent"] = {"source": "fixed", "value": min_price}

        # Show formula summary
        if base_multiplier != 1.0 or base_flat != 0:
            formula_str = f"{base_multiplier}× base"
            if base_flat > 0:
                formula_str += f" + ${base_flat:.2f}"
            elif base_flat < 0:
                formula_str += f" - ${abs(base_flat):.2f}"
            st.success(f"First year price = {formula_str}")
        else:
            st.success("First year price = base price (no adjustment)")

    else:  # Snake draft
        st.caption("How is the keeper round determined?")

        snake_base = st.selectbox(
            "Keeper round based on:",
            ["Draft round (where you picked them)", "Fixed round for all keepers"],
            key="snake_base"
        )

        if snake_base == "Fixed round for all keepers":
            fixed_round = st.number_input("Fixed round", min_value=1, max_value=20, value=10, key="fixed_keeper_round")
            base_cost_rules["snake"] = {"source": "fixed", "value": fixed_round}
        else:
            base_cost_rules["snake"] = {"source": "draft_round"}

        undrafted_round = st.number_input(
            "Undrafted players kept at round:",
            min_value=1, max_value=20, value=10,
            key="undrafted_round"
        )
        base_cost_rules["undrafted"] = {"source": "fixed", "value": undrafted_round}

    st.markdown("---")

    # Step 4: Year-over-Year Escalation
    st.markdown("#### 4. Price Change Each Year Kept")

    if is_auction:
        st.caption("How does the keeper price change year to year?")

        escalation_type = st.selectbox(
            "Each year:",
            [
                "No change",
                "Add flat amount (e.g., +$5)",
                "Multiply (e.g., 1.5×)",
                "Multiply + add (e.g., 1.5× + $7.50)",
            ],
            key="escalation_type"
        )

        formulas_by_keeper_year = {}

        if escalation_type == "No change":
            formulas_by_keeper_year["1"] = {"expression": "base_cost", "description": "Base price"}
            formulas_by_keeper_year["2+"] = {"expression": "base_cost", "description": "Same each year"}

        elif escalation_type == "Add flat amount (e.g., +$5)":
            flat_inc = st.number_input("Add per year ($)", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="flat_inc")
            formulas_by_keeper_year["1"] = {"expression": "base_cost", "description": "Base price"}
            formulas_by_keeper_year["2+"] = {
                "expression": f"base_cost + {flat_inc} * (keeper_year - 1)",
                "description": f"+${flat_inc:.2f}/year",
                "flat_per_year": flat_inc
            }

        elif escalation_type == "Multiply (e.g., 1.5×)":
            mult = st.number_input("Multiply by", min_value=1.0, max_value=5.0, value=1.5, step=0.1, key="esc_mult")
            formulas_by_keeper_year["1"] = {"expression": "base_cost", "description": "Base price"}
            formulas_by_keeper_year["2+"] = {
                "expression": f"base_cost * ({mult} ** (keeper_year - 1))",
                "description": f"×{mult}/year",
                "multiplier": mult,
                "recursive": True
            }

        else:  # Multiply + add
            col1, col2 = st.columns(2)
            with col1:
                esc_mult = st.number_input("Multiply by", min_value=0.5, max_value=5.0, value=1.5, step=0.1, key="esc_mult2")
            with col2:
                esc_flat = st.number_input("Then add ($)", min_value=0.0, max_value=100.0, value=7.5, step=0.5, key="esc_flat")

            formulas_by_keeper_year["1"] = {"expression": "base_cost", "description": "Base price"}
            formulas_by_keeper_year["2+"] = {
                "expression": f"prev_cost * {esc_mult} + {esc_flat}",
                "description": f"{esc_mult}× + ${esc_flat:.2f}",
                "multiplier": esc_mult,
                "flat_add": esc_flat,
                "recursive": True
            }

    else:  # Snake draft escalation
        st.caption("How does the keeper round change year to year?")

        snake_penalty = st.selectbox(
            "Round penalty:",
            ["No penalty (same round)", "Lose 1 round per year", "Lose 2 rounds per year", "Custom"],
            key="snake_penalty_type"
        )

        if snake_penalty == "Custom":
            rounds_lost = st.number_input("Rounds lost per year", min_value=0, max_value=5, value=1, key="custom_penalty")
        else:
            rounds_lost = {"No penalty (same round)": 0, "Lose 1 round per year": 1, "Lose 2 rounds per year": 2}.get(snake_penalty, 0)

        formulas_by_keeper_year = {
            "1": {"expression": "base_round", "description": "Draft round"},
            "2+": {"expression": f"base_round - {rounds_lost} * (keeper_year - 1)", "description": f"-{rounds_lost} round/year", "rounds_lost": rounds_lost}
        }

    st.markdown("---")

    # Step 5: Live Preview
    st.markdown("#### 5. Preview")

    if is_auction:
        needs_faab = any(r.get("source") == "max_of_draft_faab" for r in base_cost_rules.values())

        if needs_faab:
            col1, col2, col3 = st.columns(3)
            with col1:
                preview_draft = st.number_input("Draft price ($)", min_value=0, max_value=200, value=8, key="prev_draft")
            with col2:
                preview_faab = st.number_input("FAAB bid ($)", min_value=0, max_value=200, value=20, key="prev_faab")
            with col3:
                preview_years = st.slider("Years", 1, min(10, max_years if max_years != 99 else 10), 5, key="prev_years")
        else:
            col1, col2 = st.columns(2)
            with col1:
                preview_draft = st.number_input("Draft price ($)", min_value=0, max_value=200, value=25, key="prev_draft")
            with col2:
                preview_years = st.slider("Years", 1, min(10, max_years if max_years != 99 else 10), 5, key="prev_years")
            preview_faab = 0

        # Calculate base cost
        rule = base_cost_rules.get("auction", {})
        if rule.get("source") == "max_of_draft_faab":
            faab_mult = rule.get("faab_multiplier", 0.5)
            base_value = max(preview_draft, faab_mult * preview_faab)
            base_note = f"MAX(${preview_draft}, {faab_mult}×${preview_faab})"
        else:
            base_value = preview_draft
            base_note = f"${preview_draft}"

        # Apply base adjustments
        mult = rule.get("multiplier", 1.0)
        flat = rule.get("flat", 0.0)
        base_cost = base_value * mult + flat
        base_cost = max(min_price, base_cost)
        if max_price:
            base_cost = min(max_price, base_cost)

        st.markdown(f"**Base: {base_note} → ${base_cost:.0f}**")

        # Year-by-year
        preview_data = []
        prev_cost = base_cost
        for yr in range(1, preview_years + 1):
            if yr == 1:
                cost = base_cost
            else:
                formula = formulas_by_keeper_year.get("2+", {})
                if formula.get("recursive"):
                    m = formula.get("multiplier", 1)
                    f = formula.get("flat_add", 0)
                    cost = prev_cost * m + f
                elif formula.get("flat_per_year"):
                    cost = base_cost + formula["flat_per_year"] * (yr - 1)
                else:
                    cost = base_cost

            cost = max(min_price, cost)
            if max_price:
                cost = min(max_price, cost)
            if round_to_integer:
                cost = round(cost)

            preview_data.append({"Year": yr, "Cost": f"${cost:.0f}", "Change": f"+${cost - prev_cost:.0f}" if yr > 1 else "-"})
            prev_cost = cost

        import pandas as pd
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)

    else:  # Snake preview
        col1, col2 = st.columns(2)
        with col1:
            preview_round = st.number_input("Draft round", min_value=1, max_value=20, value=5, key="prev_round")
        with col2:
            preview_years = st.slider("Years", 1, min(10, max_years if max_years != 99 else 10), 5, key="prev_years")

        rounds_lost = formulas_by_keeper_year.get("2+", {}).get("rounds_lost", 0)
        preview_data = []
        for yr in range(1, preview_years + 1):
            keeper_round = max(1, preview_round - rounds_lost * (yr - 1))
            preview_data.append({"Year": yr, "Round": keeper_round})

        import pandas as pd
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)

    # Build keeper_rules dict
    keeper_rules = {
        "enabled": True,
        "draft_type": "auction" if is_auction else "snake",
        "max_keepers": max_keepers,
        "max_years": max_years if max_years != 99 else None,
        "budget": budget,
        "min_price": min_price,
        "max_price": max_price,
        "round_to_integer": round_to_integer,
        "formulas_by_keeper_year": formulas_by_keeper_year,
        "base_cost_rules": base_cost_rules,
    }

    return keeper_rules


# =========================
# External Data Files Configuration UI
# =========================
def render_external_data_ui() -> Optional[dict]:
    """
    Render the external data files configuration UI.
    Returns a dict with uploaded file data if any files uploaded.
    """
    import pandas as pd
    import io

    st.markdown("### Import Historical Data Files")
    st.caption("Upload data from previous years, ESPN leagues, or other sources. Files are merged with Yahoo data.")

    templates = get_data_templates()

    # Template download section
    with st.expander("Download Templates", expanded=False):
        st.markdown("Download CSV templates for each data type:")

        cols = st.columns(len(templates))
        for i, (data_type, template) in enumerate(templates.items()):
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

    st.markdown("---")

    # File upload section
    uploaded_files = {}

    st.markdown("#### Upload Your Data Files")
    st.caption("Supports CSV, Excel (.xlsx), and Parquet files. You can upload multiple files per category.")

    for data_type, template in templates.items():
        with st.expander(f"{data_type.title()} Data", expanded=False):
            st.markdown(f"**{template['description']}**")
            st.markdown(f"Required columns: `{', '.join(template['required_columns'])}`")

            files = st.file_uploader(
                f"Upload {data_type} files",
                type=["csv", "xlsx", "parquet"],
                accept_multiple_files=True,
                key=f"upload_{data_type}",
                label_visibility="collapsed"
            )

            if files:
                uploaded_files[data_type] = []
                for file in files:
                    try:
                        # Read file based on extension
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        elif file.name.endswith('.xlsx'):
                            df = pd.read_excel(file)
                        elif file.name.endswith('.parquet'):
                            df = pd.read_parquet(file)
                        else:
                            st.error(f"Unsupported file type: {file.name}")
                            continue

                        # Validate required columns
                        missing_cols = set(template["required_columns"]) - set(df.columns)
                        if missing_cols:
                            st.warning(f"{file.name}: Missing columns: {missing_cols}")
                        else:
                            st.success(f"{file.name}: {len(df):,} rows loaded")

                            # Show preview
                            with st.expander(f"Preview: {file.name}"):
                                st.dataframe(df.head(10), use_container_width=True)

                            # Store as dict for JSON serialization
                            uploaded_files[data_type].append({
                                "filename": file.name,
                                "data": df.to_dict(orient="records"),
                                "columns": list(df.columns),
                                "row_count": len(df)
                            })

                    except Exception as e:
                        st.error(f"Error reading {file.name}: {e}")

    # Summary
    if uploaded_files:
        st.markdown("---")
        st.markdown("#### Upload Summary")
        for data_type, files in uploaded_files.items():
            total_rows = sum(f["row_count"] for f in files)
            st.markdown(f"- **{data_type.title()}**: {len(files)} file(s), {total_rows:,} total rows")

        return {"external_data": uploaded_files}

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
# MotherDuck Upload
# =========================
def _slug(s: str, lead_prefix: str) -> str:
    """Create a valid database/table name from a string"""
    x = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip().lower()).strip("_")
    if not x:
        x = "db"
    if re.match(r"^\d", x):
        x = f"{lead_prefix}_{x}"
    return x[:63]


def upload_to_motherduck(files: list[Path], db_name: str, token: str) -> list[tuple[str, int]]:
    """Upload parquet files directly to MotherDuck"""
    if not files:
        return []

    if token:
        os.environ["MOTHERDUCK_TOKEN"] = token

    db = _slug(db_name, "l")

    con = duckdb.connect("md:")
    con.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    con.execute(f"USE {db}")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS public")

    # Table name mapping (handle common aliases)
    aliases = {
        "players_by_year": "player",
        "yahoo_player_stats_multi_year_all_weeks": "player",
        "matchups": "matchup",
        "schedules": "schedule",
        "transaction": "transactions",
    }

    results = []
    for pf in files:
        stem = pf.stem.lower()
        stem = aliases.get(stem, stem)
        tbl = _slug(stem, "t")

        try:
            st.info(f"📤 Uploading {pf.name} → {db}.public.{tbl}...")
            con.execute(f"CREATE OR REPLACE TABLE public.{tbl} AS SELECT * FROM read_parquet(?)", [str(pf)])
            cnt = con.execute(f"SELECT COUNT(*) FROM public.{tbl}").fetchone()[0]
            results.append((tbl, int(cnt)))
            st.success(f"✅ {tbl}: {cnt:,} rows")
        except Exception as e:
            st.error(f"❌ Failed to upload {pf.name}: {e}")

    con.close()
    return results


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
                    ctx = LeagueContext(
                        league_id=league_key,
                        league_name=league_name,
                        oauth_file_path=oauth_file,
                        start_year=start_year,
                        end_year=end_year,  # Full history for this league
                        num_teams=int(num_teams) if num_teams else None,
                        data_directory=league_data_dir,
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


def render_landing_page():
    """Render the landing page with existing leagues dropdown and register button."""
    render_hero()

    st.markdown("### Choose Your Path")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">View Existing League</div>
            <div class="feature-desc">Select a league that's already been imported to view analytics</div>
        </div>
        """, unsafe_allow_html=True)

        # Get existing databases
        with st.spinner("Loading available leagues..."):
            existing_dbs = get_existing_league_databases()

        if existing_dbs:
            selected_db = st.selectbox(
                "Select League:",
                options=[""] + existing_dbs,
                format_func=lambda x: "-- Choose a league --" if x == "" else format_league_display_name(x),
                key="league_selector"
            )

            if selected_db:
                if st.button("View Analytics", type="primary", use_container_width=True, key="view_analytics_btn"):
                    # Store selected database in session state
                    st.session_state.selected_league_db = selected_db
                    st.session_state.app_mode = "analytics"
                    st.rerun()
        else:
            st.info("No existing leagues found. Register a new league to get started!")

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">➕</div>
            <div class="feature-title">Register New League</div>
            <div class="feature-desc">Connect your Yahoo account and import a new fantasy football league</div>
        </div>
        """, unsafe_allow_html=True)

        # Direct OAuth redirect - skip the middle page
        auth_url = build_authorize_url()
        st.link_button("Register New League", auth_url, type="secondary", use_container_width=True)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Made with ❤️ for fantasy football managers | Powered by MotherDuck & GitHub Actions")


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

    # Import and run the KMFFL app
    try:
        # Path to the app_homepage.py file
        app_homepage_path = ROOT_DIR / "KMFFLApp" / "streamlit_ui" / "app_homepage.py"

        if not app_homepage_path.exists():
            st.error(f"Analytics app not found at: {app_homepage_path}")
            if st.button("Back to Home"):
                st.session_state.app_mode = "landing"
                st.rerun()
            return

        # Add required directories to path for the app's imports to work
        streamlit_ui_dir = ROOT_DIR / "KMFFLApp" / "streamlit_ui"
        if str(streamlit_ui_dir) not in sys.path:
            sys.path.insert(0, str(streamlit_ui_dir))

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

        # League selection
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📋 Select Your League")

            season_options = {f"{g['season']} NFL Season": g['game_key'] for g in football_games}
            selected_season = st.selectbox("Season:", list(season_options.keys()), label_visibility="collapsed")

            if selected_season:
                game_key = season_options[selected_season]

                if "current_game_key" not in st.session_state or st.session_state.current_game_key != game_key:
                    with st.spinner("Loading leagues..."):
                        try:
                            leagues_data = get_user_football_leagues(st.session_state.access_token, game_key)
                            st.session_state.current_leagues = leagues_data
                            st.session_state.current_game_key = game_key
                        except Exception as e:
                            st.error(f"Error: {e}")

                if "current_leagues" in st.session_state:
                    try:
                        leagues = (
                            st.session_state.current_leagues.get("fantasy_content", {})
                            .get("users", {}).get("0", {}).get("user", [])[1]
                            .get("games", {}).get("0", {}).get("game", [])[1]
                            .get("leagues", {})
                        )
                        league_list = []
                        for key in leagues:
                            if key == "count":
                                continue
                            league = leagues[key].get("league", [])[0]
                            league_list.append({
                                "league_key": league.get("league_key"),
                                "name": league.get("name"),
                                "num_teams": league.get("num_teams"),
                                "season": league.get("season"),
                            })

                        if league_list:
                            league_names = [f"{l['name']} ({l['num_teams']} teams)" for l in league_list]
                            selected_name = st.radio("", league_names, label_visibility="collapsed")
                            selected_league = league_list[league_names.index(selected_name)]

                            # Store selected league in session for import
                            st.session_state.selected_league = selected_league

                            # Use a proper Streamlit button to preserve session state
                            # Anchor links lose session state on Streamlit Cloud!
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg,#667eea,#7f5af0); padding:1.5rem; border-radius:0.75rem; color:white; text-align:center; margin-bottom:1rem;">
                                <h2 style="margin:0;">🚀 Import {selected_league['name']}</h2>
                                <p style="margin:0.25rem 0 0.75rem; opacity:0.95;">Season {selected_league['season']} — {selected_league['num_teams']} teams</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Check if import can be started
                            can_import, block_reason = can_start_import()

                            if not can_import:
                                # Show status instead of buttons
                                st.warning(f"⏳ {block_reason}")
                                if st.session_state.get("import_job_id"):
                                    st.info(f"Job ID: `{st.session_state.import_job_id}`")
                            else:
                                # Two options: Quick Start or Advanced Settings
                                col_start, col_advanced = st.columns(2)

                                with col_start:
                                    if st.button("Start Import Now", key="start_import_btn", type="primary", use_container_width=True):
                                        mark_import_started()
                                        # Quick start - no keeper rules
                                        league_info = {
                                            "league_key": selected_league.get("league_key"),
                                            "name": selected_league.get("name"),
                                            "season": selected_league.get("season"),
                                            "num_teams": selected_league.get("num_teams"),
                                            "keeper_rules": None,
                                        }
                                        perform_import_flow(league_info)

                                with col_advanced:
                                    if st.button("Advanced Settings", key="advanced_settings_btn", type="secondary", use_container_width=True):
                                        st.session_state.show_advanced_settings = True

                            # Advanced Settings Section
                            if st.session_state.get("show_advanced_settings", False):
                                st.markdown("---")
                                st.markdown("## Advanced Settings")

                                # Hidden Manager Detection - Check automatically when opening advanced settings
                                with st.expander("Identify Hidden Managers", expanded=True):
                                    # Fetch teams across ALL years if not already done
                                    cache_key = f"teams_all_years_{selected_league.get('name')}"
                                    if cache_key not in st.session_state:
                                        with st.spinner("Checking for hidden managers across all years..."):
                                            teams = get_league_teams_all_years(
                                                st.session_state.access_token,
                                                selected_league.get("name"),
                                                st.session_state.get("games_data", {})
                                            )
                                            st.session_state[cache_key] = teams

                                    teams = st.session_state.get(cache_key, [])
                                    hidden_teams = find_hidden_managers(teams)

                                    if hidden_teams:
                                        manager_overrides = render_hidden_manager_ui(hidden_teams, teams)
                                        st.session_state.configured_manager_overrides = manager_overrides
                                    else:
                                        st.success("No hidden managers found!")
                                        st.session_state.configured_manager_overrides = {}

                                # Keeper Rules Tab
                                with st.expander("Keeper Rules", expanded=False):
                                    keeper_rules = render_keeper_rules_ui()
                                    st.session_state.configured_keeper_rules = keeper_rules

                                # External Data Files Tab
                                with st.expander("Import Historical Data (ESPN, other years, etc.)", expanded=False):
                                    external_data = render_external_data_ui()
                                    st.session_state.configured_external_data = external_data

                                st.markdown("---")

                                # Import button with advanced settings
                                can_import_adv, block_reason_adv = can_start_import()
                                if not can_import_adv:
                                    st.warning(f"⏳ {block_reason_adv}")
                                    if st.session_state.get("import_job_id"):
                                        st.info(f"Job ID: `{st.session_state.import_job_id}`")
                                elif st.button("Start Import with Settings", key="start_import_advanced_btn", type="primary", use_container_width=True):
                                    mark_import_started()
                                    league_info = {
                                        "league_key": selected_league.get("league_key"),
                                        "name": selected_league.get("name"),
                                        "season": selected_league.get("season"),
                                        "num_teams": selected_league.get("num_teams"),
                                        "keeper_rules": st.session_state.get("configured_keeper_rules"),
                                        "external_data": st.session_state.get("configured_external_data"),
                                        "manager_name_overrides": st.session_state.get("configured_manager_overrides", {}),
                                    }
                                    perform_import_flow(league_info)

                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            st.markdown("### 💡 What You'll Get")
            render_feature_card("📅", "Schedules", "All-time matchups and records")
            render_feature_card("👥", "Players", "Complete stat history")
            render_feature_card("💰", "Transactions", "Trades and pickups")
            render_feature_card("🏆", "Playoffs", "Championship data")

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
        if external_data and "external_data" in external_data:
            # Find min/max years from external data
            for data_type, files in external_data["external_data"].items():
                for file_info in files:
                    for row in file_info.get("data", []):
                        if "year" in row and row["year"]:
                            try:
                                year = int(row["year"])
                                start_year = min(start_year, year)
                                end_year = max(end_year, year)
                            except (ValueError, TypeError):
                                pass
            st.info(f"Extended year range with external data: {start_year}-{end_year}")

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
            "external_data": external_data,  # Pass external data files if uploaded
            "manager_name_overrides": league_info.get("manager_name_overrides", {}),  # Pass hidden manager mappings
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
                with st.spinner("Finding workflow run..."):
                    run_id = get_workflow_run_id(github_token, trigger_time)
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
