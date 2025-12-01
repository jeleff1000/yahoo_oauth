#!/usr/bin/env python3
from __future__ import annotations

import os
import urllib.parse
import base64
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
    # KeyError can occur during Streamlit's hot-reload due to sys.modules caching
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


# =========================
# Import Protection (prevent double-clicks)
# =========================
def can_start_import() -> tuple[bool, str]:
    """
    Check if a new import can be started.
    Returns (can_start, reason_if_blocked).
    """
    import time

    # Check if import is flagged as in progress
    if st.session_state.get("import_in_progress", False):
        return False, "Import already in progress"

    # Check cooldown - prevent rapid clicks (30 second cooldown)
    last_import_time = st.session_state.get("last_import_triggered_at", 0)
    elapsed = time.time() - last_import_time
    if elapsed < 30:
        remaining = int(30 - elapsed)
        return False, f"Please wait {remaining}s before starting another import"

    # Check if we already have a job for this league
    if st.session_state.get("import_job_id"):
        return False, "An import job was already started. Check the status below."

    return True, ""


def mark_import_started():
    """Mark that an import has been started (for cooldown tracking)."""
    import time
    st.session_state.import_in_progress = True
    st.session_state.last_import_triggered_at = time.time()


# =========================
# MotherDuck Database Discovery
# =========================
def format_league_display_name(db_name: str) -> str:
    """
    Format league database name for display.
    Strips 'l_' prefix that was added for digit-starting names.

    Example: 'l_5townsfootball' -> '5townsfootball'
    """
    if not db_name:
        return db_name
    # Strip the 'l_' prefix if it was added because name started with a digit
    if db_name.startswith("l_") and len(db_name) > 2 and db_name[2].isdigit():
        return db_name[2:]
    return db_name


def get_existing_league_databases() -> list[str]:
    """
    Discover existing league databases in MotherDuck.
    Returns a sorted list of database names (excluding system databases).
    """
    if not MOTHERDUCK_TOKEN:
        return []

    try:
        os.environ.setdefault("MOTHERDUCK_TOKEN", MOTHERDUCK_TOKEN)
        con = duckdb.connect("md:")

        # Query all databases
        result = con.execute("SHOW DATABASES").fetchall()
        con.close()

        # Filter out system databases and return sorted list
        system_dbs = {"my_db", "sample_data", "secrets", "ops", "information_schema", "md_information_schema"}
        league_dbs = [
            row[0] for row in result
            if row[0].lower() not in system_dbs
            and not row[0].startswith("_")
            and not row[0].startswith("md_")  # Exclude MotherDuck system databases
        ]

        return sorted(league_dbs, key=str.lower)
    except Exception as e:
        st.warning(f"Could not discover databases: {e}")
        return []


def validate_league_database(db_name: str) -> bool:
    """
    Validate that a database has the expected league tables (matchup, player, etc.).
    """
    if not MOTHERDUCK_TOKEN or not db_name:
        return False

    try:
        os.environ.setdefault("MOTHERDUCK_TOKEN", MOTHERDUCK_TOKEN)
        con = duckdb.connect("md:")

        # Check if required tables exist
        result = con.execute(f"SHOW TABLES IN {db_name}").fetchall()
        con.close()

        tables = {row[0].lower() for row in result}
        required_tables = {"matchup", "player"}

        return required_tables.issubset(tables)
    except Exception:
        return False


# =========================
# Keeper Rules Configuration UI
# =========================
def render_keeper_rules_ui() -> Optional[dict]:
    """
    Render the keeper rules configuration UI.
    Returns a keeper_rules dict if configured, None if keepers are disabled.

    Supports formula-based keeper price calculations matching LeagueContext schema:
    - formulas_by_keeper_year: {"1": {...}, "2+": {...}}
    - base_cost_rules: {"auction": {...}, "faab_only": {...}, "free_agent": {...}}
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

    # Step 1: Basic Settings
    st.markdown("#### 1. Basic Settings")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_keepers = st.number_input(
            "Maximum keepers per team",
            min_value=1, max_value=15, value=3,
            key="max_keepers",
            help="How many players can each team keep?"
        )
    with col2:
        max_years = st.number_input(
            "Maximum years a player can be kept",
            min_value=1, max_value=10, value=3,
            key="max_keeper_years",
            help="How many consecutive years can you keep the same player? (1 = can keep once, then must release)"
        )
    with col3:
        budget = st.number_input(
            "Auction budget",
            min_value=50, max_value=1000, value=200,
            key="keeper_budget",
            help="Total auction dollars each team has"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        min_price = st.number_input(
            "Minimum keeper price",
            min_value=0, max_value=50, value=1,
            key="min_keeper_price",
            help="Lowest cost a keeper can be"
        )
    with col2:
        max_price = st.number_input(
            "Maximum keeper price",
            min_value=0, max_value=500, value=0,
            key="max_keeper_price",
            help="Highest cost a keeper can be (0 = no limit)"
        )
    with col3:
        round_to_integer = st.checkbox(
            "Round prices to whole dollars",
            value=True,
            key="round_to_integer",
            help="Round calculated keeper prices to nearest dollar"
        )

    st.markdown("---")

    # Step 2: Base Cost Rules (How original cost is determined)
    st.markdown("#### 2. Base Cost Rules")
    st.caption("How is the original keeper cost determined based on how you acquired the player?")

    # Acquisition type selector
    acquisition_types_info = {
        "auction": "Players drafted in an auction (you paid $X)",
        "snake": "Players drafted in a snake draft (you used a round pick)",
        "faab_only": "Players picked up via FAAB waivers only (no draft cost)",
        "free_agent": "Players picked up as free agents ($0 FAAB)"
    }

    base_cost_rules = {}

    for acq_type, description in acquisition_types_info.items():
        with st.expander(f"**{acq_type.replace('_', ' ').title()}** - {description}", expanded=(acq_type == "auction")):

            source_options = {
                "auction": ["Draft price paid", "Draft price + flat adjustment", "Draft price × multiplier", "Custom formula"],
                "snake": ["Round number as dollars", "Fixed cost per round", "Custom formula"],
                "faab_only": ["FAAB bid amount", "FAAB × multiplier", "Fixed cost", "FAAB + flat adjustment", "Custom formula"],
                "free_agent": ["Fixed cost", "Custom formula"]
            }

            source = st.selectbox(
                "Base cost calculation:",
                source_options[acq_type],
                key=f"base_cost_source_{acq_type}"
            )

            rule = {"source": source}

            if "flat adjustment" in source.lower():
                rule["adjustment"] = st.number_input(
                    "Adjustment ($)",
                    min_value=-100, max_value=100, value=0,
                    key=f"base_cost_adj_{acq_type}",
                    help="Amount to add to the base cost (can be negative)"
                )

            if "multiplier" in source.lower():
                rule["multiplier"] = st.number_input(
                    "Multiplier",
                    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                    key=f"base_cost_mult_{acq_type}",
                    help="Multiply the base cost by this factor (e.g., 0.5 = half)"
                )

            if "fixed" in source.lower():
                rule["value"] = st.number_input(
                    "Fixed cost ($)",
                    min_value=0, max_value=100, value=5 if acq_type != "free_agent" else 1,
                    key=f"base_cost_fixed_{acq_type}"
                )

            if "per round" in source.lower():
                rule["dollars_per_round"] = st.number_input(
                    "Dollars per round",
                    min_value=1, max_value=50, value=10,
                    key=f"base_cost_per_round_{acq_type}",
                    help="e.g., Round 3 pick = 3 × $10 = $30"
                )

            if "custom formula" in source.lower():
                st.markdown("**Custom Formula Variables:**")
                st.caption("`draft_price`, `faab_bid`, `round`, `keeper_year`")
                rule["formula"] = st.text_input(
                    "Formula expression:",
                    value="draft_price + 5" if acq_type == "auction" else "faab_bid * 0.5" if acq_type == "faab_only" else "5",
                    key=f"base_cost_formula_{acq_type}",
                    help="Python-style expression. Example: draft_price * 1.2 + 5"
                )
                rule["description"] = st.text_input(
                    "Description (for display):",
                    value="",
                    key=f"base_cost_desc_{acq_type}",
                    placeholder="e.g., Draft price plus $5"
                )

            base_cost_rules[acq_type] = rule

    st.markdown("---")

    # Step 3: Year-over-Year Escalation (Formulas by Keeper Year)
    st.markdown("#### 3. Keeper Price Escalation")
    st.caption("How do keeper costs change when you keep a player for multiple years?")

    escalation_mode = st.radio(
        "Escalation method:",
        [
            "Simple (same rule every year)",
            "Per-year formulas (different rules for Year 1, Year 2+, etc.)"
        ],
        horizontal=True,
        key="escalation_mode"
    )

    formulas_by_keeper_year = {}

    if escalation_mode == "Simple (same rule every year)":
        simple_escalation = st.selectbox(
            "How does cost change each year kept?",
            [
                "No change (same as base cost)",
                "Flat increase per year (e.g., +$5)",
                "Percentage increase per year (e.g., +20%)",
                "Multiplier per year (e.g., ×1.2)"
            ],
            key="simple_escalation_type"
        )

        if "flat increase" in simple_escalation.lower():
            flat_inc = st.number_input(
                "Dollar increase per keeper year",
                min_value=0, max_value=100, value=5,
                key="simple_flat_increase"
            )
            for yr in range(1, max_years + 1):
                formulas_by_keeper_year[str(yr)] = {
                    "expression": f"base_cost + {flat_inc * (yr - 1)}",
                    "description": f"Base cost + ${flat_inc * (yr - 1)}"
                }

        elif "percentage increase" in simple_escalation.lower():
            pct_inc = st.number_input(
                "Percentage increase per year",
                min_value=0, max_value=200, value=20,
                key="simple_pct_increase"
            )
            for yr in range(1, max_years + 1):
                mult = (1 + pct_inc / 100) ** (yr - 1)
                formulas_by_keeper_year[str(yr)] = {
                    "expression": f"base_cost * {mult:.4f}",
                    "description": f"Base cost × {mult:.2f} ({pct_inc}% compound for {yr - 1} years)"
                }

        elif "multiplier" in simple_escalation.lower():
            mult_per_year = st.number_input(
                "Multiplier per year",
                min_value=1.0, max_value=3.0, value=1.2, step=0.1,
                key="simple_multiplier"
            )
            for yr in range(1, max_years + 1):
                total_mult = mult_per_year ** (yr - 1)
                formulas_by_keeper_year[str(yr)] = {
                    "expression": f"base_cost * {total_mult:.4f}",
                    "description": f"Base cost × {total_mult:.2f}"
                }

        else:  # No change
            for yr in range(1, max_years + 1):
                formulas_by_keeper_year[str(yr)] = {
                    "expression": "base_cost",
                    "description": "Same as base cost"
                }

    else:  # Per-year formulas
        st.markdown("**Define formula for each keeper year:**")
        st.caption("Use variables: `base_cost`, `draft_price`, `faab_bid`, `keeper_year`")

        # Allow wildcard years like "2+"
        use_wildcard = st.checkbox(
            "Use '2+' wildcard for years 2 and beyond",
            value=True,
            key="use_wildcard_years",
            help="Instead of defining Year 2, Year 3, etc. separately, use '2+' to apply the same formula to all years 2+"
        )

        if use_wildcard:
            years_to_configure = ["1", "2+"]
        else:
            years_to_configure = [str(yr) for yr in range(1, max_years + 1)]

        for yr_key in years_to_configure:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Year {yr_key}:**")
            with col2:
                default_formula = "base_cost" if yr_key == "1" else "base_cost * 1.2 + 5"
                formula_expr = st.text_input(
                    f"Formula for year {yr_key}",
                    value=default_formula,
                    key=f"formula_year_{yr_key}",
                    label_visibility="collapsed"
                )
                formula_desc = st.text_input(
                    f"Description for year {yr_key}",
                    value="",
                    key=f"formula_desc_{yr_key}",
                    placeholder="e.g., 20% increase + $5",
                    label_visibility="collapsed"
                )
                formulas_by_keeper_year[yr_key] = {
                    "expression": formula_expr,
                    "description": formula_desc or formula_expr
                }

    st.markdown("---")

    # Step 4: Snake Draft Settings (if applicable)
    st.markdown("#### 4. Snake Draft Keeper Rules (Optional)")

    has_snake = st.checkbox(
        "League uses snake draft (configure round penalties)",
        value=False,
        key="has_snake_draft"
    )

    snake_rules = None
    if has_snake:
        snake_penalty_type = st.selectbox(
            "How are keeper rounds determined?",
            [
                "Keep at draft round (no penalty)",
                "Lose 1 round per year kept",
                "Lose 2 rounds per year kept",
                "Custom round penalty"
            ],
            key="snake_penalty_type"
        )

        if snake_penalty_type == "Custom round penalty":
            snake_penalty = st.number_input(
                "Rounds lost per year kept",
                min_value=0, max_value=5, value=1,
                key="snake_penalty"
            )
        else:
            snake_penalty = {
                "Keep at draft round (no penalty)": 0,
                "Lose 1 round per year kept": 1,
                "Lose 2 rounds per year kept": 2
            }.get(snake_penalty_type, 0)

        undrafted_round = st.number_input(
            "Undrafted players kept at round:",
            min_value=1, max_value=20, value=10,
            key="undrafted_round",
            help="What round pick does it cost to keep an undrafted player?"
        )

        snake_rules = {
            "penalty_per_year": snake_penalty,
            "undrafted_round": undrafted_round
        }

    st.markdown("---")

    # Step 5: Preview Section
    st.markdown("#### 5. Cost Preview")

    preview_col1, preview_col2, preview_col3 = st.columns(3)
    with preview_col1:
        preview_acq_type = st.selectbox(
            "Acquisition type:",
            list(base_cost_rules.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
            key="preview_acq_type"
        )
    with preview_col2:
        preview_draft_price = st.number_input(
            "Draft price / FAAB bid ($):",
            min_value=0, max_value=200, value=25,
            key="preview_draft_price"
        )
    with preview_col3:
        preview_round = st.number_input(
            "Draft round (if snake):",
            min_value=1, max_value=20, value=5,
            key="preview_round"
        )

    # Calculate and display preview
    def evaluate_formula(expr: str, context: dict) -> float:
        """Safely evaluate a formula expression with given context."""
        try:
            # Only allow specific operations
            allowed_names = {"base_cost", "draft_price", "faab_bid", "round", "keeper_year"}
            # Simple expression evaluation
            result = eval(expr, {"__builtins__": {}}, context)
            return float(result)
        except Exception:
            return 0.0

    def calculate_base_cost(rule: dict, draft_price: float, faab_bid: float, round_num: int) -> float:
        """Calculate base cost from acquisition rule."""
        source = rule.get("source", "").lower()

        if "draft price paid" in source or "draft price" == source:
            base = draft_price
        elif "faab" in source and "amount" in source:
            base = faab_bid
        elif "round number" in source:
            base = round_num
        elif "fixed" in source:
            base = rule.get("value", 1)
        elif "custom" in source:
            context = {"draft_price": draft_price, "faab_bid": faab_bid, "round": round_num, "keeper_year": 1}
            base = evaluate_formula(rule.get("formula", "0"), context)
        else:
            base = draft_price

        # Apply adjustments
        if "adjustment" in rule:
            base += rule["adjustment"]
        if "multiplier" in rule:
            base *= rule["multiplier"]
        if "dollars_per_round" in rule:
            base = round_num * rule["dollars_per_round"]

        return base

    # Calculate base cost for preview
    acq_rule = base_cost_rules.get(preview_acq_type, {})
    base_cost = calculate_base_cost(acq_rule, preview_draft_price, preview_draft_price, preview_round)

    st.markdown(f"**Base cost for {preview_acq_type.replace('_', ' ')}: ${base_cost:.0f}**")

    # Show year-by-year preview
    st.markdown("**Keeper cost by year:**")
    preview_results = []
    for yr in range(1, max_years + 1):
        yr_key = str(yr)
        if yr_key not in formulas_by_keeper_year:
            # Check for wildcard
            if "2+" in formulas_by_keeper_year and yr >= 2:
                yr_key = "2+"
            else:
                continue

        formula = formulas_by_keeper_year[yr_key]
        context = {
            "base_cost": base_cost,
            "draft_price": preview_draft_price,
            "faab_bid": preview_draft_price,
            "round": preview_round,
            "keeper_year": yr
        }
        cost = evaluate_formula(formula["expression"], context)

        # Apply min/max
        cost = max(min_price, cost)
        if max_price > 0:
            cost = min(max_price, cost)
        if round_to_integer:
            cost = round(cost)

        preview_results.append({"year": yr, "cost": cost, "desc": formula.get("description", "")})

    for res in preview_results:
        st.markdown(f"- **Year {res['year']}**: ${res['cost']:.0f}" + (f" ({res['desc']})" if res['desc'] else ""))

    st.markdown("---")

    # Build keeper_rules dict matching LeagueContext schema
    keeper_rules = {
        "enabled": True,
        "max_keepers": max_keepers,
        "max_years": max_years,
        "budget": budget,
        "min_price": min_price,
        "max_price": max_price if max_price > 0 else None,
        "round_to_integer": round_to_integer,
        "formulas_by_keeper_year": formulas_by_keeper_year,
        "base_cost_rules": base_cost_rules,
    }

    if snake_rules:
        keeper_rules["snake"] = snake_rules

    return keeper_rules


# =========================
# External Data Files Configuration UI
# =========================
def get_data_templates() -> dict:
    """Return template structures for each data type."""
    return {
        "matchup": {
            "description": "Weekly matchup results (scores, wins/losses)",
            "required_columns": ["week", "year", "manager", "team_name", "team_points", "opponent", "opponent_points", "win", "loss"],
            "optional_columns": ["team_projected_points", "opponent_projected_points", "margin", "division_id", "is_playoffs", "is_consolation"],
            "example_row": {
                "week": 1, "year": 2013, "manager": "John", "team_name": "Team Awesome",
                "team_points": 125.5, "opponent": "Jane", "opponent_points": 110.2,
                "win": 1, "loss": 0, "division_id": "", "is_playoffs": 0, "is_consolation": 0
            }
        },
        "player": {
            "description": "Weekly player stats (points scored per player per week)",
            "required_columns": ["year", "week", "manager", "player", "points"],
            "optional_columns": ["nfl_position", "lineup_position", "nfl_team", "projected_points", "percent_started", "percent_owned"],
            "example_row": {
                "year": 2013, "week": 1, "manager": "John", "player": "Patrick Mahomes",
                "points": 28.5, "nfl_position": "QB", "lineup_position": "QB", "nfl_team": "KC"
            }
        },
        "draft": {
            "description": "Draft results (picks and costs)",
            "required_columns": ["year", "round", "pick", "manager", "player"],
            "optional_columns": ["cost", "keeper", "draft_type"],
            "example_row": {
                "year": 2013, "round": 1, "pick": 1, "manager": "John",
                "player": "Adrian Peterson", "cost": 65, "keeper": 0, "draft_type": "auction"
            }
        },
        "transactions": {
            "description": "Waiver/FA pickups, drops, and trades",
            "required_columns": ["year", "week", "manager", "player", "transaction_type"],
            "optional_columns": ["faab_bid", "source_type", "destination_type", "trade_partner"],
            "example_row": {
                "year": 2013, "week": 3, "manager": "John", "player": "Tyreek Hill",
                "transaction_type": "add", "faab_bid": 15, "source_type": "waivers"
            }
        },
        "schedule": {
            "description": "Season schedule (who plays who each week)",
            "required_columns": ["year", "week", "manager", "opponent"],
            "optional_columns": ["is_playoffs", "is_consolation", "week_start", "week_end"],
            "example_row": {
                "year": 2013, "week": 1, "manager": "John", "opponent": "Jane",
                "is_playoffs": 0, "is_consolation": 0
            }
        }
    }


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
# Config / Secrets
# =========================
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID") or st.secrets.get("YAHOO_CLIENT_ID", None)
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET") or st.secrets.get("YAHOO_CLIENT_SECRET", None)
MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN") or st.secrets.get("MOTHERDUCK_TOKEN", "")
REDIRECT_URI = os.environ.get("REDIRECT_URI", "https://leaguehistory.streamlit.app")

AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

ROOT_DIR = Path(__file__).parent.resolve()
OAUTH_DIR = ROOT_DIR / "oauth"


# =========================
# Custom CSS
# =========================
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }

    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .hero p {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-bottom: 2rem;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: #667eea;
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1a202c;
    }

    .feature-desc {
        color: #718096;
        font-size: 0.9rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }

    .status-queued {
        background: #fef3c7;
        color: #92400e;
    }

    .status-running {
        background: #dbeafe;
        color: #1e40af;
    }

    .status-success {
        background: #d1fae5;
        color: #065f46;
    }

    .status-failed {
        background: #fee2e2;
        color: #991b1b;
    }

    /* Job card */
    .job-card {
        background: linear-gradient(135deg, #f6f8fc 0%, #ffffff 100%);
        border-left: 4px solid #667eea;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .job-id {
        font-family: 'Courier New', monospace;
        background: #f1f5f9;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }

    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.25rem;
        text-align: center;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.25rem;
    }

    .stat-label {
        color: #718096;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Timeline */
    .timeline-item {
        position: relative;
        padding-left: 2rem;
        padding-bottom: 1.5rem;
        border-left: 2px solid #e2e8f0;
    }

    .timeline-item:last-child {
        border-left: 2px solid transparent;
    }

    .timeline-dot {
        position: absolute;
        left: -0.5rem;
        width: 1rem;
        height: 1rem;
        border-radius: 50%;
        background: #667eea;
        border: 2px solid white;
        box-shadow: 0 0 0 3px #e2e8f0;
    }

    .timeline-content {
        margin-top: -0.25rem;
    }

    /* Button enhancements */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# =========================
# OAuth Helpers (same as before)
# =========================
def get_auth_header() -> str:
    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


def build_authorize_url(state: str | None = None) -> str:
    params = {"client_id": CLIENT_ID, "redirect_uri": REDIRECT_URI, "response_type": "code"}
    if state:
        params["state"] = state
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code_for_tokens(code: str) -> dict:
    headers = {"Authorization": get_auth_header(), "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "authorization_code", "redirect_uri": REDIRECT_URI, "code": code}
    resp = requests.post(TOKEN_URL, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()


def yahoo_api_call(access_token: str, endpoint: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/{endpoint}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def get_user_games(access_token: str):
    return yahoo_api_call(access_token, "users;use_login=1/games?format=json")


def get_user_football_leagues(access_token: str, game_key: str):
    return yahoo_api_call(access_token, f"users;use_login=1/games;game_keys={game_key}/leagues?format=json")


def get_league_teams(access_token: str, league_key: str, year: int = None) -> list[dict]:
    """
    Fetch all teams/managers for a league.
    Returns list of dicts with team_name, manager_name, team_key, year.
    """
    try:
        # Request teams with managers sub-resource
        data = yahoo_api_call(access_token, f"league/{league_key}/teams/managers?format=json")
        teams = []

        league_data = data.get("fantasy_content", {}).get("league", [])
        if len(league_data) > 1:
            teams_data = league_data[1].get("teams", {})
            for key in teams_data:
                if key == "count":
                    continue

                team_entry = teams_data[key].get("team", [])
                team_name = "Unknown Team"
                manager_name = None

                # Parse team info - it's a nested structure
                for part in team_entry:
                    if isinstance(part, list):
                        for item in part:
                            if isinstance(item, dict):
                                if "name" in item:
                                    team_name = item["name"]
                                if "managers" in item:
                                    # Extract manager nickname
                                    mgrs = item["managers"]
                                    if isinstance(mgrs, list) and mgrs:
                                        mgr_data = mgrs[0].get("manager", {})
                                        manager_name = mgr_data.get("nickname") or mgr_data.get("manager_id")
                                    elif isinstance(mgrs, dict):
                                        for mk in mgrs:
                                            if mk != "count":
                                                mgr_data = mgrs[mk].get("manager", {})
                                                manager_name = mgr_data.get("nickname") or mgr_data.get("manager_id")
                                                break
                    elif isinstance(part, dict):
                        if "name" in part:
                            team_name = part["name"]
                        if "managers" in part:
                            mgrs = part["managers"]
                            if isinstance(mgrs, list) and mgrs:
                                mgr_data = mgrs[0].get("manager", {})
                                manager_name = mgr_data.get("nickname") or mgr_data.get("manager_id")

                # Only add if we have a valid manager_name (could be --hidden-- or actual name)
                teams.append({
                    "team_key": "",
                    "team_name": team_name,
                    "manager_name": manager_name if manager_name else "Unknown",
                    "year": year,
                })

        return teams
    except Exception as e:
        # Don't warn for old years that might not exist
        return []


def get_league_teams_all_years(access_token: str, league_name: str, games_data: dict) -> list[dict]:
    """
    Fetch teams/managers across all years for a league.
    Returns list of dicts with team_name, manager_name, team_key, year.
    """
    all_teams = []
    football_games = extract_football_games(games_data)

    for game in football_games:
        game_key = game.get("game_key")
        year = game.get("season")

        try:
            # Get leagues for this game/year
            leagues_data = get_user_football_leagues(access_token, game_key)
            leagues = (
                leagues_data.get("fantasy_content", {})
                .get("users", {}).get("0", {}).get("user", [])[1]
                .get("games", {}).get("0", {}).get("game", [])[1]
                .get("leagues", {})
            )

            for key in leagues:
                if key == "count":
                    continue
                league = leagues[key].get("league", [])[0]
                if league.get("name") == league_name:
                    league_key = league.get("league_key")
                    teams = get_league_teams(access_token, league_key, year)
                    all_teams.extend(teams)
                    break

        except Exception:
            continue

    return all_teams


def find_hidden_managers(teams: list[dict]) -> list[dict]:
    """Find teams with hidden manager names (only --hidden-- pattern)"""
    hidden_teams = []

    for team in teams:
        mgr_name = (team.get("manager_name") or "").strip()
        # Only flag actual "--hidden--" managers, not unknown/empty
        if mgr_name == "--hidden--" or mgr_name.lower() == "hidden":
            hidden_teams.append(team)

    return hidden_teams


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


def extract_football_games(games_data):
    """Extract NFL games from Yahoo API response, sorted by season descending (latest first)."""
    football_games = []
    try:
        games = games_data.get("fantasy_content", {}).get("users", {}).get("0", {}).get("user", [])[1].get("games", {})
        for key in games:
            if key == "count":
                continue
            game = games[key].get("game")
            if isinstance(game, list):
                game = game[0]
            if game and game.get("code") == "nfl":
                football_games.append({
                    "game_key": game.get("game_key"),
                    "season": game.get("season"),
                    "name": game.get("name"),
                })
    except Exception:
        pass
    # Sort by season descending so latest year appears first in dropdown
    football_games.sort(key=lambda g: int(g.get("season", 0)), reverse=True)
    return football_games


def save_oauth_token(token_data: dict, league_info: dict | None = None) -> Path:
    """
    Save OAuth token. Behavior:
    - Always write a global token-only file at oauth/Oauth.json (no league_info) for yahoo-oauth compatibility.
    - If `league_info` is provided, also write a per-league file named oauth/Oauth_<league_key>.json that includes league_info.
    Returns the Path to the file written (per-league file when league_info provided, otherwise global file).
    """
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    oauth_file = OAUTH_DIR / "Oauth.json"

    # Token data (keeps global file free of league metadata)
    oauth_data = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "consumer_key": CLIENT_ID,
        "consumer_secret": CLIENT_SECRET,
        "token_type": token_data.get("token_type", "bearer"),
        "expires_in": token_data.get("expires_in", 3600),
        "token_time": datetime.now(timezone.utc).timestamp(),
        "guid": token_data.get("xoauth_yahoo_guid"),
        "timestamp": datetime.now().isoformat(),
    }

    # Write the global token-only file. For Streamlit Cloud behavior we want
    # the global oauth/Oauth.json to reflect the token for the league being
    # imported so library code reading the default path picks it up. We'll
    # write atomically to avoid partial files.
    def _atomic_write(path: Path, data: dict):
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            tmp.replace(path)
        except Exception:
            # best-effort fallback
            tmp.rename(path)

    try:
        # Always ensure a global token exists (overwrite when saving per-league)
        _atomic_write(oauth_file, oauth_data)
    except Exception:
        # If writing global file fails, continue — per-league file (below) may still be written
        pass

    # If league_info provided, write a per-league file so selecting a league doesn't overwrite the global token file
    if league_info:
        league_key = league_info.get("league_key") or league_info.get("league_id") or "unknown"
        # sanitize league_key for filename
        safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", str(league_key))
        per_file = OAUTH_DIR / f"Oauth_{safe_key}.json"
        per_data = oauth_data.copy()
        per_data["league_info"] = league_info
        try:
            # Write per-league file atomically as well
            tmp = per_file.with_name(f".{per_file.name}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(per_data, f, indent=2)
            try:
                tmp.replace(per_file)
            except Exception:
                tmp.rename(per_file)
            return per_file
        except Exception:
            # fallback to returning global file when per-league write fails
            return oauth_file

    return oauth_file


def save_token_to_motherduck(token_data: dict, league_info: Optional[dict] = None) -> Optional[str]:
    try:
        os.environ.setdefault("MOTHERDUCK_TOKEN", MOTHERDUCK_TOKEN)
        con = duckdb.connect("md:")
        con.execute("""
            CREATE TABLE IF NOT EXISTS secrets.yahoo_oauth_tokens (
                id TEXT, league_key TEXT, token_json TEXT, updated_at TIMESTAMP
            )
        """)
        row_id = str(uuid.uuid4())
        league_key = league_info.get("league_key") if league_info else None
        token_for_storage = {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "consumer_key": CLIENT_ID,
            "consumer_secret": CLIENT_SECRET,
            "token_type": token_data.get("token_type", "bearer"),
            "expires_in": token_data.get("expires_in", 3600),
            "token_time": datetime.now(timezone.utc).timestamp(),
            "guid": token_data.get("xoauth_yahoo_guid"),
            "league_info": league_info,
        }
        token_json = json.dumps(token_for_storage)
        con.execute("INSERT INTO secrets.yahoo_oauth_tokens VALUES (?,?,?,?)",
                    [row_id, league_key, token_json, datetime.now()])
        con.close()
        return row_id
    except Exception:
        return None


def create_import_job_in_motherduck(league_info: dict) -> Optional[str]:
    try:
        os.environ.setdefault("MOTHERDUCK_TOKEN", MOTHERDUCK_TOKEN)
        con = duckdb.connect("md:")
        con.execute("""
            CREATE TABLE IF NOT EXISTS ops.import_status (
                job_id TEXT, league_key TEXT, league_name TEXT, season TEXT,
                status TEXT, created_at TIMESTAMP, updated_at TIMESTAMP
            )
        """)
        job_id = str(uuid.uuid4())
        now = datetime.now()
        con.execute("INSERT INTO ops.import_status VALUES (?,?,?,?,?,?,?)",
                    [job_id, league_info.get("league_key"), league_info.get("name"),
                     str(league_info.get("season", "")), "queued", now, now])
        con.close()
        return job_id
    except Exception:
        return None


def get_job_status(job_id: str) -> dict:
    try:
        os.environ.setdefault("MOTHERDUCK_TOKEN", MOTHERDUCK_TOKEN)
        con = duckdb.connect("md:")
        result = con.execute(
            "SELECT status, updated_at FROM ops.import_status WHERE job_id = ?",
            [job_id]
        ).fetchone()
        con.close()
        if result:
            return {"status": result[0], "updated_at": result[1]}
        return {"status": "not_found"}
    except Exception:
        return {"status": "error"}


# Paths used by the import runner
DATA_DIR = ROOT_DIR / "fantasy_football_data"
SCRIPTS_DIR = ROOT_DIR / "fantasy_football_data_scripts"
INITIAL_IMPORT_SCRIPT = SCRIPTS_DIR / "initial_import_v2.py"


# =========================
# Simplified File Collection
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
# Season Discovery
# =========================
def seasons_for_league_name(access_token: str, all_games: list[dict], target_league_name: str) -> list[str]:
    """Find all seasons where this league exists"""
    seasons = set()
    for g in all_games:
        game_key = g.get("game_key")
        season = str(g.get("season", "")).strip()
        if not game_key or not season:
            continue
        try:
            leagues_data = get_user_football_leagues(access_token, game_key)
            leagues = (
                leagues_data.get("fantasy_content", {})
                .get("users", {}).get("0", {}).get("user", [])[1]
                .get("games", {}).get("0", {}).get("game", [])[1]
                .get("leagues", {})
            )
            for key in leagues:
                if key == "count":
                    continue
                league = leagues[key].get("league", [])[0]
                name = league.get("name")
                if name == target_league_name:
                    seasons.add(season)
                    break
        except Exception:
            pass
    return sorted(seasons)


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
# UI Components
# =========================
def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>🏈 Fantasy Football Analytics</h1>
        <p>Transform your Yahoo Fantasy Football data into powerful insights</p>
    </div>
    """, unsafe_allow_html=True)


def render_feature_card(icon: str, title: str, description: str):
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status: str) -> str:
    status_map = {
        "queued": ("⏳", "status-queued", "Queued"),
        "running": ("🔄", "status-running", "Running"),
        "success": ("✅", "status-success", "Complete"),
        "failed": ("❌", "status-failed", "Failed"),
    }
    icon, css_class, label = status_map.get(status, ("", "status-queued", status))
    return f'<span class="status-badge {css_class}">{icon} {label}</span>'


def render_job_card(job_id: str, league_name: str, status: str):
    st.markdown(f"""
    <div class="job-card">
        <h3>🎯 {league_name}</h3>
        <p><strong>Job ID:</strong> <span class="job-id">{job_id}</span></p>
        <p><strong>Status:</strong> {render_status_badge(status)}</p>
    </div>
    """, unsafe_allow_html=True)


def get_motherduck_progress(job_id: str) -> Optional[dict]:
    """
    Query MotherDuck for import progress.
    Returns progress dict or None if not available.
    """
    motherduck_token = os.environ.get("MOTHERDUCK_TOKEN")
    if not motherduck_token or not job_id:
        return None

    try:
        import duckdb
        con = duckdb.connect("md:")

        result = con.execute("""
            SELECT job_id, league_name, phase, stage, stage_detail,
                   current_step, total_steps, overall_pct, status,
                   error_message, started_at, updated_at
            FROM ops.import_progress
            WHERE job_id = ?
        """, [job_id]).fetchone()

        con.close()

        if result:
            return {
                "job_id": result[0],
                "league_name": result[1],
                "phase": result[2],
                "stage": result[3],
                "stage_detail": result[4],
                "current_step": result[5],
                "total_steps": result[6],
                "overall_pct": result[7] or 0,
                "status": result[8],
                "error_message": result[9],
                "started_at": result[10],
                "updated_at": result[11],
            }
        return None
    except Exception as e:
        # Table might not exist yet
        return None


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


def render_timeline():
    st.markdown("""
    <div style="margin: 2rem 0;">
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 1: Connect</strong>
                <p style="color: #718096; font-size: 0.9rem;">Authenticate with your Yahoo account</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 2: Select</strong>
                <p style="color: #718096; font-size: 0.9rem;">Choose your league and season</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 3: Import</strong>
                <p style="color: #718096; font-size: 0.9rem;">Queue your data for processing</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 4: Analyze</strong>
                <p style="color: #718096; font-size: 0.9rem;">Query your data from anywhere</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


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
