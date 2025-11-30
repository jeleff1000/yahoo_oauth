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
# MotherDuck Database Discovery
# =========================
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

    col1, col2 = st.columns(2)
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

    st.markdown("---")

    # Step 2: Draft Type
    st.markdown("#### 2. Draft Type")

    draft_type = st.radio(
        "What type of draft does your league use?",
        ["Auction ($$)", "Snake (Rounds)", "Both (Hybrid)"],
        horizontal=True,
        key="keeper_draft_type",
        help="This determines how keeper costs are calculated"
    )

    is_auction = draft_type in ["Auction ($$)", "Both (Hybrid)"]
    is_snake = draft_type in ["Snake (Rounds)", "Both (Hybrid)"]

    # Auction-specific settings
    if is_auction:
        st.markdown("---")
        st.markdown("#### 3. Auction Keeper Costs")

        col1, col2, col3 = st.columns(3)
        with col1:
            budget = st.number_input(
                "Auction budget",
                min_value=50, max_value=1000, value=200,
                key="keeper_budget",
                help="Total auction dollars each team has"
            )
        with col2:
            min_price = st.number_input(
                "Minimum keeper price",
                min_value=0, max_value=50, value=1,
                key="min_keeper_price",
                help="Lowest cost a keeper can be"
            )
        with col3:
            max_price = st.number_input(
                "Maximum keeper price",
                min_value=0, max_value=500, value=0,
                key="max_keeper_price",
                help="Highest cost a keeper can be (0 = no limit)"
            )

        st.markdown("##### How do keeper costs increase each year?")

        escalation_type = st.selectbox(
            "Cost escalation method",
            [
                "No increase (same cost each year)",
                "Flat increase (e.g., +$5 each year)",
                "Percentage increase (e.g., +20% each year)",
                "Custom per year"
            ],
            key="escalation_type",
            help="How does the keeper cost change from year to year?"
        )

        # Collect escalation details based on type
        year_costs = []  # Will hold (year, cost_increase, description) tuples

        if escalation_type == "No increase (same cost each year)":
            for yr in range(1, max_years + 1):
                year_costs.append({"year": yr, "increase": 0, "increase_type": "flat"})

        elif escalation_type == "Flat increase (e.g., +$5 each year)":
            col1, col2 = st.columns(2)
            with col1:
                flat_increase = st.number_input(
                    "Dollar increase per year",
                    min_value=0, max_value=100, value=5,
                    key="flat_increase",
                    help="How much does the cost go up each year?"
                )
            with col2:
                first_year_free = st.checkbox(
                    "First year at original cost (no increase)",
                    value=True,
                    key="first_year_free",
                    help="If checked, Year 1 = original cost, increases start in Year 2"
                )

            for yr in range(1, max_years + 1):
                if first_year_free and yr == 1:
                    year_costs.append({"year": yr, "increase": 0, "increase_type": "flat"})
                else:
                    years_of_increase = yr - 1 if first_year_free else yr
                    year_costs.append({"year": yr, "increase": flat_increase * years_of_increase, "increase_type": "flat"})

        elif escalation_type == "Percentage increase (e.g., +20% each year)":
            col1, col2 = st.columns(2)
            with col1:
                pct_increase = st.number_input(
                    "Percentage increase per year",
                    min_value=0, max_value=200, value=20,
                    key="pct_increase",
                    help="What percentage does the cost increase each year?"
                )
            with col2:
                pct_first_year_free = st.checkbox(
                    "First year at original cost",
                    value=True,
                    key="pct_first_year_free"
                )

            for yr in range(1, max_years + 1):
                if pct_first_year_free and yr == 1:
                    # First year = original cost (multiplier of 1.0)
                    year_costs.append({"year": yr, "increase": 1.0, "increase_type": "percentage_multiplier"})
                else:
                    years_of_increase = yr - 1 if pct_first_year_free else yr
                    # Compound percentage: (1 + pct/100)^years
                    multiplier = (1 + pct_increase / 100) ** years_of_increase
                    year_costs.append({"year": yr, "increase": multiplier, "increase_type": "percentage_multiplier"})

        else:  # Custom per year
            st.markdown("Enter the cost increase for each keeper year:")
            for yr in range(1, max_years + 1):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**Year {yr}:**")
                with col2:
                    custom_increase = st.number_input(
                        f"Add to original cost",
                        min_value=0, max_value=200,
                        value=0 if yr == 1 else (yr - 1) * 5,
                        key=f"custom_yr_{yr}",
                        label_visibility="collapsed"
                    )
                    year_costs.append({"year": yr, "increase": custom_increase, "increase_type": "flat"})

        # Base cost rules for different acquisition types
        st.markdown("##### How is the original keeper cost determined?")

        with st.expander("Base Cost Rules (click to customize)", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Drafted players**")
                auction_base = st.selectbox(
                    "Base cost:",
                    ["Draft price paid", "Draft price + adjustment"],
                    key="auction_base_rule"
                )
                auction_adjustment = 0
                if auction_base == "Draft price + adjustment":
                    auction_adjustment = st.number_input(
                        "Adjustment ($)",
                        min_value=-50, max_value=50, value=0,
                        key="auction_adjustment"
                    )

            with col2:
                st.markdown("**FAAB pickups**")
                faab_base = st.selectbox(
                    "Base cost:",
                    ["FAAB price paid", "Half of FAAB price", "Fixed cost"],
                    key="faab_base_rule"
                )
                faab_fixed = None
                if faab_base == "Fixed cost":
                    faab_fixed = st.number_input(
                        "Fixed cost ($)",
                        min_value=0, max_value=50, value=5,
                        key="faab_fixed"
                    )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Free agent pickups ($0 FAAB)**")
                fa_cost = st.number_input(
                    "Keeper cost ($)",
                    min_value=0, max_value=50, value=1,
                    key="fa_keeper_cost",
                    help="What's the keeper cost for players picked up for free?"
                )
            with col2:
                st.markdown("**Undrafted players**")
                undrafted_cost = st.number_input(
                    "Keeper cost ($)",
                    min_value=0, max_value=50, value=1,
                    key="undrafted_keeper_cost",
                    help="What's the keeper cost for players not drafted?"
                )

        # Build base rules
        base_rules = {
            "auction": {
                "source": "draft_price",
                "adjustment": auction_adjustment if auction_base == "Draft price + adjustment" else 0
            },
            "faab": {
                "source": "faab_price" if faab_base == "FAAB price paid" else ("half_faab" if faab_base == "Half of FAAB price" else "fixed"),
                "value": faab_fixed if faab_base == "Fixed cost" else None
            },
            "free_agent": {"source": "fixed", "value": fa_cost},
            "undrafted": {"source": "fixed", "value": undrafted_cost}
        }

        # Preview section
        st.markdown("---")
        st.markdown("#### Cost Preview")

        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            preview_cost = st.number_input(
                "Enter a sample draft price to preview:",
                min_value=1, max_value=200, value=25,
                key="preview_cost"
            )

        # Calculate and display preview
        preview_data = []
        for yc in year_costs:
            yr = yc["year"]
            if yc["increase_type"] == "flat":
                cost = preview_cost + yc["increase"]
            elif yc["increase_type"] == "percentage_multiplier":
                cost = preview_cost * yc["increase"]
            else:
                cost = preview_cost

            # Apply min/max
            cost = max(min_price, cost)
            if max_price > 0:
                cost = min(max_price, cost)

            preview_data.append({"Year Kept": yr, "Keeper Cost": f"${cost:.0f}"})

        with preview_col2:
            st.markdown(f"**Player drafted for ${preview_cost}:**")
            for row in preview_data:
                st.markdown(f"- Year {row['Year Kept']}: {row['Keeper Cost']}")

    # Snake draft settings
    if is_snake:
        st.markdown("---")
        st.markdown("#### Snake Draft Keeper Rules")

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

        st.markdown("##### Undrafted player keeper rules")
        undrafted_round = st.number_input(
            "Undrafted players kept at round:",
            min_value=1, max_value=20, value=10,
            key="undrafted_round",
            help="What round pick does it cost to keep an undrafted player?"
        )

    st.markdown("---")

    # Build keeper_rules dict
    keeper_rules = {
        "enabled": True,
        "max_keepers": max_keepers,
        "max_years": max_years,
        "draft_type": draft_type,
    }

    if is_auction:
        keeper_rules["auction"] = {
            "budget": budget,
            "min_price": min_price,
            "max_price": max_price if max_price > 0 else None,
            "escalation_type": escalation_type,
            "year_costs": year_costs,
            "base_rules": base_rules
        }

    if is_snake:
        keeper_rules["snake"] = {
            "penalty_per_year": snake_penalty,
            "undrafted_round": undrafted_round
        }

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
            def format_league_name(x: str) -> str:
                """Format league name for display - strip 'l_' prefix added for digit-starting names."""
                if x == "":
                    return "-- Choose a league --"
                # Strip the 'l_' prefix if it was added because name started with a digit
                if x.startswith("l_") and len(x) > 2 and x[2].isdigit():
                    return x[2:]
                return x

            selected_db = st.selectbox(
                "Select League:",
                options=[""] + existing_dbs,
                format_func=format_league_name,
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

        if st.button("Register New League", type="secondary", use_container_width=True, key="register_btn"):
            st.session_state.app_mode = "register"
            st.rerun()

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

        st.markdown(f"**Current League:** {selected_db}")

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

                            # Two options: Quick Start or Advanced Settings
                            col_start, col_advanced = st.columns(2)

                            with col_start:
                                # Disable button if import already in progress
                                import_in_progress = st.session_state.get("import_in_progress", False)
                                if import_in_progress:
                                    st.info("Import in progress...")
                                if st.button("Start Import Now", key="start_import_btn", type="primary", use_container_width=True, disabled=import_in_progress):
                                    st.session_state.import_in_progress = True
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

                                # Import button with advanced settings - disable if import already in progress
                                import_in_progress = st.session_state.get("import_in_progress", False)
                                if st.button("Start Import with Settings", key="start_import_advanced_btn", type="primary", use_container_width=True, disabled=import_in_progress):
                                    st.session_state.import_in_progress = True
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

        # Job status section
        if "job_id" in st.session_state:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 📊 Your Import Job")

            status_info = get_job_status(st.session_state.job_id)
            render_job_card(
                st.session_state.job_id,
                st.session_state.get("job_league_name", "League"),
                status_info.get("status", "queued")
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Status", use_container_width=True):
                    st.rerun()
            with col2:
                github_repo = os.environ.get("GITHUB_REPOSITORY", "your-username/your-repo")
                st.link_button(
                    "🔗 View on GitHub",
                    f"https://github.com/{github_repo}/actions",
                    use_container_width=True
                )

            if status_info.get("status") == "success":
                st.success("🎉 Your data is ready in MotherDuck!")
                league_info = st.session_state.get("league_info", {})
                db_name = league_info.get('name', 'league').lower().replace(' ', '_').replace('-', '_')
                st.code(f"SELECT * FROM {db_name}.public.matchup LIMIT 10;", language="sql")

    else:
        # Landing page: show hero and make the Connect CTA the primary focus (large centered block)
        render_hero()

        auth_url = build_authorize_url()
        # Full-width centered CTA with max-width so it looks prominent on desktop and mobile
        connect_html_center = f'''
        <div style="display:flex; justify-content:center; margin: 1.25rem 0;">
            <a href="{auth_url}" target="_blank" rel="noopener noreferrer" style="text-decoration:none; width:100%; max-width:980px;">
                <div style="background: linear-gradient(90deg,#ff6b4a,#ff8a5a); color:white; padding:1.25rem 1.5rem; border-radius:0.75rem; text-align:center; font-weight:700; font-size:1.15rem; box-shadow:0 10px 30px rgba(0,0,0,0.08);">
                    🔐 Connect Yahoo Account
                </div>
            </a>
        </div>
        '''
        st.markdown(connect_html_center, unsafe_allow_html=True)

        # Put feature cards below in a compact grid so the CTA remains the main focus
        st.markdown("<div style='max-width:980px; margin:0 auto;'>", unsafe_allow_html=True)
        features_col1, features_col2 = st.columns(2)
        with features_col1:
            render_feature_card("📈", "Win Probability", "Track your playoff chances")
            render_feature_card("🎯", "Optimal Lineups", "See your best possible scores")
        with features_col2:
            render_feature_card("📊", "Advanced Stats", "Deep dive into performance")
            render_feature_card("🔮", "Predictions", "Expected vs actual records")
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption("🔒 Your data is secure. We only access league statistics, never personal information.")

    # Note: Import is now triggered via st.button() in the league selection UI
    # This preserves session state (including OAuth tokens) which anchor links did not

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
            from streamlit_helpers.trigger_import_workflow import trigger_import_workflow

            result = trigger_import_workflow(
                league_data=league_data,
                github_token=github_token
            )

            if result['success']:
                st.success("✅ Import Started Successfully!")
                st.session_state.import_job_id = result['user_id']
                st.session_state.import_in_progress = False  # Reset since workflow is now queued

                st.markdown(f"""
                ### 📊 Import Job Details
                - **Job ID**: `{result['user_id']}`
                - **League**: {league_data['league_name']} ({league_data['season']})
                - **Estimated Time**: {result.get('estimated_time', '60-120 minutes')}
                - **Track Progress**: [View Workflow]({result['workflow_run_url']})

                The workflow will:
                1. ✅ Fetch all fantasy data from Yahoo
                2. ✅ Merge with NFL stats
                3. ✅ Create parquet files
                4. ✅ Upload to MotherDuck
                5. ✅ Prepare your custom analytics site

                You'll be notified when complete (check back in 1-2 hours).
                """)

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
