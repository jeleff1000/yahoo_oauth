"""
Keeper Settings Tab

Dedicated configuration page for keeper/dynasty league rules.
Uses a 3-step wizard approach with live summary for clarity.
"""

import streamlit as st
import pandas as pd
from typing import Optional
import json


class KeeperSettingsViewer:
    """Keeper league settings configuration interface with wizard UX."""

    def __init__(self):
        # Initialize session state for wizard
        if "keeper_wizard_step" not in st.session_state:
            st.session_state.keeper_wizard_step = 0
        if "keeper_draft_values" not in st.session_state:
            st.session_state.keeper_draft_values = {}

    @st.fragment
    def display(self):
        # Load existing rules
        existing_rules = self._load_rules_from_db()

        # Initialize values from existing rules or defaults
        self._init_values(existing_rules)

        # Two-column layout: Settings (left) + Live Summary (right)
        col_main, col_summary = st.columns([2, 1])

        with col_main:
            st.markdown("## Keeper League Settings")

            # Step indicator
            steps = ["Basic Rules", "First-Year Price", "Escalation & Preview"]
            current_step = st.session_state.keeper_wizard_step

            # Tab-style step navigation
            step_cols = st.columns(len(steps))
            for i, (col, step_name) in enumerate(zip(step_cols, steps)):
                with col:
                    if i == current_step:
                        st.markdown(f"**{i+1}. {step_name}**")
                    elif i < current_step:
                        if st.button(f"{i+1}. {step_name}", key=f"step_nav_{i}", use_container_width=True):
                            st.session_state.keeper_wizard_step = i
                            st.rerun()
                    else:
                        st.markdown(f":gray[{i+1}. {step_name}]")

            st.markdown("---")

            # Render current step
            if current_step == 0:
                self._render_step_basic()
            elif current_step == 1:
                self._render_step_first_year()
            elif current_step == 2:
                self._render_step_escalation_preview(existing_rules)

        with col_summary:
            self._render_live_summary()

    def _init_values(self, existing_rules: Optional[dict]):
        """Initialize form values from existing rules or defaults."""
        v = st.session_state.keeper_draft_values

        if existing_rules and not v:
            # Load from existing rules
            v["draft_type"] = existing_rules.get("draft_type", "auction")
            v["max_keepers"] = existing_rules.get("max_keepers", 3)
            v["max_years"] = existing_rules.get("max_years") or 3
            v["min_price"] = existing_rules.get("min_price", 1)
            v["budget"] = existing_rules.get("budget", 200)

            base = existing_rules.get("base_cost_rules", {})
            auction = base.get("auction", {})
            faab = base.get("faab_only", {})

            v["draft_mult"] = auction.get("multiplier", 1.0)
            v["draft_flat"] = auction.get("flat", 0.0)
            v["faab_mult"] = faab.get("multiplier", 1.0)
            v["faab_flat"] = faab.get("flat", 10.0)

            esc = existing_rules.get("formulas_by_keeper_year", {}).get("2+", {})
            year1 = existing_rules.get("formulas_by_keeper_year", {}).get("1", {})

            v["esc_mult"] = esc.get("multiplier", 1.0)
            v["esc_flat"] = esc.get("flat_add", 5.0)
            v["esc_from_year1"] = year1.get("apply_year1", False)

        elif not v:
            # Set defaults
            v["draft_type"] = "auction"
            v["max_keepers"] = 3
            v["max_years"] = 3
            v["min_price"] = 1
            v["budget"] = 200
            v["draft_mult"] = 1.0
            v["draft_flat"] = 0.0
            v["faab_mult"] = 1.0
            v["faab_flat"] = 10.0
            v["esc_mult"] = 1.0
            v["esc_flat"] = 5.0
            v["esc_from_year1"] = False

    def _render_step_basic(self):
        """Step 1: Basic Rules - draft type and limits."""
        v = st.session_state.keeper_draft_values

        st.markdown("### Step 1: Basic Rules")
        st.caption("Define your league's keeper format and constraints.")

        # Draft Type - simple radio
        st.markdown("**What type of draft does your league use?**")
        draft_type = st.radio(
            "Draft type",
            ["Auction", "Snake"],
            index=0 if v["draft_type"] == "auction" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="basic_draft_type"
        )
        v["draft_type"] = "auction" if draft_type == "Auction" else "snake"

        st.markdown("")

        # Core limits in a clean grid
        col1, col2 = st.columns(2)

        with col1:
            v["max_keepers"] = st.number_input(
                "Maximum keepers per team",
                min_value=1, max_value=15,
                value=v["max_keepers"],
                key="basic_max_keepers"
            )

        with col2:
            max_years = st.number_input(
                "Maximum years a player can be kept",
                min_value=0, max_value=20,
                value=v["max_years"],
                help="0 = unlimited",
                key="basic_max_years"
            )
            v["max_years"] = max_years if max_years > 0 else None

        # Auction-specific settings
        if v["draft_type"] == "auction":
            col1, col2 = st.columns(2)
            with col1:
                v["budget"] = st.number_input(
                    "Auction budget ($)",
                    min_value=50, max_value=1000,
                    value=v["budget"],
                    key="basic_budget"
                )
            with col2:
                v["min_price"] = st.number_input(
                    "Minimum keeper price ($)",
                    min_value=0, max_value=50,
                    value=v["min_price"],
                    key="basic_min_price"
                )

        # Advanced options (collapsed by default)
        with st.expander("Advanced Constraints"):
            st.caption("Additional rules (most leagues don't need these)")
            st.checkbox("Prevent keeping players drafted in round 1", key="adv_no_rd1", disabled=True)
            st.checkbox("Require minimum games played to keep", key="adv_min_games", disabled=True)
            st.info("Advanced constraints coming soon.")

        # Navigation
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Next: First-Year Price", type="primary", use_container_width=True):
                st.session_state.keeper_wizard_step = 1
                st.rerun()

    def _render_step_first_year(self):
        """Step 2: First-Year Keeper Price."""
        v = st.session_state.keeper_draft_values

        st.markdown("### Step 2: First-Year Keeper Cost")
        st.caption("How is the keeper price calculated the FIRST time a player is kept?")

        if v["draft_type"] == "auction":
            # Two scenarios: Drafted vs FAAB pickup
            st.markdown("**Drafted Players**")
            st.markdown("Players acquired in the draft.")

            # Show as friendly sentence
            draft_cost = self._calculate_example_cost(25, v["draft_mult"], v["draft_flat"], v["min_price"])
            st.success(f"A $25 drafted player costs **${draft_cost}** to keep")

            with st.expander("Edit formula"):
                col1, col2 = st.columns(2)
                with col1:
                    v["draft_mult"] = st.number_input(
                        "Multiply draft price by",
                        min_value=0.1, max_value=5.0,
                        value=float(v["draft_mult"]),
                        step=0.1,
                        key="fy_draft_mult"
                    )
                with col2:
                    v["draft_flat"] = st.number_input(
                        "Then add ($)",
                        min_value=-50.0, max_value=100.0,
                        value=float(v["draft_flat"]),
                        step=1.0,
                        key="fy_draft_flat"
                    )

            st.markdown("")
            st.markdown("**FAAB/Waiver Pickups**")
            st.markdown("Players acquired via free agency during the season.")

            faab_cost = self._calculate_example_cost(15, v["faab_mult"], v["faab_flat"], v["min_price"])
            st.success(f"A $15 FAAB pickup costs **${faab_cost}** to keep")

            with st.expander("Edit formula"):
                col1, col2 = st.columns(2)
                with col1:
                    v["faab_mult"] = st.number_input(
                        "Multiply FAAB bid by",
                        min_value=0.1, max_value=5.0,
                        value=float(v["faab_mult"]),
                        step=0.1,
                        key="fy_faab_mult"
                    )
                with col2:
                    v["faab_flat"] = st.number_input(
                        "Then add ($)",
                        min_value=-50.0, max_value=100.0,
                        value=float(v["faab_flat"]),
                        step=1.0,
                        key="fy_faab_flat"
                    )

        else:
            # Snake draft
            st.info("Snake draft keeper costs are based on draft round.")
            st.markdown("Players are kept at the round they were drafted, minus any escalation.")

        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Back", use_container_width=True):
                st.session_state.keeper_wizard_step = 0
                st.rerun()
        with col3:
            if st.button("Next: Escalation", type="primary", use_container_width=True):
                st.session_state.keeper_wizard_step = 2
                st.rerun()

    def _render_step_escalation_preview(self, existing_rules: Optional[dict]):
        """Step 3: Escalation & Preview."""
        v = st.session_state.keeper_draft_values

        st.markdown("### Step 3: Escalation & Preview")
        st.caption("How does the keeper cost change each additional year?")

        if v["draft_type"] == "auction":
            # Friendly escalation summary
            esc_desc = self._format_escalation_friendly(v["esc_mult"], v["esc_flat"], v["esc_from_year1"])
            st.success(f"Escalation: **{esc_desc}**")

            with st.expander("Edit escalation formula"):
                col1, col2 = st.columns(2)
                with col1:
                    v["esc_mult"] = st.number_input(
                        "Multiply previous year's cost by",
                        min_value=0.5, max_value=5.0,
                        value=float(v["esc_mult"]),
                        step=0.1,
                        key="esc_mult"
                    )
                with col2:
                    v["esc_flat"] = st.number_input(
                        "Then add ($)",
                        min_value=0.0, max_value=100.0,
                        value=float(v["esc_flat"]),
                        step=1.0,
                        key="esc_flat"
                    )

                v["esc_from_year1"] = st.checkbox(
                    "Apply escalation starting Year 1 (instead of Year 2)",
                    value=v["esc_from_year1"],
                    key="esc_year1"
                )

        # Preview tables
        st.markdown("---")
        st.markdown("### Preview: Cost Progression")

        if v["draft_type"] == "auction":
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**$25 Drafted Player**")
                self._show_preview_table(v, 25, "draft")

            with col2:
                st.markdown("**$15 FAAB Pickup**")
                self._show_preview_table(v, 15, "faab")
        else:
            st.info("Snake draft preview coming soon.")

        # Final actions
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Back", use_container_width=True):
                st.session_state.keeper_wizard_step = 1
                st.rerun()

        with col2:
            if st.button("Save to Session", use_container_width=True):
                config = self._build_config(v)
                st.session_state.configured_keeper_rules = config
                st.success("Saved to session!")

        with col3:
            if st.button("Save to Database", type="primary", use_container_width=True):
                config = self._build_config(v)
                st.session_state.configured_keeper_rules = config
                if self._save_rules_to_db(config):
                    st.success("Saved to database!")
                    st.info("Go to 'Recalculate' to update keeper prices.")

        # Recalculate section
        st.markdown("---")
        st.markdown("### Recalculate Keeper Prices")

        if existing_rules:
            st.caption("Apply your rules to recalculate all keeper prices in MotherDuck.")
            if st.button("Recalculate Now", type="secondary"):
                with st.spinner("Recalculating..."):
                    config = self._build_config(v)
                    success = self._run_recalculation(config)
                    if success:
                        st.success("Keeper prices recalculated!")
                        st.cache_data.clear()
        else:
            st.warning("Save your rules first before recalculating.")

    def _render_live_summary(self):
        """Right sidebar: live summary of current settings."""
        v = st.session_state.keeper_draft_values

        st.markdown("### Your Rules")
        st.markdown("---")

        # Basic info
        st.markdown(f"**Draft Type:** {v.get('draft_type', 'auction').title()}")
        st.markdown(f"**Max Keepers:** {v.get('max_keepers', 3)}")

        max_years = v.get('max_years')
        st.markdown(f"**Max Years:** {max_years if max_years else 'Unlimited'}")

        if v.get("draft_type") == "auction":
            st.markdown(f"**Budget:** ${v.get('budget', 200)}")
            st.markdown(f"**Min Price:** ${v.get('min_price', 1)}")

            st.markdown("---")
            st.markdown("**First-Year Cost**")

            draft_formula = self._format_formula_friendly(v.get("draft_mult", 1.0), v.get("draft_flat", 0.0), "draft price")
            faab_formula = self._format_formula_friendly(v.get("faab_mult", 1.0), v.get("faab_flat", 10.0), "FAAB bid")

            st.markdown(f"Drafted: {draft_formula}")
            st.markdown(f"FAAB: {faab_formula}")

            st.markdown("---")
            st.markdown("**Escalation**")

            esc_desc = self._format_escalation_friendly(
                v.get("esc_mult", 1.0),
                v.get("esc_flat", 5.0),
                v.get("esc_from_year1", False)
            )
            st.markdown(esc_desc)

    def _format_formula_friendly(self, mult: float, flat: float, source: str) -> str:
        """Format a formula as friendly text."""
        if mult == 1.0 and flat == 0.0:
            return f"= {source}"
        elif mult == 1.0 and flat > 0:
            return f"= {source} + ${flat:.0f}"
        elif mult == 1.0 and flat < 0:
            return f"= {source} - ${abs(flat):.0f}"
        elif flat == 0.0:
            return f"= {mult}x {source}"
        elif flat > 0:
            return f"= {mult}x {source} + ${flat:.0f}"
        else:
            return f"= {mult}x {source} - ${abs(flat):.0f}"

    def _format_escalation_friendly(self, mult: float, flat: float, from_year1: bool) -> str:
        """Format escalation as friendly text."""
        parts = []

        if mult != 1.0:
            parts.append(f"{mult}x previous cost")

        if flat > 0:
            if parts:
                parts.append(f"+ ${flat:.0f}")
            else:
                parts.append(f"+${flat:.0f}/year")

        if not parts:
            return "None (cost stays the same)"

        result = " ".join(parts)
        start = "Year 1" if from_year1 else "Year 2"
        return f"{result} (starting {start})"

    def _calculate_example_cost(self, base: float, mult: float, flat: float, min_price: int) -> int:
        """Calculate example keeper cost."""
        cost = base * mult + flat
        return max(min_price, round(cost))

    def _show_preview_table(self, v: dict, base_value: float, acq_type: str):
        """Show preview table of keeper cost progression."""
        if acq_type == "draft":
            mult = v.get("draft_mult", 1.0)
            flat = v.get("draft_flat", 0.0)
        else:
            mult = v.get("faab_mult", 1.0)
            flat = v.get("faab_flat", 10.0)

        base_cost = base_value * mult + flat

        esc_mult = v.get("esc_mult", 1.0)
        esc_flat = v.get("esc_flat", 5.0)
        apply_year1 = v.get("esc_from_year1", False)

        min_price = v.get("min_price", 1)
        max_years = v.get("max_years") or 10

        rows = []
        prev_cost = base_cost

        for yr in range(1, min(6, max_years + 1) if max_years else 6):
            if yr == 1:
                if apply_year1:
                    cost = base_cost * esc_mult + esc_flat
                else:
                    cost = base_cost
            else:
                cost = prev_cost * esc_mult + esc_flat

            cost = max(min_price, round(cost))
            change = f"+${cost - prev_cost:.0f}" if yr > 1 and cost > prev_cost else "-"
            rows.append({"Year": yr, "Cost": f"${cost}", "Change": change})
            prev_cost = cost

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    def _build_config(self, v: dict) -> dict:
        """Build configuration dict from form values."""
        base_cost_rules = {
            "auction": {"source": "draft_price", "multiplier": v["draft_mult"], "flat": v["draft_flat"]},
            "faab_only": {"source": "faab_bid", "multiplier": v["faab_mult"], "flat": v["faab_flat"]},
            "free_agent": {"source": "fixed", "value": v["min_price"]}
        }

        formulas = {}
        if v["esc_from_year1"]:
            formulas["1"] = {
                "expression": f"base_cost * {v['esc_mult']} + {v['esc_flat']}",
                "multiplier": v["esc_mult"],
                "flat_add": v["esc_flat"],
                "apply_year1": True
            }
        else:
            formulas["1"] = {"expression": "base_cost"}

        formulas["2+"] = {
            "expression": f"prev_cost * {v['esc_mult']} + {v['esc_flat']}",
            "multiplier": v["esc_mult"],
            "flat_add": v["esc_flat"],
            "recursive": True
        }

        return {
            "enabled": True,
            "draft_type": v["draft_type"],
            "max_keepers": v["max_keepers"],
            "max_years": v["max_years"],
            "budget": v["budget"],
            "min_price": v["min_price"],
            "max_price": None,
            "round_to_integer": True,
            "base_cost_rules": base_cost_rules,
            "formulas_by_keeper_year": formulas
        }

    def _load_rules_from_db(self) -> Optional[dict]:
        """Load keeper rules from MotherDuck league_context table."""
        try:
            from md.core import run_query, T

            query = f"""
                SELECT keeper_rules_json
                FROM {T.get('league_context', 'league_context')}
                LIMIT 1
            """
            df = run_query(query)
            if df is not None and not df.empty:
                rules_json = df.iloc[0].get('keeper_rules_json')
                if rules_json and pd.notna(rules_json):
                    return json.loads(rules_json)
            return None
        except Exception:
            return None

    def _save_rules_to_db(self, rules: dict) -> bool:
        """Save keeper rules to MotherDuck league_context table."""
        try:
            from md.core import get_connection, T

            rules_json = json.dumps(rules)
            conn = get_connection()
            conn.execute(f"""
                UPDATE {T.get('league_context', 'league_context')}
                SET keeper_rules_json = ?
            """, [rules_json])
            return True
        except Exception as e:
            st.error(f"Failed to save rules: {e}")
            return False

    def _run_recalculation(self, rules: dict) -> bool:
        """Run keeper economics recalculation."""
        try:
            from md.core import run_query, T, get_connection
            import sys
            from pathlib import Path

            scripts_path = Path(__file__).resolve().parents[4] / "fantasy_football_data_scripts"
            if str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))

            from multi_league.transformations.draft.modules.consecutive_keeper_calculator import (
                calculate_consecutive_keeper_years
            )
            from multi_league.transformations.draft.keeper_economics_v2 import (
                KeeperPriceCalculator
            )

            st.write("Loading draft data...")
            draft_table = T.get('draft', 'draft')
            draft = run_query(f"SELECT * FROM {draft_table}")

            if draft is None or draft.empty:
                st.error("No draft data found.")
                return False

            st.write(f"Loaded {len(draft):,} records")

            calculator = KeeperPriceCalculator(rules)

            st.write("Calculating keeper years...")
            draft = calculate_consecutive_keeper_years(
                draft,
                player_id_col='yahoo_player_id',
                keeper_col='is_keeper_status',
                year_col='year',
                manager_col='manager' if 'manager' in draft.columns else None
            )
            draft['keeper_year'] = draft['consecutive_years_kept']

            # Detect draft types
            years = sorted(draft['year'].dropna().unique())
            draft_types = {}
            for year in years:
                year_df = draft[draft['year'] == year]
                if 'draft_type' in year_df.columns:
                    dtype = year_df['draft_type'].dropna().str.lower().iloc[0] if len(year_df['draft_type'].dropna()) > 0 else 'snake'
                    draft_types[year] = 'auction' if dtype in ['live', 'auction', 'offline'] else 'snake'
                elif 'cost' in year_df.columns:
                    cost = pd.to_numeric(year_df['cost'], errors='coerce')
                    has_cost = (cost.notna() & (cost > 0)).sum() >= max(1, len(year_df) * 0.25)
                    draft_types[year] = 'auction' if has_cost else 'snake'
                else:
                    draft_types[year] = 'snake'

            st.write("Calculating prices...")
            draft['keeper_price'] = pd.NA
            draft['previous_keeper_price'] = pd.NA

            sort_cols = ['yahoo_player_id', 'year']
            if 'manager' in draft.columns:
                sort_cols = ['yahoo_player_id', 'manager', 'year']
            draft = draft.sort_values(sort_cols).copy()

            keeper_price_history = {}
            count = 0

            for idx, row in draft.iterrows():
                player_id = str(row.get('yahoo_player_id', ''))
                year = row.get('year')
                manager = str(row.get('manager', '')) if 'manager' in draft.columns else ''
                keeper_year = int(row.get('keeper_year', 0))

                if pd.isna(year) or not player_id:
                    continue

                year = int(year)
                draft_type = draft_types.get(year, 'snake')
                key = (player_id, manager)

                if key not in keeper_price_history:
                    keeper_price_history[key] = {}

                prev_year = year - 1
                previous_keeper_price = keeper_price_history[key].get(prev_year, 0.0)

                cost = float(row.get('cost', 0)) if not pd.isna(row.get('cost')) else 0.0
                pick = row.get('pick')
                round_num = row.get('round')
                faab_bid = float(row.get('max_faab_bid', 0)) if not pd.isna(row.get('max_faab_bid')) else 0.0

                is_drafted = pd.notna(row.get('pick')) or (cost > 0)

                if is_drafted:
                    keeper_price = calculator.calculate_keeper_price(
                        draft_type=draft_type,
                        cost=cost,
                        pick=int(pick) if pd.notna(pick) else 0,
                        round_num=int(round_num) if pd.notna(round_num) else 0,
                        faab_bid=faab_bid,
                        is_keeper=keeper_year > 0,
                        previous_keeper_price=previous_keeper_price,
                        keeper_year=keeper_year if keeper_year > 0 else 1,
                    )

                    keeper_price_history[key][year] = keeper_price
                    draft.at[idx, 'keeper_price'] = keeper_price
                    draft.at[idx, 'previous_keeper_price'] = previous_keeper_price if previous_keeper_price > 0 else pd.NA
                    count += 1

            st.write(f"Calculated {count:,} prices")

            update_cols = ['yahoo_player_id', 'year', 'keeper_year', 'keeper_price', 'previous_keeper_price']
            update_df = draft[draft['keeper_price'].notna()][update_cols].copy()

            if len(update_df) > 0:
                conn = get_connection()
                conn.execute("CREATE OR REPLACE TEMP TABLE keeper_updates AS SELECT * FROM update_df")
                conn.execute(f"""
                    UPDATE {draft_table} AS d
                    SET
                        keeper_year = u.keeper_year,
                        keeper_price = u.keeper_price,
                        previous_keeper_price = u.previous_keeper_price
                    FROM keeper_updates AS u
                    WHERE d.yahoo_player_id = u.yahoo_player_id
                      AND d.year = u.year
                """)
                st.success(f"Updated {len(update_df):,} records")
                return True

            st.warning("No prices to update")
            return False

        except Exception as e:
            import traceback
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())
            return False
