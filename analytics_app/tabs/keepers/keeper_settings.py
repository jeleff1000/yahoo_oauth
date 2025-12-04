"""
Keeper Settings Tab

Dedicated configuration page for keeper/dynasty league rules.
This tab allows users to:
1. Configure keeper cost formulas
2. View and edit keeper rules stored in MotherDuck
3. Re-run keeper economics calculations with updated rules
"""

import streamlit as st
import pandas as pd
from typing import Optional
import json


class KeeperSettingsViewer:
    """Keeper league settings configuration interface."""

    def __init__(self):
        pass

    @st.fragment
    def display(self):
        st.markdown("## Keeper League Settings")
        st.caption("Configure how keeper costs are calculated for your league.")

        # Try to load existing rules from MotherDuck
        existing_rules = self._load_rules_from_db()

        # Tab navigation
        tab_names = ["Configuration", "Current Rules", "Recalculate"]
        tabs = st.tabs(tab_names)

        with tabs[0]:
            self._display_configuration_form(existing_rules)

        with tabs[1]:
            self._display_current_rules(existing_rules)

        with tabs[2]:
            self._display_recalculate_section(existing_rules)

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
        except Exception as e:
            # Table might not exist yet
            return None

    def _save_rules_to_db(self, rules: dict) -> bool:
        """Save keeper rules to MotherDuck league_context table."""
        try:
            from md.core import run_query, T, get_connection

            rules_json = json.dumps(rules)

            # Update the league_context table
            conn = get_connection()
            conn.execute(f"""
                UPDATE {T.get('league_context', 'league_context')}
                SET keeper_rules_json = ?
            """, [rules_json])
            conn.close()
            return True
        except Exception as e:
            st.error(f"Failed to save rules: {e}")
            return False

    def _display_configuration_form(self, existing_rules: Optional[dict]):
        """Display the keeper configuration form."""

        st.markdown("### Configure Keeper Rules")

        # Use form to batch inputs
        with st.form("keeper_settings_form"):

            # Draft Type
            st.markdown("#### Draft Type")
            current_draft_type = existing_rules.get("draft_type", "auction") if existing_rules else "auction"

            draft_type = st.radio(
                "What type of draft?",
                ["Auction (bid $ on players)", "Snake (pick by round)"],
                index=0 if current_draft_type == "auction" else 1,
                horizontal=True,
                key="ks_draft_type"
            )
            is_auction = draft_type.startswith("Auction")

            st.markdown("---")

            # Basic Settings
            st.markdown("#### Basic Settings")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                max_keepers = st.number_input(
                    "Max keepers/team",
                    min_value=1, max_value=15,
                    value=existing_rules.get("max_keepers", 3) if existing_rules else 3,
                    key="ks_max_keepers"
                )

            with col2:
                budget = st.number_input(
                    "Auction budget ($)",
                    min_value=50, max_value=1000,
                    value=existing_rules.get("budget", 200) if existing_rules else 200,
                    key="ks_budget"
                )

            with col3:
                max_years = st.number_input(
                    "Max years kept",
                    min_value=0, max_value=20,
                    value=existing_rules.get("max_years") or 3 if existing_rules else 3,
                    key="ks_max_years",
                    help="0 = unlimited"
                )

            with col4:
                min_price = st.number_input(
                    "Min price ($)",
                    min_value=0, max_value=50,
                    value=existing_rules.get("min_price", 1) if existing_rules else 1,
                    key="ks_min_price"
                )

            st.markdown("---")

            if is_auction:
                # First Year Cost
                st.markdown("#### First Year Keeper Cost")
                st.caption("Formula for the first time a player is kept")

                col1, col2 = st.columns(2)

                # Get existing values
                existing_base = existing_rules.get("base_cost_rules", {}) if existing_rules else {}
                existing_auction = existing_base.get("auction", {})
                existing_faab = existing_base.get("faab_only", {})

                with col1:
                    st.markdown("**Drafted Players**")
                    draft_mult = st.number_input(
                        "× draft price",
                        min_value=0.1, max_value=5.0,
                        value=float(existing_auction.get("multiplier", 1.0)),
                        step=0.1,
                        key="ks_draft_mult"
                    )
                    draft_flat = st.number_input(
                        "+ flat amount ($)",
                        min_value=-50.0, max_value=100.0,
                        value=float(existing_auction.get("flat", 0.0)),
                        step=1.0,
                        key="ks_draft_flat"
                    )

                with col2:
                    st.markdown("**FAAB/Waiver Pickups**")
                    faab_mult = st.number_input(
                        "× FAAB bid",
                        min_value=0.1, max_value=5.0,
                        value=float(existing_faab.get("multiplier", 1.0)),
                        step=0.1,
                        key="ks_faab_mult"
                    )
                    faab_flat = st.number_input(
                        "+ flat amount ($)",
                        min_value=-50.0, max_value=100.0,
                        value=float(existing_faab.get("flat", 10.0)),
                        step=1.0,
                        key="ks_faab_flat"
                    )

                st.markdown("---")

                # Escalation
                st.markdown("#### Year-Over-Year Escalation")
                st.caption("How cost changes each additional year kept")

                existing_esc = existing_rules.get("formulas_by_keeper_year", {}).get("2+", {}) if existing_rules else {}

                col1, col2, col3 = st.columns(3)

                with col1:
                    esc_mult = st.number_input(
                        "× previous cost",
                        min_value=0.5, max_value=5.0,
                        value=float(existing_esc.get("multiplier", 1.0)),
                        step=0.1,
                        key="ks_esc_mult",
                        help="1.0 = no multiplication"
                    )

                with col2:
                    esc_flat = st.number_input(
                        "+ flat amount ($)",
                        min_value=0.0, max_value=100.0,
                        value=float(existing_esc.get("flat_add", 5.0)),
                        step=1.0,
                        key="ks_esc_flat"
                    )

                with col3:
                    existing_year1 = existing_rules.get("formulas_by_keeper_year", {}).get("1", {}) if existing_rules else {}
                    esc_from_year1 = st.checkbox(
                        "Apply from Year 1",
                        value=existing_year1.get("apply_year1", False),
                        key="ks_esc_year1",
                        help="Check to apply escalation starting year 1"
                    )

            else:
                # Snake draft settings
                st.markdown("#### Snake Draft Settings")
                rounds_lost = st.number_input(
                    "Rounds lost per year",
                    min_value=0, max_value=5, value=1,
                    key="ks_rounds_lost"
                )
                draft_mult = 1.0
                draft_flat = 0.0
                faab_mult = 1.0
                faab_flat = 0.0
                esc_mult = 1.0
                esc_flat = 0.0
                esc_from_year1 = False

            st.markdown("---")

            # Submit
            col1, col2 = st.columns(2)
            with col1:
                save_to_session = st.form_submit_button("Save to Session", use_container_width=True)
            with col2:
                save_to_db = st.form_submit_button("Save to Database", type="primary", use_container_width=True)

        # Handle form submission
        if save_to_session or save_to_db:
            # Build config
            base_cost_rules = {
                "auction": {"source": "draft_price", "multiplier": draft_mult, "flat": draft_flat},
                "faab_only": {"source": "faab_bid", "multiplier": faab_mult, "flat": faab_flat},
                "free_agent": {"source": "fixed", "value": min_price}
            }

            formulas = {}
            if esc_from_year1:
                formulas["1"] = {
                    "expression": f"base_cost * {esc_mult} + {esc_flat}",
                    "multiplier": esc_mult,
                    "flat_add": esc_flat,
                    "apply_year1": True
                }
            else:
                formulas["1"] = {"expression": "base_cost"}

            formulas["2+"] = {
                "expression": f"prev_cost * {esc_mult} + {esc_flat}",
                "multiplier": esc_mult,
                "flat_add": esc_flat,
                "recursive": True
            }

            config = {
                "enabled": True,
                "draft_type": "auction" if is_auction else "snake",
                "max_keepers": max_keepers,
                "max_years": max_years if max_years > 0 else None,
                "budget": budget,
                "min_price": min_price,
                "max_price": None,
                "round_to_integer": True,
                "base_cost_rules": base_cost_rules,
                "formulas_by_keeper_year": formulas
            }

            # Save to session
            st.session_state.configured_keeper_rules = config

            if save_to_db:
                if self._save_rules_to_db(config):
                    st.success("Keeper rules saved to database!")
                    st.info("Run 'Recalculate' to update keeper prices with new rules.")
            else:
                st.success("Keeper rules saved to session!")

    def _display_current_rules(self, existing_rules: Optional[dict]):
        """Display currently configured rules."""

        st.markdown("### Current Keeper Rules")

        if not existing_rules:
            st.warning("No keeper rules configured yet.")
            st.info("Use the Configuration tab to set up your league's keeper rules.")
            return

        # Display summary
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Basic Settings**")
            st.write(f"- Draft Type: {existing_rules.get('draft_type', 'unknown').title()}")
            st.write(f"- Max Keepers: {existing_rules.get('max_keepers', 'N/A')}")
            st.write(f"- Max Years: {existing_rules.get('max_years') or 'Unlimited'}")
            st.write(f"- Budget: ${existing_rules.get('budget', 200)}")
            st.write(f"- Min Price: ${existing_rules.get('min_price', 1)}")

        with col2:
            st.markdown("**Cost Formulas**")
            base_rules = existing_rules.get("base_cost_rules", {})

            auction = base_rules.get("auction", {})
            faab = base_rules.get("faab_only", {})

            draft_formula = self._format_formula(auction)
            faab_formula = self._format_formula(faab)

            st.write(f"- Drafted: {draft_formula}")
            st.write(f"- FAAB: {faab_formula}")

            esc = existing_rules.get("formulas_by_keeper_year", {}).get("2+", {})
            esc_formula = self._format_escalation(esc)
            st.write(f"- Escalation: {esc_formula}")

        # Show example calculations
        st.markdown("---")
        st.markdown("**Example: $20 Drafted Player**")

        self._show_example_table(existing_rules, 20, "draft")

        st.markdown("**Example: $15 FAAB Pickup**")
        self._show_example_table(existing_rules, 15, "faab")

    def _format_formula(self, rule: dict) -> str:
        """Format a cost rule as text."""
        mult = rule.get("multiplier", 1.0)
        flat = rule.get("flat", 0.0)

        if mult == 1.0 and flat == 0.0:
            return "= acquisition cost"
        elif mult == 1.0:
            sign = "+" if flat >= 0 else ""
            return f"= acquisition {sign} ${flat:.0f}"
        elif flat == 0.0:
            return f"= {mult}× acquisition"
        else:
            sign = "+" if flat >= 0 else ""
            return f"= {mult}× acquisition {sign} ${flat:.0f}"

    def _format_escalation(self, rule: dict) -> str:
        """Format escalation rule."""
        mult = rule.get("multiplier", 1.0)
        flat = rule.get("flat_add", 0.0)

        if mult == 1.0 and flat == 0.0:
            return "None"
        elif mult == 1.0:
            return f"+${flat:.0f}/year"
        elif flat == 0.0:
            return f"×{mult}/year"
        else:
            return f"×{mult} + ${flat:.0f}/year"

    def _show_example_table(self, rules: dict, base_value: float, acq_type: str):
        """Show example keeper cost progression."""

        base_rules = rules.get("base_cost_rules", {})
        if acq_type == "draft":
            rule = base_rules.get("auction", {})
        else:
            rule = base_rules.get("faab_only", {})

        mult = rule.get("multiplier", 1.0)
        flat = rule.get("flat", 0.0)
        base_cost = base_value * mult + flat

        esc = rules.get("formulas_by_keeper_year", {}).get("2+", {})
        esc_mult = esc.get("multiplier", 1.0)
        esc_flat = esc.get("flat_add", 0.0)

        year1_rule = rules.get("formulas_by_keeper_year", {}).get("1", {})
        apply_year1 = year1_rule.get("apply_year1", False)

        min_price = rules.get("min_price", 1)
        max_years = rules.get("max_years") or 10

        rows = []
        prev_cost = base_cost

        for yr in range(1, min(6, max_years + 1)):
            if yr == 1:
                if apply_year1:
                    cost = base_cost * esc_mult + esc_flat
                else:
                    cost = base_cost
            else:
                cost = prev_cost * esc_mult + esc_flat

            cost = max(min_price, round(cost))
            rows.append({"Year": yr, "Cost": f"${cost}"})
            prev_cost = cost

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    def _display_recalculate_section(self, existing_rules: Optional[dict]):
        """Display recalculation options."""

        st.markdown("### Recalculate Keeper Prices")
        st.caption("Re-run the keeper economics transformation with current rules.")

        if not existing_rules:
            st.warning("Configure keeper rules first before recalculating.")
            return

        st.info("""
        **What this does:**
        1. Loads your draft data from MotherDuck
        2. Calculates `keeper_year` for each player (consecutive years kept by same manager)
        3. Applies your cost formulas to calculate `keeper_price`
        4. Updates the draft table in MotherDuck
        """)

        if st.button("Recalculate Keeper Prices", type="primary"):
            with st.spinner("Recalculating keeper economics..."):
                success = self._run_recalculation(existing_rules)
                if success:
                    st.success("Keeper prices recalculated!")
                    st.cache_data.clear()
                    st.info("Refresh the page to see updated data.")
                else:
                    st.error("Recalculation failed. Check the error above.")

    def _run_recalculation(self, rules: dict) -> bool:
        """Run keeper economics recalculation."""
        try:
            from md.core import run_query, T, get_connection
            import sys
            from pathlib import Path

            # Add transformations path
            scripts_path = Path(__file__).resolve().parents[4] / "fantasy_football_data_scripts"
            if str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))

            # Import the keeper economics calculator
            from multi_league.transformations.draft.modules.consecutive_keeper_calculator import (
                calculate_consecutive_keeper_years
            )
            from multi_league.transformations.draft.keeper_economics_v2 import (
                KeeperPriceCalculator
            )

            st.write("Loading draft data...")

            # Load draft data from MotherDuck
            draft_table = T.get('draft', 'draft')
            draft = run_query(f"SELECT * FROM {draft_table}")

            if draft is None or draft.empty:
                st.error("No draft data found.")
                return False

            st.write(f"Loaded {len(draft):,} draft records")

            # Initialize price calculator with rules
            calculator = KeeperPriceCalculator(rules)

            # Calculate consecutive keeper years
            st.write("Calculating consecutive keeper years...")
            draft = calculate_consecutive_keeper_years(
                draft,
                player_id_col='yahoo_player_id',
                keeper_col='is_keeper_status',
                year_col='year',
                manager_col='manager' if 'manager' in draft.columns else None
            )
            draft['keeper_year'] = draft['consecutive_years_kept']

            # Detect draft types per year
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

            # Calculate keeper prices
            st.write("Calculating keeper prices...")
            draft['keeper_price'] = pd.NA
            draft['previous_keeper_price'] = pd.NA

            # Sort for sequential processing
            sort_cols = ['yahoo_player_id', 'year']
            if 'manager' in draft.columns:
                sort_cols = ['yahoo_player_id', 'manager', 'year']
            draft = draft.sort_values(sort_cols).copy()

            # Track keeper prices: {(player_id, manager): {year: price}}
            keeper_price_history = {}
            keeper_prices_calculated = 0

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
                is_keeper = keeper_year > 0

                if is_drafted:
                    keeper_price = calculator.calculate_keeper_price(
                        draft_type=draft_type,
                        cost=cost,
                        pick=int(pick) if pd.notna(pick) else 0,
                        round_num=int(round_num) if pd.notna(round_num) else 0,
                        faab_bid=faab_bid,
                        is_keeper=is_keeper,
                        previous_keeper_price=previous_keeper_price,
                        keeper_year=keeper_year if keeper_year > 0 else 1,
                    )

                    keeper_price_history[key][year] = keeper_price
                    draft.at[idx, 'keeper_price'] = keeper_price
                    draft.at[idx, 'previous_keeper_price'] = previous_keeper_price if previous_keeper_price > 0 else pd.NA
                    keeper_prices_calculated += 1

            st.write(f"Calculated keeper prices for {keeper_prices_calculated:,} players")

            # Update MotherDuck with new keeper prices
            st.write("Updating MotherDuck...")

            # Columns to update
            update_cols = ['yahoo_player_id', 'year', 'keeper_year', 'keeper_price', 'previous_keeper_price']
            if 'consecutive_years_kept' in draft.columns:
                update_cols.append('consecutive_years_kept')

            # Get only rows with calculated prices
            update_df = draft[draft['keeper_price'].notna()][update_cols].copy()

            if len(update_df) > 0:
                conn = get_connection()

                # Create temp table with updated values
                conn.execute("CREATE OR REPLACE TEMP TABLE keeper_updates AS SELECT * FROM update_df")

                # Update draft table
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

                st.success(f"Updated {len(update_df):,} records in MotherDuck")
                return True
            else:
                st.warning("No keeper prices calculated - nothing to update")
                return False

        except Exception as e:
            import traceback
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())
            return False
