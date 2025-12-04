"""
Keeper/Dynasty League Configuration

A form-based configuration page for keeper league rules.
Uses st.form to batch inputs and prevent constant reruns.
"""

import streamlit as st
from typing import Optional


def render_keeper_rules_ui() -> Optional[dict]:
    """
    Render keeper rules configuration using a form to prevent reruns.

    Returns:
        dict with keeper rules configuration, or None if not a keeper league
    """

    # Check if already configured (use configured_keeper_rules for compatibility with main.py)
    existing_config = st.session_state.get("configured_keeper_rules")

    # Quick toggle at top
    is_keeper_league = st.checkbox(
        "This is a Keeper/Dynasty League",
        value=existing_config is not None,
        key="is_keeper_league_toggle"
    )

    if not is_keeper_league:
        st.session_state.configured_keeper_rules = None
        return None

    st.markdown("---")

    # Use a form to batch all inputs
    with st.form("keeper_rules_form"):
        st.markdown("### Configure Keeper Rules")
        st.caption("Fill out all settings, then click 'Save Configuration' at the bottom.")

        # =======================================================================
        # SECTION 1: DRAFT TYPE & BASICS
        # =======================================================================
        st.markdown("#### Draft Type & Basics")

        col1, col2 = st.columns(2)
        with col1:
            draft_type = st.selectbox(
                "Draft Type",
                ["Auction (bid $ on players)", "Snake (pick by round)"],
                index=0 if not existing_config else (0 if existing_config.get("draft_type") == "auction" else 1),
                key="form_draft_type"
            )
            is_auction = draft_type.startswith("Auction")

            max_keepers = st.number_input(
                "Max keepers per team",
                min_value=1, max_value=15,
                value=existing_config.get("max_keepers", 3) if existing_config else 3,
                key="form_max_keepers"
            )

        with col2:
            if is_auction:
                budget = st.number_input(
                    "Auction budget ($)",
                    min_value=50, max_value=1000,
                    value=existing_config.get("budget", 200) if existing_config else 200,
                    key="form_budget"
                )
            else:
                budget = 200
                st.number_input(
                    "Total draft rounds",
                    min_value=5, max_value=25, value=15,
                    key="form_total_rounds"
                )

            max_years = st.number_input(
                "Max years a player can be kept (0 = unlimited)",
                min_value=0, max_value=20,
                value=existing_config.get("max_years", 3) if existing_config else 3,
                key="form_max_years"
            )

        # =======================================================================
        # SECTION 2: FIRST YEAR COST (Base Cost)
        # =======================================================================
        st.markdown("---")
        st.markdown("#### First Year Keeper Cost")
        st.caption("How is the cost determined the FIRST time a player is kept?")

        if is_auction:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Drafted Players**")
                draft_mult = st.number_input(
                    "Multiply draft price by",
                    min_value=0.1, max_value=5.0,
                    value=existing_config.get("base_cost_rules", {}).get("auction", {}).get("multiplier", 1.0) if existing_config else 1.0,
                    step=0.1,
                    key="form_draft_mult",
                    help="1.0 = draft price unchanged"
                )
                draft_flat = st.number_input(
                    "Then add flat $",
                    min_value=-50.0, max_value=100.0,
                    value=existing_config.get("base_cost_rules", {}).get("auction", {}).get("flat", 0.0) if existing_config else 0.0,
                    step=1.0,
                    key="form_draft_flat"
                )

            with col2:
                st.markdown("**FAAB/Waiver Pickups**")
                faab_mult = st.number_input(
                    "Multiply FAAB bid by",
                    min_value=0.1, max_value=5.0,
                    value=existing_config.get("base_cost_rules", {}).get("faab_only", {}).get("multiplier", 1.0) if existing_config else 1.0,
                    step=0.1,
                    key="form_faab_mult"
                )
                faab_flat = st.number_input(
                    "Then add flat $",
                    min_value=-50.0, max_value=100.0,
                    value=existing_config.get("base_cost_rules", {}).get("faab_only", {}).get("flat", 10.0) if existing_config else 10.0,
                    step=1.0,
                    key="form_faab_flat",
                    help="e.g., 10 means keeper cost = FAAB bid + $10"
                )

            min_price = st.number_input(
                "Minimum keeper price ($)",
                min_value=0, max_value=50,
                value=existing_config.get("min_price", 1) if existing_config else 1,
                key="form_min_price"
            )

        else:
            # Snake draft
            st.markdown("**Round-Based Keeper Cost**")
            snake_base = st.selectbox(
                "Keeper round is based on:",
                ["Draft round (where picked)", "Fixed round for all keepers"],
                key="form_snake_base"
            )

            if snake_base == "Fixed round for all keepers":
                fixed_round = st.number_input("Fixed round", min_value=1, max_value=20, value=10, key="form_fixed_round")

            undrafted_round = st.number_input(
                "Undrafted/FAAB players kept at round:",
                min_value=1, max_value=20, value=10,
                key="form_undrafted_round"
            )

            min_price = 1
            draft_mult = 1.0
            draft_flat = 0.0
            faab_mult = 1.0
            faab_flat = 0.0

        # =======================================================================
        # SECTION 3: YEAR-OVER-YEAR ESCALATION
        # =======================================================================
        st.markdown("---")
        st.markdown("#### Year-Over-Year Escalation")
        st.caption("How does the keeper cost CHANGE each additional year kept?")

        if is_auction:
            col1, col2, col3 = st.columns(3)

            with col1:
                esc_mult = st.number_input(
                    "Multiply previous cost by",
                    min_value=0.5, max_value=5.0,
                    value=existing_config.get("formulas_by_keeper_year", {}).get("2+", {}).get("multiplier", 1.0) if existing_config else 1.0,
                    step=0.1,
                    key="form_esc_mult",
                    help="1.0 = no multiplication"
                )

            with col2:
                esc_flat = st.number_input(
                    "Then add flat $",
                    min_value=0.0, max_value=100.0,
                    value=existing_config.get("formulas_by_keeper_year", {}).get("2+", {}).get("flat_add", 5.0) if existing_config else 5.0,
                    step=1.0,
                    key="form_esc_flat",
                    help="e.g., 5 means +$5 each year"
                )

            with col3:
                esc_from_year1 = st.checkbox(
                    "Apply from Year 1",
                    value=existing_config.get("formulas_by_keeper_year", {}).get("1", {}).get("apply_year1", False) if existing_config else False,
                    key="form_esc_year1",
                    help="If checked, escalation applies starting year 1. If unchecked, year 1 = base cost, escalation starts year 2."
                )
        else:
            # Snake escalation
            rounds_lost = st.number_input(
                "Rounds lost per year kept",
                min_value=0, max_value=5,
                value=1,
                key="form_rounds_lost",
                help="e.g., 1 = Round 8 → 7 → 6 each year"
            )
            esc_mult = 1.0
            esc_flat = 0.0
            esc_from_year1 = False

        # =======================================================================
        # SECTION 4: PREVIEW
        # =======================================================================
        st.markdown("---")
        st.markdown("#### Preview (after saving)")
        st.caption("Preview will update after you save the configuration.")

        # =======================================================================
        # SUBMIT BUTTON
        # =======================================================================
        st.markdown("---")
        submitted = st.form_submit_button("Save Keeper Configuration", type="primary", use_container_width=True)

    # Process form submission
    if submitted:
        # Build base cost rules
        base_cost_rules = {}

        if is_auction:
            base_cost_rules["auction"] = {
                "source": "draft_price",
                "multiplier": draft_mult,
                "flat": draft_flat
            }
            base_cost_rules["faab_only"] = {
                "source": "faab_bid",
                "multiplier": faab_mult,
                "flat": faab_flat
            }
            base_cost_rules["free_agent"] = {
                "source": "fixed",
                "value": min_price
            }
        else:
            if snake_base == "Fixed round for all keepers":
                base_cost_rules["snake"] = {"source": "fixed", "value": fixed_round}
            else:
                base_cost_rules["snake"] = {"source": "draft_round"}
            base_cost_rules["undrafted"] = {"source": "fixed", "value": undrafted_round}

        # Build escalation formulas
        formulas_by_keeper_year = {}

        if is_auction:
            if esc_from_year1:
                formulas_by_keeper_year["1"] = {
                    "expression": f"base_cost * {esc_mult} + {esc_flat}",
                    "multiplier": esc_mult,
                    "flat_add": esc_flat,
                    "apply_year1": True
                }
            else:
                formulas_by_keeper_year["1"] = {
                    "expression": "base_cost",
                    "description": "Base cost only"
                }

            formulas_by_keeper_year["2+"] = {
                "expression": f"prev_cost * {esc_mult} + {esc_flat}",
                "multiplier": esc_mult,
                "flat_add": esc_flat,
                "recursive": True
            }
        else:
            formulas_by_keeper_year["1"] = {"expression": "base_round"}
            formulas_by_keeper_year["2+"] = {
                "expression": f"base_round - {rounds_lost} * (keeper_year - 1)",
                "rounds_lost": rounds_lost
            }

        # Build final config
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
            "formulas_by_keeper_year": formulas_by_keeper_year,
        }

        st.session_state.configured_keeper_rules = config
        st.success("Keeper configuration saved!")

        return config

    # Return existing config if not submitting
    if existing_config:
        # Show current config summary
        st.markdown("---")
        st.markdown("#### Current Configuration")

        if existing_config.get("draft_type") == "auction":
            base_rules = existing_config.get("base_cost_rules", {})
            auction_rule = base_rules.get("auction", {})
            faab_rule = base_rules.get("faab_only", {})
            esc_rule = existing_config.get("formulas_by_keeper_year", {}).get("2+", {})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Drafted:** " + _format_cost_formula(auction_rule))
                st.markdown("**FAAB:** " + _format_cost_formula(faab_rule))
            with col2:
                st.markdown("**Escalation:** " + _format_escalation(esc_rule))
                st.markdown(f"**Max keepers:** {existing_config.get('max_keepers')}")

            # Show preview table
            _show_preview_table(existing_config)

        return existing_config

    return None


def _format_cost_formula(rule: dict) -> str:
    """Format a base cost rule as readable text."""
    mult = rule.get("multiplier", 1.0)
    flat = rule.get("flat", 0.0)

    if mult == 1.0 and flat == 0.0:
        return "= acquisition cost"
    elif mult == 1.0:
        return f"= acquisition + ${flat:.0f}"
    elif flat == 0.0:
        return f"= {mult}× acquisition"
    else:
        return f"= {mult}× acquisition + ${flat:.0f}"


def _format_escalation(rule: dict) -> str:
    """Format escalation rule as readable text."""
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


def _show_preview_table(config: dict):
    """Show a preview table of keeper costs."""
    import pandas as pd

    st.markdown("**Preview: $25 drafted player**")

    base_rules = config.get("base_cost_rules", {})
    auction_rule = base_rules.get("auction", {})
    esc_rule = config.get("formulas_by_keeper_year", {}).get("2+", {})
    year1_rule = config.get("formulas_by_keeper_year", {}).get("1", {})

    # Calculate base cost
    draft_price = 25
    base_mult = auction_rule.get("multiplier", 1.0)
    base_flat = auction_rule.get("flat", 0.0)
    base_cost = draft_price * base_mult + base_flat

    esc_mult = esc_rule.get("multiplier", 1.0)
    esc_flat = esc_rule.get("flat_add", 0.0)
    apply_year1 = year1_rule.get("apply_year1", False)

    min_price = config.get("min_price", 1)

    rows = []
    prev_cost = base_cost

    for yr in range(1, 6):
        if yr == 1:
            if apply_year1:
                cost = base_cost * esc_mult + esc_flat
            else:
                cost = base_cost
        else:
            cost = prev_cost * esc_mult + esc_flat

        cost = max(min_price, cost)
        cost = round(cost)

        rows.append({"Year": yr, "Cost": f"${cost}"})
        prev_cost = cost

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
