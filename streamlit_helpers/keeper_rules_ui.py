"""
Keeper Rules Configuration UI

A comprehensive, user-friendly interface for configuring keeper league rules.
Supports the full spectrum of keeper rule types used across fantasy leagues.
"""

import streamlit as st
from typing import Optional


# =============================================================================
# PRESETS - Common keeper rule configurations
# =============================================================================

KEEPER_PRESETS = {
    "none": {
        "name": "Not a Keeper League",
        "description": "Standard redraft league",
        "config": None
    },
    "simple_auction": {
        "name": "Simple Auction (+$5/year)",
        "description": "Keep at draft price + $5 for each year kept",
        "config": {
            "draft_type": "auction",
            "max_keepers": 3,
            "max_years": 3,
            "base_cost": {"drafted": {"source": "draft_price"}, "faab": {"source": "faab_bid", "flat": 5}},
            "escalation": {"multiplier": 1.0, "flat": 5.0, "starts_year": 2},
            "min_price": 1,
            "max_price": None,
        }
    },
    "auction_faab_max": {
        "name": "Auction (MAX of Draft/FAAB)",
        "description": "Keeper price is higher of draft cost or FAAB-based cost",
        "config": {
            "draft_type": "auction",
            "max_keepers": 3,
            "max_years": 3,
            "base_cost": {"source": "max_of_draft_faab", "faab_multiplier": 0.5, "faab_flat": 10},
            "escalation": {"multiplier": 1.0, "flat": 5.0, "starts_year": 2},
            "min_price": 1,
            "max_price": None,
        }
    },
    "simple_snake": {
        "name": "Simple Snake (-1 round/year)",
        "description": "Keep at draft round, lose 1 round each year",
        "config": {
            "draft_type": "snake",
            "max_keepers": 3,
            "max_years": 3,
            "base_cost": {"source": "draft_round"},
            "escalation": {"rounds_lost": 1},
        }
    },
    "snake_2_rounds": {
        "name": "Snake (-2 rounds/year)",
        "description": "Keep at draft round, lose 2 rounds each year",
        "config": {
            "draft_type": "snake",
            "max_keepers": 2,
            "max_years": 2,
            "base_cost": {"source": "draft_round"},
            "escalation": {"rounds_lost": 2},
        }
    },
    "dynasty": {
        "name": "Dynasty (No Cost)",
        "description": "Keep players indefinitely at no cost",
        "config": {
            "draft_type": "auction",
            "max_keepers": 99,
            "max_years": 99,
            "base_cost": {"source": "fixed", "value": 0},
            "escalation": {"multiplier": 1.0, "flat": 0},
            "min_price": 0,
        }
    },
    "custom": {
        "name": "Custom Rules",
        "description": "Configure your own keeper rules",
        "config": None
    }
}


def render_keeper_rules_ui() -> Optional[dict]:
    """
    Render the keeper rules configuration UI.

    Returns:
        dict with keeper rules configuration, or None if not a keeper league
    """

    # Header
    st.markdown("### Keeper League Settings")

    # Step 1: Quick preset selection
    preset_options = {k: f"{v['name']}" for k, v in KEEPER_PRESETS.items()}

    selected_preset = st.radio(
        "Select your league type:",
        options=list(preset_options.keys()),
        format_func=lambda x: preset_options[x],
        horizontal=True,
        key="keeper_preset"
    )

    # Show preset description
    preset_info = KEEPER_PRESETS[selected_preset]
    if preset_info["description"]:
        st.caption(preset_info["description"])

    # Not a keeper league
    if selected_preset == "none":
        return None

    # Use preset config as starting point
    if selected_preset != "custom" and preset_info["config"]:
        preset_config = preset_info["config"]
        is_auction = preset_config.get("draft_type") == "auction"
    else:
        preset_config = {}
        is_auction = True  # Default to auction for custom

    st.markdown("---")

    # ==========================================================================
    # CUSTOM CONFIGURATION
    # ==========================================================================

    if selected_preset == "custom":
        # Draft Type Selection
        draft_type = st.radio(
            "Draft Type",
            ["Auction (bid $$ on players)", "Snake (pick by round)"],
            horizontal=True,
            key="keeper_draft_type"
        )
        is_auction = draft_type.startswith("Auction")

    # Basic Settings - Always shown for non-preset configurations
    with st.expander("Basic Settings", expanded=(selected_preset == "custom")):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            max_keepers = st.number_input(
                "Max keepers",
                min_value=1, max_value=15,
                value=preset_config.get("max_keepers", 3),
                key="kr_max_keepers",
                help="Maximum players each team can keep"
            )

        with col2:
            if is_auction:
                budget = st.number_input(
                    "Auction budget",
                    min_value=50, max_value=1000,
                    value=preset_config.get("budget", 200),
                    key="kr_budget"
                )
            else:
                budget = 200
                total_rounds = st.number_input(
                    "Draft rounds",
                    min_value=5, max_value=25, value=15,
                    key="kr_total_rounds"
                )

        with col3:
            max_years = st.number_input(
                "Max years kept",
                min_value=1, max_value=99,
                value=preset_config.get("max_years", 3),
                key="kr_max_years",
                help="99 = unlimited"
            )

        with col4:
            min_price = st.number_input(
                "Min price",
                min_value=0, max_value=50,
                value=preset_config.get("min_price", 1),
                key="kr_min_price"
            )

    # ==========================================================================
    # COST RULES
    # ==========================================================================

    base_cost_rules = {}

    if is_auction:
        with st.expander("First Year Cost", expanded=(selected_preset == "custom")):
            st.caption("How is the keeper price determined the first time a player is kept?")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Drafted Players**")
                draft_mult = st.number_input(
                    "× draft price",
                    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                    key="kr_draft_mult",
                    help="1.0 = use draft price as-is"
                )
                draft_flat = st.number_input(
                    "+ flat $",
                    min_value=-50.0, max_value=100.0, value=0.0, step=1.0,
                    key="kr_draft_flat"
                )
                if draft_mult == 1.0 and draft_flat == 0:
                    st.caption("= draft price")
                else:
                    st.caption(f"= {draft_mult}× draft {'+' if draft_flat >= 0 else ''} ${draft_flat:.0f}")

            with col2:
                st.markdown("**FAAB Pickups**")
                faab_mult = st.number_input(
                    "× FAAB bid",
                    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                    key="kr_faab_mult"
                )
                faab_flat = st.number_input(
                    "+ flat $",
                    min_value=-50.0, max_value=100.0, value=10.0, step=1.0,
                    key="kr_faab_flat",
                    help="e.g., $10 + FAAB bid"
                )
                if faab_mult == 1.0 and faab_flat == 0:
                    st.caption("= FAAB bid")
                else:
                    st.caption(f"= {faab_mult}× FAAB {'+' if faab_flat >= 0 else ''} ${faab_flat:.0f}")

            base_cost_rules["auction"] = {"source": "draft_price", "multiplier": draft_mult, "flat": draft_flat}
            base_cost_rules["faab_only"] = {"source": "faab_bid", "multiplier": faab_mult, "flat": faab_flat}
            base_cost_rules["free_agent"] = {"source": "fixed", "value": min_price}

    else:
        # Snake draft
        with st.expander("Keeper Round Rules", expanded=(selected_preset == "custom")):
            snake_base = st.selectbox(
                "Keeper round based on:",
                ["Draft round (where you picked them)", "Fixed round for all keepers"],
                key="kr_snake_base"
            )

            if snake_base == "Fixed round for all keepers":
                fixed_round = st.number_input("Fixed round", min_value=1, max_value=20, value=10, key="kr_fixed_round")
                base_cost_rules["snake"] = {"source": "fixed", "value": fixed_round}
            else:
                base_cost_rules["snake"] = {"source": "draft_round"}

            undrafted_round = st.number_input(
                "Undrafted players kept at round:",
                min_value=1, max_value=20, value=10,
                key="kr_undrafted_round"
            )
            base_cost_rules["undrafted"] = {"source": "fixed", "value": undrafted_round}

    # ==========================================================================
    # ESCALATION
    # ==========================================================================

    formulas_by_keeper_year = {}

    with st.expander("Year-over-Year Escalation", expanded=(selected_preset == "custom")):
        st.caption("How does the keeper cost change each year?")

        if is_auction:
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                esc_mult = st.number_input(
                    "Multiply by",
                    min_value=0.5, max_value=5.0, value=1.0, step=0.1,
                    key="kr_esc_mult",
                    help="1.0 = no multiplier, 1.5 = 50% increase"
                )

            with col2:
                esc_flat = st.number_input(
                    "Then add $",
                    min_value=0.0, max_value=100.0, value=5.0, step=1.0,
                    key="kr_esc_flat"
                )

            with col3:
                esc_year1 = st.checkbox(
                    "From Year 1",
                    value=False,
                    key="kr_esc_year1",
                    help="Apply escalation starting year 1 (vs year 2)"
                )

            # Show formula
            start = "1" if esc_year1 else "2"
            if esc_mult == 1.0 and esc_flat == 0:
                st.info("No escalation")
            elif esc_mult == 1.0:
                st.success(f"+${esc_flat:.0f} per year (starting year {start})")
            elif esc_flat == 0:
                st.success(f"×{esc_mult} per year (starting year {start})")
            else:
                st.success(f"×{esc_mult} + ${esc_flat:.0f} per year (starting year {start})")

            # Build formulas
            if esc_year1:
                formulas_by_keeper_year["1"] = {
                    "expression": f"base_cost * {esc_mult} + {esc_flat}",
                    "multiplier": esc_mult,
                    "flat_add": esc_flat,
                    "apply_year1": True
                }
            else:
                formulas_by_keeper_year["1"] = {"expression": "base_cost", "description": "Base price"}

            formulas_by_keeper_year["2+"] = {
                "expression": f"prev_cost * {esc_mult} + {esc_flat}",
                "multiplier": esc_mult,
                "flat_add": esc_flat,
                "recursive": True
            }

        else:
            # Snake
            rounds_lost = st.number_input(
                "Rounds lost per year",
                min_value=0, max_value=5, value=1,
                key="kr_rounds_lost",
                help="e.g., 1 = Round 8 → 7 → 6 → ..."
            )

            formulas_by_keeper_year["1"] = {"expression": "base_round", "description": "Draft round"}
            formulas_by_keeper_year["2+"] = {
                "expression": f"base_round - {rounds_lost} * (keeper_year - 1)",
                "rounds_lost": rounds_lost
            }

    # ==========================================================================
    # PREVIEW
    # ==========================================================================

    with st.expander("Preview Calculator", expanded=True):
        if is_auction:
            col1, col2, col3 = st.columns(3)
            with col1:
                prev_draft = st.number_input("Draft $", min_value=0, max_value=300, value=25, key="kr_prev_draft")
            with col2:
                prev_faab = st.number_input("FAAB $", min_value=0, max_value=200, value=15, key="kr_prev_faab")
            with col3:
                prev_years = st.number_input("Years", min_value=1, max_value=10, value=5, key="kr_prev_years")

            # Calculate preview
            st.markdown("**Drafted Player Preview**")

            # Get base cost
            draft_base = prev_draft * draft_mult + draft_flat
            draft_base = max(min_price, draft_base)

            preview_rows = []
            prev_cost = draft_base

            for yr in range(1, prev_years + 1):
                if yr == 1:
                    if esc_year1:
                        cost = draft_base * esc_mult + esc_flat
                    else:
                        cost = draft_base
                else:
                    cost = prev_cost * esc_mult + esc_flat

                cost = max(min_price, cost)
                cost = round(cost)

                change = f"+${cost - prev_cost:.0f}" if yr > 1 else "—"
                preview_rows.append({"Year": yr, "Price": f"${cost}", "Change": change})
                prev_cost = cost

            import pandas as pd
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

            # FAAB preview
            if faab_flat != draft_flat or faab_mult != draft_mult:
                st.markdown("**FAAB Pickup Preview**")
                faab_base = prev_faab * faab_mult + faab_flat
                faab_base = max(min_price, faab_base)

                faab_rows = []
                prev_cost = faab_base

                for yr in range(1, prev_years + 1):
                    if yr == 1:
                        if esc_year1:
                            cost = faab_base * esc_mult + esc_flat
                        else:
                            cost = faab_base
                    else:
                        cost = prev_cost * esc_mult + esc_flat

                    cost = max(min_price, cost)
                    cost = round(cost)

                    change = f"+${cost - prev_cost:.0f}" if yr > 1 else "—"
                    faab_rows.append({"Year": yr, "Price": f"${cost}", "Change": change})
                    prev_cost = cost

                st.dataframe(pd.DataFrame(faab_rows), use_container_width=True, hide_index=True)

        else:
            # Snake preview
            col1, col2 = st.columns(2)
            with col1:
                prev_round = st.number_input("Draft round", min_value=1, max_value=20, value=5, key="kr_prev_round")
            with col2:
                prev_years = st.number_input("Years", min_value=1, max_value=10, value=5, key="kr_prev_years_snake")

            rounds_lost_val = formulas_by_keeper_year.get("2+", {}).get("rounds_lost", 0)

            preview_rows = []
            for yr in range(1, prev_years + 1):
                keeper_round = max(1, prev_round - rounds_lost_val * (yr - 1))
                preview_rows.append({"Year": yr, "Round": keeper_round})

            import pandas as pd
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    # ==========================================================================
    # BUILD FINAL CONFIG
    # ==========================================================================

    max_price = None  # Could add UI for this
    round_to_integer = True

    return {
        "enabled": True,
        "draft_type": "auction" if is_auction else "snake",
        "max_keepers": max_keepers,
        "max_years": max_years,
        "budget": budget if is_auction else 200,
        "min_price": min_price,
        "max_price": max_price,
        "round_to_integer": round_to_integer,
        "base_cost_rules": base_cost_rules,
        "formulas_by_keeper_year": formulas_by_keeper_year,
    }
