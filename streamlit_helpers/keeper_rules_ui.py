"""
Keeper/Dynasty League Configuration Wizard

A 3-step wizard for configuring keeper league rules during initial import.
Supports multiple escalation types including "from base" and "compounding" formulas.
"""

import streamlit as st
import pandas as pd
from typing import Optional


def render_keeper_rules_ui() -> Optional[dict]:
    """
    Render keeper rules configuration wizard.

    Returns:
        dict with keeper rules configuration, or None if not a keeper league
    """
    # Initialize session state
    if "keeper_wizard_step" not in st.session_state:
        st.session_state.keeper_wizard_step = 0
    if "keeper_values" not in st.session_state:
        st.session_state.keeper_values = _get_defaults()

    v = st.session_state.keeper_values
    existing_config = st.session_state.get("configured_keeper_rules")

    # Load existing config into values if first time
    if existing_config and not v.get("_loaded"):
        _load_from_config(v, existing_config)
        v["_loaded"] = True

    # Quick toggle
    is_keeper = st.checkbox(
        "This is a Keeper/Dynasty League",
        value=v.get("enabled", False),
        key="keeper_toggle"
    )
    v["enabled"] = is_keeper

    if not is_keeper:
        st.session_state.configured_keeper_rules = None
        return None

    st.markdown("---")

    # Two-column layout: Wizard (left) + Live Summary (right)
    col_wizard, col_summary = st.columns([2, 1])

    with col_wizard:
        _render_wizard(v)

    with col_summary:
        _render_live_summary(v)

    # Return config if complete
    return st.session_state.get("configured_keeper_rules")


def _get_defaults() -> dict:
    """Get default values for keeper configuration."""
    return {
        "enabled": False,
        "draft_type": "auction",
        "max_keepers": 3,
        "max_years": 3,
        "min_price": 1,
        "budget": 200,
        # First-year cost
        "draft_mult": 1.0,
        "draft_flat": 0.0,
        "faab_mult": 1.0,
        "faab_flat": 10.0,
        # Escalation
        "esc_type": "from_base",  # "from_base" or "compounding"
        "esc_flat_per_year": 5.0,  # For from_base: adds this × years_kept
        "esc_mult": 1.0,  # For compounding: prev_cost × this
        "esc_flat": 0.0,  # For compounding: + this each year
        "_loaded": False,
    }


def _load_from_config(v: dict, config: dict):
    """Load values from existing configuration."""
    v["draft_type"] = config.get("draft_type", "auction")
    v["max_keepers"] = config.get("max_keepers", 3)
    v["max_years"] = config.get("max_years") or 3
    v["min_price"] = config.get("min_price", 1)
    v["budget"] = config.get("budget", 200)

    base = config.get("base_cost_rules", {})
    auction = base.get("auction", {})
    faab = base.get("faab_only", {})

    v["draft_mult"] = auction.get("multiplier", 1.0)
    v["draft_flat"] = auction.get("flat", 0.0)
    v["faab_mult"] = faab.get("multiplier", 1.0)
    v["faab_flat"] = faab.get("flat", 10.0)

    # Load escalation config
    esc = config.get("formulas_by_keeper_year", {}).get("2+", {})
    v["esc_type"] = esc.get("type", "from_base")
    v["esc_flat_per_year"] = esc.get("flat_per_year", 5.0)
    v["esc_mult"] = esc.get("multiplier", 1.0)
    v["esc_flat"] = esc.get("flat_add", 0.0)


def _render_wizard(v: dict):
    """Render the 3-step wizard."""
    steps = ["Basic Rules", "First-Year Price", "Escalation & Preview"]
    current = st.session_state.keeper_wizard_step

    # Step indicator
    cols = st.columns(len(steps))
    for i, (col, name) in enumerate(zip(cols, steps)):
        with col:
            if i == current:
                st.markdown(f"**{i+1}. {name}**")
            elif i < current:
                if st.button(f"{i+1}. {name}", key=f"nav_{i}", use_container_width=True):
                    st.session_state.keeper_wizard_step = i
                    st.rerun()
            else:
                st.markdown(f":gray[{i+1}. {name}]")

    st.markdown("---")

    # Render current step
    if current == 0:
        _render_step_basic(v)
    elif current == 1:
        _render_step_first_year(v)
    else:
        _render_step_escalation(v)


def _render_step_basic(v: dict):
    """Step 1: Basic Rules."""
    st.markdown("### Step 1: Basic Rules")
    st.caption("Define your league's keeper format and constraints.")

    # Draft type
    st.markdown("**What type of draft does your league use?**")
    draft_type = st.radio(
        "Draft type",
        ["Auction", "Snake"],
        index=0 if v["draft_type"] == "auction" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="step1_draft_type"
    )
    v["draft_type"] = "auction" if draft_type == "Auction" else "snake"

    st.markdown("")

    # Core limits
    col1, col2 = st.columns(2)
    with col1:
        v["max_keepers"] = st.number_input(
            "Maximum keepers per team",
            min_value=1, max_value=15,
            value=v["max_keepers"],
            key="step1_max_keepers"
        )
    with col2:
        max_years = st.number_input(
            "Maximum years kept",
            min_value=0, max_value=20,
            value=v["max_years"] if v["max_years"] else 0,
            help="0 = unlimited",
            key="step1_max_years"
        )
        v["max_years"] = max_years if max_years > 0 else None

    # Auction-specific
    if v["draft_type"] == "auction":
        col1, col2 = st.columns(2)
        with col1:
            v["budget"] = st.number_input(
                "Auction budget ($)",
                min_value=50, max_value=1000,
                value=v["budget"],
                key="step1_budget"
            )
        with col2:
            v["min_price"] = st.number_input(
                "Minimum keeper price ($)",
                min_value=0, max_value=50,
                value=v["min_price"],
                key="step1_min_price"
            )

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Next", type="primary", use_container_width=True, key="step1_next"):
            st.session_state.keeper_wizard_step = 1
            st.rerun()


def _render_step_first_year(v: dict):
    """Step 2: First-Year Keeper Cost."""
    st.markdown("### Step 2: First-Year Keeper Cost")
    st.caption("How is the keeper price calculated the FIRST time a player is kept?")

    if v["draft_type"] == "auction":
        # Drafted players
        st.markdown("**Drafted Players**")
        draft_cost = _calc_first_year(25, v["draft_mult"], v["draft_flat"], v["min_price"])
        st.success(f"A $25 drafted player costs **${draft_cost}** to keep (Year 1)")

        with st.expander("Edit formula"):
            col1, col2 = st.columns(2)
            with col1:
                v["draft_mult"] = st.number_input(
                    "Multiply draft price by",
                    min_value=0.1, max_value=5.0,
                    value=float(v["draft_mult"]),
                    step=0.1,
                    key="step2_draft_mult"
                )
            with col2:
                v["draft_flat"] = st.number_input(
                    "Then add ($)",
                    min_value=-50.0, max_value=100.0,
                    value=float(v["draft_flat"]),
                    step=1.0,
                    key="step2_draft_flat"
                )

        st.markdown("")

        # FAAB pickups
        st.markdown("**FAAB/Waiver Pickups**")
        faab_cost = _calc_first_year(15, v["faab_mult"], v["faab_flat"], v["min_price"])
        st.success(f"A $15 FAAB pickup costs **${faab_cost}** to keep (Year 1)")

        with st.expander("Edit formula"):
            col1, col2 = st.columns(2)
            with col1:
                v["faab_mult"] = st.number_input(
                    "Multiply FAAB bid by",
                    min_value=0.1, max_value=5.0,
                    value=float(v["faab_mult"]),
                    step=0.1,
                    key="step2_faab_mult"
                )
            with col2:
                v["faab_flat"] = st.number_input(
                    "Then add ($)",
                    min_value=-50.0, max_value=100.0,
                    value=float(v["faab_flat"]),
                    step=1.0,
                    key="step2_faab_flat"
                )
    else:
        st.info("Snake draft keeper costs are based on draft round.")
        st.markdown("Players are kept at the round they were drafted.")

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Back", use_container_width=True, key="step2_back"):
            st.session_state.keeper_wizard_step = 0
            st.rerun()
    with col3:
        if st.button("Next", type="primary", use_container_width=True, key="step2_next"):
            st.session_state.keeper_wizard_step = 2
            st.rerun()


def _render_step_escalation(v: dict):
    """Step 3: Escalation & Preview."""
    st.markdown("### Step 3: Escalation & Preview")
    st.caption("How does the keeper cost change each additional year?")

    if v["draft_type"] == "auction":
        # Escalation type selector
        st.markdown("**How does escalation work?**")

        esc_options = [
            "Add flat amount per year (from original cost)",
            "Compound from previous year's price",
            "No escalation (same price every year)"
        ]

        # Map current value to index
        if v["esc_type"] == "none":
            esc_idx = 2
        elif v["esc_type"] == "compounding":
            esc_idx = 1
        else:
            esc_idx = 0

        esc_choice = st.radio(
            "Escalation type",
            esc_options,
            index=esc_idx,
            label_visibility="collapsed",
            key="step3_esc_type"
        )

        # Update type based on selection
        if "flat amount per year" in esc_choice:
            v["esc_type"] = "from_base"
        elif "Compound" in esc_choice:
            v["esc_type"] = "compounding"
        else:
            v["esc_type"] = "none"

        st.markdown("")

        # Show appropriate controls based on type
        if v["esc_type"] == "from_base":
            # Example: draft_price + (5 × years_kept)
            example_yr2 = _calc_from_base(25, v["draft_mult"], v["draft_flat"], v["esc_flat_per_year"], 2, v["min_price"])
            example_yr3 = _calc_from_base(25, v["draft_mult"], v["draft_flat"], v["esc_flat_per_year"], 3, v["min_price"])

            st.info(f"Year 2: ${example_yr2} | Year 3: ${example_yr3} | (+${v['esc_flat_per_year']:.0f} each year)")

            with st.expander("Edit escalation"):
                v["esc_flat_per_year"] = st.number_input(
                    "Add this amount per year kept ($)",
                    min_value=0.0, max_value=50.0,
                    value=float(v["esc_flat_per_year"]),
                    step=1.0,
                    key="step3_flat_per_year",
                    help="e.g., $5 means Year 1 = base, Year 2 = base + $5, Year 3 = base + $10"
                )

        elif v["esc_type"] == "compounding":
            # Example: prev_cost × mult + flat
            example_yr2 = _calc_compounding(25, v["draft_mult"], v["draft_flat"], v["esc_mult"], v["esc_flat"], 2, v["min_price"])
            example_yr3 = _calc_compounding(25, v["draft_mult"], v["draft_flat"], v["esc_mult"], v["esc_flat"], 3, v["min_price"])

            esc_desc = _format_compounding(v["esc_mult"], v["esc_flat"])
            st.info(f"Year 2: ${example_yr2} | Year 3: ${example_yr3} | ({esc_desc})")

            with st.expander("Edit escalation"):
                col1, col2 = st.columns(2)
                with col1:
                    v["esc_mult"] = st.number_input(
                        "Multiply previous year by",
                        min_value=0.5, max_value=3.0,
                        value=float(v["esc_mult"]),
                        step=0.1,
                        key="step3_esc_mult"
                    )
                with col2:
                    v["esc_flat"] = st.number_input(
                        "Then add ($)",
                        min_value=0.0, max_value=100.0,
                        value=float(v["esc_flat"]),
                        step=1.0,
                        key="step3_esc_flat"
                    )

        else:
            st.info("Keeper cost stays the same every year.")

        # Preview tables
        st.markdown("---")
        st.markdown("### Preview: Cost Progression")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**$25 Drafted Player**")
            _show_preview_table(v, 25, "draft")
        with col2:
            st.markdown("**$15 FAAB Pickup**")
            _show_preview_table(v, 15, "faab")
    else:
        st.info("Snake draft escalation coming soon.")

    # Navigation + Save
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Back", use_container_width=True, key="step3_back"):
            st.session_state.keeper_wizard_step = 1
            st.rerun()

    with col3:
        if st.button("Save Configuration", type="primary", use_container_width=True, key="step3_save"):
            config = _build_config(v)
            st.session_state.configured_keeper_rules = config
            st.success("Keeper rules saved! Ready for import.")


def _render_live_summary(v: dict):
    """Right sidebar: live summary."""
    st.markdown("### Your Rules")
    st.markdown("---")

    st.markdown(f"**Draft Type:** {v.get('draft_type', 'auction').title()}")
    st.markdown(f"**Max Keepers:** {v.get('max_keepers', 3)}")

    max_years = v.get('max_years')
    st.markdown(f"**Max Years:** {max_years if max_years else 'Unlimited'}")

    if v.get("draft_type") == "auction":
        st.markdown(f"**Budget:** ${v.get('budget', 200)}")
        st.markdown(f"**Min Price:** ${v.get('min_price', 1)}")

        st.markdown("---")
        st.markdown("**First-Year Cost**")

        draft_formula = _format_first_year(v.get("draft_mult", 1.0), v.get("draft_flat", 0.0), "draft price")
        faab_formula = _format_first_year(v.get("faab_mult", 1.0), v.get("faab_flat", 10.0), "FAAB bid")

        st.caption(f"Drafted: {draft_formula}")
        st.caption(f"FAAB: {faab_formula}")

        st.markdown("---")
        st.markdown("**Escalation**")

        esc_type = v.get("esc_type", "from_base")
        if esc_type == "from_base":
            flat_per = v.get("esc_flat_per_year", 5.0)
            st.caption(f"+${flat_per:.0f} per year kept")
            st.caption("(from original cost)")
        elif esc_type == "compounding":
            st.caption(_format_compounding(v.get("esc_mult", 1.0), v.get("esc_flat", 0.0)))
            st.caption("(compounds each year)")
        else:
            st.caption("None")

    # Status indicator
    st.markdown("---")
    if st.session_state.get("configured_keeper_rules"):
        st.success("Saved")
    else:
        st.warning("Not saved")


def _format_first_year(mult: float, flat: float, source: str) -> str:
    """Format first-year formula as friendly text."""
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


def _format_compounding(mult: float, flat: float) -> str:
    """Format compounding escalation."""
    parts = []
    if mult != 1.0:
        parts.append(f"{mult}x prev")
    if flat > 0:
        parts.append(f"+ ${flat:.0f}")
    if not parts:
        return "no change"
    return " ".join(parts)


def _calc_first_year(base: float, mult: float, flat: float, min_price: int) -> int:
    """Calculate first-year keeper cost."""
    cost = base * mult + flat
    return max(min_price, round(cost))


def _calc_from_base(base: float, mult: float, flat: float, flat_per_year: float, year: int, min_price: int) -> int:
    """Calculate keeper cost using from-base escalation."""
    # Year 1: base_cost
    # Year 2: base_cost + flat_per_year
    # Year 3: base_cost + 2 * flat_per_year
    base_cost = base * mult + flat
    cost = base_cost + flat_per_year * (year - 1)
    return max(min_price, round(cost))


def _calc_compounding(base: float, mult: float, flat: float, esc_mult: float, esc_flat: float, year: int, min_price: int) -> int:
    """Calculate keeper cost using compounding escalation."""
    base_cost = base * mult + flat
    cost = base_cost
    for _ in range(1, year):
        cost = cost * esc_mult + esc_flat
    return max(min_price, round(cost))


def _show_preview_table(v: dict, base_value: float, acq_type: str):
    """Show preview table of cost progression."""
    if acq_type == "draft":
        mult = v.get("draft_mult", 1.0)
        flat = v.get("draft_flat", 0.0)
    else:
        mult = v.get("faab_mult", 1.0)
        flat = v.get("faab_flat", 10.0)

    min_price = v.get("min_price", 1)
    max_years = v.get("max_years") or 10
    esc_type = v.get("esc_type", "from_base")

    rows = []
    prev_cost = None

    for yr in range(1, min(6, max_years + 1) if max_years else 6):
        if esc_type == "from_base":
            cost = _calc_from_base(base_value, mult, flat, v.get("esc_flat_per_year", 5.0), yr, min_price)
        elif esc_type == "compounding":
            cost = _calc_compounding(base_value, mult, flat, v.get("esc_mult", 1.0), v.get("esc_flat", 0.0), yr, min_price)
        else:
            cost = _calc_first_year(base_value, mult, flat, min_price)

        if prev_cost is not None and cost > prev_cost:
            change = f"+${cost - prev_cost:.0f}"
        else:
            change = "-"

        rows.append({"Year": yr, "Cost": f"${cost}", "Change": change})
        prev_cost = cost

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _build_config(v: dict) -> dict:
    """Build configuration dict from wizard values."""
    base_cost_rules = {
        "auction": {"source": "draft_price", "multiplier": v["draft_mult"], "flat": v["draft_flat"]},
        "faab_only": {"source": "faab_bid", "multiplier": v["faab_mult"], "flat": v["faab_flat"]},
        "free_agent": {"source": "fixed", "value": v["min_price"]}
    }

    esc_type = v.get("esc_type", "from_base")

    formulas = {}

    if esc_type == "from_base":
        # Year 1: base_cost
        # Year 2+: base_cost + flat_per_year * (keeper_year - 1)
        formulas["1"] = {"expression": "base_cost"}
        formulas["2+"] = {
            "type": "from_base",
            "expression": f"base_cost + {v['esc_flat_per_year']} * (keeper_year - 1)",
            "flat_per_year": v["esc_flat_per_year"],
        }
    elif esc_type == "compounding":
        # Year 1: base_cost
        # Year 2+: prev_cost * mult + flat
        formulas["1"] = {"expression": "base_cost"}
        formulas["2+"] = {
            "type": "compounding",
            "expression": f"prev_cost * {v['esc_mult']} + {v['esc_flat']}",
            "multiplier": v["esc_mult"],
            "flat_add": v["esc_flat"],
            "recursive": True
        }
    else:
        # No escalation
        formulas["1"] = {"expression": "base_cost"}
        formulas["2+"] = {
            "type": "none",
            "expression": "base_cost",
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
