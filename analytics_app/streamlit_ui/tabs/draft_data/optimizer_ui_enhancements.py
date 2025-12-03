"""
Draft Optimizer UI Enhancements

Drop-in improvements for draft_optimizer.py that add:
1. League intelligence panel
2. Visual efficiency indicators
3. Contextual recommendations
4. Improved budget allocation UI
5. Strategy outcome predictions

Integration:
    Add these imports to draft_optimizer.py:

    from optimizer_ui_enhancements import (
        render_league_insights_panel,
        render_position_efficiency_badges,
        render_budget_allocation_visual,
        render_strategy_outcome_prediction,
        render_smart_recommendations,
        enhance_bench_recommendations
    )

    Then call these functions at appropriate points in display_draft_optimizer()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# Graceful Streamlit import (allows module to be tested without Streamlit)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Try to import plotly for visualizations (graceful fallback if not available)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import league intelligence
try:
    from analytics_app.streamlit_ui.tabs.draft_data.league_intelligence import LeagueIntelligence
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    try:
        from .league_intelligence import LeagueIntelligence
        INTELLIGENCE_AVAILABLE = True
    except ImportError:
        try:
            from league_intelligence import LeagueIntelligence
            INTELLIGENCE_AVAILABLE = True
        except ImportError:
            INTELLIGENCE_AVAILABLE = False


# =============================================================================
# CACHING DECORATOR (works with or without Streamlit)
# =============================================================================

def optional_cache(func):
    """Cache decorator that works with or without Streamlit."""
    if STREAMLIT_AVAILABLE and st is not None:
        return st.cache_data(ttl=3600, show_spinner=False)(func)
    return func


# =============================================================================
# LEAGUE INSIGHTS PANEL
# =============================================================================

@optional_cache
def compute_league_insights(draft_df_json: str) -> Optional[Dict]:
    """
    Compute league insights from draft data.
    Uses JSON string for caching (DataFrames aren't hashable).
    """
    if not INTELLIGENCE_AVAILABLE:
        return None

    try:
        draft_df = pd.read_json(draft_df_json, orient='split')
        intel = LeagueIntelligence(draft_df)
        return {
            'efficiency': intel.calculate_position_efficiency(),
            'cheap_strategy': intel.analyze_cheap_pick_strategy(),
            'patterns': intel.analyze_winning_patterns(),
            'recommendations': intel.analyze().recommendations,
            'expected_edge': intel.analyze().expected_edge_vs_average,
            'confidence': intel.confidence_level,
            'intel': intel  # Keep reference for slot analysis
        }
    except Exception as e:
        print(f"Error computing insights: {e}")
        return None


def render_league_insights_panel(draft_df: pd.DataFrame) -> Optional[Dict]:
    """
    Render the league insights panel at the top of the optimizer.

    Args:
        draft_df: Historical draft data

    Returns:
        Dict of insights for use elsewhere, or None if unavailable
    """
    if draft_df is None or draft_df.empty:
        return None

    # Convert to JSON for caching
    try:
        df_json = draft_df.to_json(orient='split')
        insights = compute_league_insights(df_json)
    except Exception:
        insights = None

    if not insights:
        # Show minimal fallback with debug info
        with st.expander("📊 League Intelligence", expanded=False):
            if not INTELLIGENCE_AVAILABLE:
                st.warning("League Intelligence module not available - check imports")
            elif draft_df is None:
                st.warning("No draft data provided")
            elif draft_df.empty:
                st.warning("Draft data is empty")
            else:
                st.info("Could not compute insights - check data columns")
                st.caption(f"DataFrame has {len(draft_df)} rows, columns: {list(draft_df.columns)[:10]}...")
        return None

    # Render the panel
    with st.expander("🧠 League Intelligence - Market Inefficiencies", expanded=True):
        efficiency = insights.get('efficiency', {})

        if not efficiency:
            st.warning("Not enough data for efficiency analysis")
            return insights

        # Header metrics row
        col1, col2, col3, col4 = st.columns(4)

        # Find extremes
        sorted_eff = sorted(
            [(p, m.efficiency_ratio) for p, m in efficiency.items()],
            key=lambda x: x[1],
            reverse=True
        )

        with col1:
            if sorted_eff:
                best_pos, best_eff = sorted_eff[0]
                st.metric(
                    "🟢 Best Value",
                    best_pos,
                    f"{best_eff:.1f}x efficiency"
                )

        with col2:
            if sorted_eff:
                worst_pos, worst_eff = sorted_eff[-1]
                delta_color = "inverse" if worst_eff < 1 else "normal"
                st.metric(
                    "🔴 Worst Value",
                    worst_pos,
                    f"{worst_eff:.1f}x efficiency",
                    delta_color=delta_color
                )

        with col3:
            edge = insights.get('expected_edge', 0)
            st.metric(
                "📈 Potential Edge",
                f"+{edge:.0f}",
                "SPAR vs avg"
            )

        with col4:
            confidence = insights.get('confidence', 'unknown')
            emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(confidence, '⚪')
            st.metric(
                "📊 Confidence",
                confidence.upper(),
                emoji
            )

        # Efficiency details table
        st.markdown("##### Position Efficiency")

        rows = []
        for pos, metric in efficiency.items():
            eff = metric.efficiency_ratio
            if eff >= 1.2:
                verdict = "🟢 EXPLOIT"
            elif eff <= 0.8:
                verdict = "🔴 AVOID OVERPAY"
            else:
                verdict = "⚪ Fair"

            rows.append({
                'Position': pos,
                'Cost %': f"{metric.cost_share:.1f}%",
                'Value %': f"{metric.value_share:.1f}%",
                'Efficiency': f"{eff:.2f}x",
                'Verdict': verdict,
                'Samples': metric.sample_size
            })

        # Sort by efficiency
        rows.sort(key=lambda x: float(x['Efficiency'].replace('x', '')), reverse=True)

        df_display = pd.DataFrame(rows)
        st.dataframe(df_display, hide_index=True, use_container_width=True)

        # Key recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            st.markdown("##### 💡 Key Recommendations")
            # Debug: show what we received
            st.caption(f"Debug: {len(recommendations)} recs: {[r[:50] if r else 'EMPTY' for r in recommendations[:3]]}")
            for rec in recommendations[:3]:
                if not rec or not rec.strip():
                    continue  # Skip empty recommendations
                if 'EXPLOIT' in rec:
                    st.success(rec)
                elif 'AVOID' in rec:
                    st.error(rec)
                else:
                    st.info(rec)

        # Slot-level efficiency (collapsible)
        with st.expander("🎯 Slot-Level Analysis (QB1 vs QB2, etc.)", expanded=False):
            try:
                intel = insights.get('intel')
                if intel:
                    slot_eff = intel.analyze_slot_efficiency()

                    if slot_eff:
                        slot_rows = []
                        for slot_name, data in sorted(slot_eff.items(), key=lambda x: (x[1]['position'], x[1]['rank'])):
                            rec = data['recommendation']
                            if 'EXPLOIT' in rec:
                                emoji = '🟢'
                            elif 'good value' in rec:
                                emoji = '🔵'
                            elif 'below' in rec or 'poor' in rec:
                                emoji = '🔴'
                            else:
                                emoji = '⚪'

                            slot_rows.append({
                                'Slot': slot_name,
                                'Avg Cost': f"${data['avg_cost']:.0f}",
                                'Avg SPAR': f"{data['avg_spar']:.1f}",
                                'SPAR/$': f"{data['spar_per_dollar']:.1f}",
                                'Recommendation': f"{emoji} {rec}"
                            })

                        st.dataframe(pd.DataFrame(slot_rows), hide_index=True, use_container_width=True)
                        st.caption("Recommendations based on efficiency vs league average. EXPLOIT = 2x+ avg efficiency.")
            except Exception as e:
                st.caption(f"Slot analysis unavailable: {e}")

        # Mathematical Optimal Allocation (collapsible)
        with st.expander("📊 Mathematical Optimal (vs League Average)", expanded=False):
            try:
                intel = insights.get('intel')
                if intel:
                    optimal = intel.calculate_mathematical_optimal()

                    if optimal.get('success'):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("League Avg SPAR", f"{optimal['league_expected_spar']:.0f}")
                        with col2:
                            st.metric("Optimal SPAR", f"{optimal['optimal_expected_spar']:.0f}")
                        with col3:
                            st.metric("Improvement", f"+{optimal['improvement_spar']:.0f}", f"{optimal['improvement_pct']:.1f}%")

                        st.markdown("**Key Opportunities:**")
                        for insight in optimal['insights'][:2]:
                            if 'SPEND MORE' in insight:
                                st.success(insight)
                            elif 'SPEND LESS' in insight:
                                st.info(insight)

                        # Show top 3 changes
                        if optimal['spend_more']:
                            st.markdown("*Underinvested slots:*")
                            for s in optimal['spend_more'][:3]:
                                conf_icon = '✓' if s['confidence'] == 'high' else '~'
                                st.write(f"{conf_icon} **{s['slot']}**: +${s['cost_diff']:.0f} → +{s['spar_gain']:.1f} SPAR")

                        st.caption("This finds TRUE optimal allocation regardless of what your league does.")
                    else:
                        st.caption(f"Optimization unavailable: {optimal.get('error', 'unknown error')}")
            except Exception as e:
                st.caption(f"Mathematical optimization unavailable: {e}")

    return insights


# =============================================================================
# POSITION EFFICIENCY BADGES
# =============================================================================

def render_position_efficiency_badges(
    insights: Optional[Dict],
    positions: List[str] = None
) -> None:
    """
    Render small efficiency badges next to position inputs.
    Call this near your position roster configuration.
    """
    if not insights:
        return

    efficiency = insights.get('efficiency', {})
    if not efficiency:
        return

    if positions is None:
        positions = ['QB', 'RB', 'WR', 'TE', 'DEF', 'K']

    # Create a compact display
    cols = st.columns(len(positions))

    for i, pos in enumerate(positions):
        with cols[i]:
            metric = efficiency.get(pos)
            if metric:
                eff = metric.efficiency_ratio
                if eff >= 1.2:
                    color = "🟢"
                    tip = "Underpaid - good value"
                elif eff <= 0.8:
                    color = "🔴"
                    tip = "Overpaid - be careful"
                else:
                    color = "⚪"
                    tip = "Fair market"

                st.caption(f"{pos}: {color} {eff:.1f}x")


# =============================================================================
# BUDGET ALLOCATION VISUAL
# =============================================================================

def render_budget_allocation_visual(
    allocation: Dict[str, float],
    budget: int,
    insights: Optional[Dict] = None
) -> None:
    """
    Render a visual representation of budget allocation.
    Shows how your allocation compares to league average and top drafters.
    """
    if not PLOTLY_AVAILABLE:
        # Fallback to simple table
        st.markdown("##### Budget Allocation")
        for pos, amount in allocation.items():
            pct = amount / budget * 100
            st.write(f"{pos}: ${amount:.0f} ({pct:.1f}%)")
        return

    positions = list(allocation.keys())
    amounts = [allocation[p] for p in positions]
    percentages = [a / budget * 100 for a in amounts]

    # Get comparison data
    league_avg = {}
    top_drafter = {}

    if insights:
        efficiency = insights.get('efficiency', {})
        patterns = insights.get('patterns', {})

        for pos in positions:
            if pos in efficiency:
                league_avg[pos] = efficiency[pos].cost_share

            pos_diffs = patterns.get('position_differences', {})
            if pos in pos_diffs:
                top_drafter[pos] = pos_diffs[pos].get('top_drafter_allocation', 0)

    # Create grouped bar chart
    fig = go.Figure()

    # Your allocation
    fig.add_trace(go.Bar(
        name='Your Plan',
        x=positions,
        y=percentages,
        marker_color='#667eea',
        text=[f"${a:.0f}" for a in amounts],
        textposition='auto'
    ))

    # League average (if available)
    if league_avg:
        fig.add_trace(go.Bar(
            name='League Avg',
            x=positions,
            y=[league_avg.get(p, 0) for p in positions],
            marker_color='#adb5bd',
            opacity=0.7
        ))

    # Top drafters (if available)
    if top_drafter:
        fig.add_trace(go.Bar(
            name='Top Drafters',
            x=positions,
            y=[top_drafter.get(p, 0) for p in positions],
            marker_color='#28a745',
            opacity=0.7
        ))

    fig.update_layout(
        barmode='group',
        title='Budget Allocation Comparison',
        yaxis_title='% of Budget',
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# STRATEGY OUTCOME PREDICTION
# =============================================================================

def render_strategy_outcome_prediction(
    strategy_name: str,
    allocation: Dict[str, float],
    insights: Optional[Dict],
    budget: int
) -> None:
    """
    Render predicted outcomes for the selected strategy.
    Shows expected SPAR, comparison to alternatives, risk assessment.
    """
    st.markdown("##### 📊 Strategy Prediction")

    col1, col2, col3 = st.columns(3)

    # Calculate expected outcome - use league data if available
    if insights and insights.get('patterns'):
        patterns = insights.get('patterns', {})
        top_avg = patterns.get('top_avg_spar', 0)
        bottom_avg = patterns.get('bottom_avg_spar', 0)
        # Base SPAR is midpoint between top and bottom
        if top_avg and bottom_avg:
            base_spar = (top_avg + bottom_avg) / 2
        else:
            base_spar = 650  # Fallback only if no data
    else:
        base_spar = 650  # Fallback only if no insights

    edge = insights.get('expected_edge', 0) if insights else 0

    # Adjust based on strategy - use percentages of the edge, not fixed values
    strategy_modifiers = {
        'Balanced': 0,
        'Zero RB': -edge * 0.1 if edge else -10,  # Higher variance
        'Hero RB': edge * 0.05 if edge else 5,
        'Robust RB': edge * 0.1 if edge else 10,
        'Late-Round QB': edge * 0.15 if edge else 15,
        'Stars & Scrubs': -edge * 0.05 if edge else -5,
        'League-Optimized': edge,
        '🌟 League-Optimized': edge,
        'Punt TE': edge * 0.5 if edge else 5
    }

    modifier = strategy_modifiers.get(strategy_name, 0)
    expected_spar = base_spar + modifier

    with col1:
        st.metric(
            "Expected Total SPAR",
            f"{expected_spar:.0f}",
            f"+{modifier:.0f} vs baseline" if modifier != 0 else "baseline"
        )

    with col2:
        # Risk assessment
        high_risk = ['Zero RB', 'Hero RB', 'Stars & Scrubs']
        low_risk = ['Balanced', 'Robust RB']

        if strategy_name in high_risk:
            risk = "🔴 High"
            risk_desc = "High variance - big upside, bigger downside"
        elif strategy_name in low_risk:
            risk = "🟢 Low"
            risk_desc = "Consistent, predictable outcomes"
        else:
            risk = "🟡 Medium"
            risk_desc = "Balanced risk/reward"

        st.metric("Risk Level", risk)
        st.caption(risk_desc)

    with col3:
        # Confidence in prediction
        if insights:
            confidence = insights.get('confidence', 'medium')
            conf_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(confidence, '⚪')
            conf_pct = {'high': 85, 'medium': 65, 'low': 45}.get(confidence, 50)
        else:
            conf_emoji = '⚪'
            conf_pct = 50

        st.metric("Prediction Confidence", f"{conf_emoji} {conf_pct}%")


# =============================================================================
# SMART RECOMMENDATIONS
# =============================================================================

def render_smart_recommendations(
    insights: Optional[Dict],
    current_strategy: str,
    current_allocation: Dict[str, float],
    budget: int
) -> None:
    """
    Render contextual recommendations based on current choices.
    """
    if not insights:
        return

    st.markdown("##### 💡 Smart Suggestions")

    efficiency = insights.get('efficiency', {})
    recommendations = []

    # Check for inefficiencies in allocation
    for pos, amount in current_allocation.items():
        if pos not in efficiency:
            continue

        metric = efficiency[pos]
        pct = amount / budget * 100
        league_pct = metric.cost_share

        if metric.efficiency_ratio >= 1.2 and pct < league_pct - 5:
            recommendations.append({
                'type': 'increase',
                'position': pos,
                'message': f"Consider increasing {pos} spend. It's {metric.efficiency_ratio:.1f}x efficient but you're allocating less than average."
            })

        elif metric.efficiency_ratio <= 0.8 and pct > league_pct + 5:
            recommendations.append({
                'type': 'decrease',
                'position': pos,
                'message': f"Consider reducing {pos} spend. It's only {metric.efficiency_ratio:.1f}x efficient and you're spending more than average."
            })

    # Cheap pick recommendation
    cheap_strategy = insights.get('cheap_strategy', {})
    if cheap_strategy.get('optimal_count_range'):
        optimal = cheap_strategy['optimal_count_range']
        priority = cheap_strategy.get('position_priority', [])[:3]
        recommendations.append({
            'type': 'info',
            'position': 'BENCH',
            'message': f"Target {optimal} cheap picks. Best value at: {', '.join(priority)}"
        })

    # Display recommendations
    if recommendations:
        for rec in recommendations[:4]:
            if rec['type'] == 'increase':
                st.success(f"📈 {rec['message']}")
            elif rec['type'] == 'decrease':
                st.warning(f"📉 {rec['message']}")
            else:
                st.info(f"💡 {rec['message']}")
    else:
        st.success("✅ Your allocation looks well-optimized!")


# =============================================================================
# ENHANCED BENCH RECOMMENDATIONS
# =============================================================================

def enhance_bench_recommendations(
    bench_budget: float,
    bench_spots: int,
    insights: Optional[Dict]
) -> Dict:
    """
    Provide enhanced bench recommendations based on league intelligence.

    Returns configuration dict for bench planning.
    """
    result = {
        'budget': bench_budget,
        'spots': bench_spots,
        'avg_per_spot': bench_budget / bench_spots if bench_spots > 0 else 0,
        'recommendations': [],
        'position_priority': ['RB', 'WR', 'QB', 'TE'],
        'optimal_cheap_count': 5
    }

    if not insights:
        result['recommendations'].append("💡 Add league_intelligence.py for data-driven bench insights")
        return result

    cheap_strategy = insights.get('cheap_strategy', {})

    # Update with data-driven values
    if cheap_strategy:
        result['position_priority'] = cheap_strategy.get('position_priority', result['position_priority'])

        optimal_range = cheap_strategy.get('optimal_count_range', '5-6')
        if '-' in str(optimal_range):
            parts = str(optimal_range).split('-')
            result['optimal_cheap_count'] = int(parts[0])

        threshold = cheap_strategy.get('cheap_threshold', 3)
        expected_spar = cheap_strategy.get('expected_spar_per_pick', 15)

        result['recommendations'].extend([
            f"🎯 Target {result['optimal_cheap_count']}-{result['optimal_cheap_count']+1} picks at ${threshold:.0f} or less",
            f"📊 Expected SPAR per cheap pick: {expected_spar:.1f}",
            f"✅ Priority: {', '.join(result['position_priority'][:3])}"
        ])

        # Warn about low-value positions
        if result['position_priority']:
            worst = result['position_priority'][-1]
            pos_details = {d.get('yahoo_position', ''): d
                         for d in cheap_strategy.get('position_details', [])}
            worst_spar = pos_details.get(worst, {}).get('avg_spar', 0)
            if worst_spar < 10:
                result['recommendations'].append(f"⚠️ Avoid cheap {worst}s ({worst_spar:.1f} avg SPAR)")

    return result


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def integrate_into_optimizer(draft_df: pd.DataFrame) -> Dict:
    """
    Main integration function. Call this at the start of display_draft_optimizer()
    to get all the enhancement data.

    Returns:
        Dict with 'insights' and helper functions
    """
    insights = None

    if INTELLIGENCE_AVAILABLE and draft_df is not None and not draft_df.empty:
        try:
            df_json = draft_df.to_json(orient='split')
            insights = compute_league_insights(df_json)
        except Exception:
            pass

    return {
        'insights': insights,
        'render_insights_panel': lambda: render_league_insights_panel(draft_df),
        'render_efficiency_badges': lambda positions=None: render_position_efficiency_badges(insights, positions),
        'render_allocation_visual': lambda alloc, budget: render_budget_allocation_visual(alloc, budget, insights),
        'render_predictions': lambda strat, alloc, budget: render_strategy_outcome_prediction(strat, alloc, insights, budget),
        'render_recommendations': lambda strat, alloc, budget: render_smart_recommendations(insights, strat, alloc, budget),
        'get_bench_config': lambda bench_budget, bench_spots: enhance_bench_recommendations(bench_budget, bench_spots, insights)
    }
