"""
Transactions tab data access.

Optimized loaders for Transactions components with pre-computed engagement metrics.

Key Features:
    - Loads ALL transactions (no row limits)
    - Pre-computed engagement metrics (grades, tiers, regret scores)
    - Pre-aggregated summaries by year and manager
    - Generic table mapping for scalability

Engagement Metrics Available:
    - transaction_grade: A-F grade based on NET SPAR percentile
    - transaction_result: Human-readable result category
    - faab_value_tier: FAAB efficiency tier (Steal → Overpay)
    - drop_regret_score/tier: Regret analysis for drops
    - timing_category: Season timing classification
    - pickup_type: Transaction source type
    - result_emoji: Quick visual indicator
"""

from .transaction_data import load_transaction_data
from .summary_data import load_transaction_summary, load_manager_transaction_summary
from .combined import load_optimized_transactions_data

__all__ = [
    "load_transaction_data",
    "load_transaction_summary",
    "load_manager_transaction_summary",
    "load_optimized_transactions_data",
]
