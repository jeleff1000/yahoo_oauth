"""
Transactions tab components.

Structure:
- Weekly: Add/Drop, Trades, Drop Regrets (transaction-by-transaction)
- Season: Add/Drop, Trades, Report Card (season aggregates)
- Career: Add/Drop, Trades, Report Card (all-time aggregates)
"""

# Weekly views
from .weekly_add_drop import display_weekly_add_drop
from .trade_by_trade_summary_data import display_trade_by_trade_summary_data
from .drop_regret_analysis import display_drop_regret_analysis

# Season views
from .season_add_drop import display_season_add_drop
from .season_trade_data import display_season_trade_data
from .transaction_report_card import display_transaction_report_card

# Career views
from .career_add_drop import display_career_add_drop
from .career_trade_data import display_career_trade_data

# Main entry point
from .transactions_overview import AllTransactionsViewer, display_transactions_overview

__all__ = [
    # Weekly
    "display_weekly_add_drop",
    "display_trade_by_trade_summary_data",
    "display_drop_regret_analysis",
    # Season
    "display_season_add_drop",
    "display_season_trade_data",
    "display_transaction_report_card",
    # Career
    "display_career_add_drop",
    "display_career_trade_data",
    # Main
    "AllTransactionsViewer",
    "display_transactions_overview",
]
