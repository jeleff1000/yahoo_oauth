"""
Draft Enrichment Modules

Provides modular functions for draft value analysis:
- draft_type_utils: Draft type detection utilities (auction vs snake)
- spar_calculator: SPAR (Season Points Above Replacement) metrics
- draft_grade_calculator: Draft grades based on SPAR percentile
- value_tier_calculator: Value tiers based on draft position vs finish rank
- draft_flags_calculator: Dynamic breakout/bust flags and draft tiers
- manager_draft_grade_calculator: Manager-level draft grades

Key Features:
- Automatically detects auction vs snake draft type PER YEAR from data
- Handles mixed datasets (some years auction, some years snake)
- Uses appropriate columns/methods based on detected draft type
"""

from .draft_type_utils import (
    detect_draft_type_for_year,
    detect_draft_type_per_year,
    add_draft_type_column,
    get_cost_column_for_year,
    get_peer_group_column,
    get_rank_delta_column_for_year,
    detect_budget_for_year,
    get_normalized_cost,
    get_weight_method_for_year,
    summarize_draft_types,
    AUCTION_TYPES,
    SNAKE_TYPES
)

from .spar_calculator import (
    calculate_all_draft_metrics,
    calculate_draft_spar,
    calculate_draft_roi,
    normalize_draft_cost,
    calculate_additional_draft_metrics
)

from .draft_grade_calculator import (
    calculate_draft_grades,
    calculate_spar_percentile,
    assign_draft_grade,
    get_grade_distribution,
    get_grade_summary_stats,
    DEFAULT_PERCENTILE_BINS,
    DEFAULT_GRADE_LABELS
)

from .value_tier_calculator import (
    calculate_value_tiers,
    calculate_rank_delta,
    assign_value_tier,
    detect_draft_type,
    get_rank_delta_column,
    get_value_tier_distribution,
    get_value_tier_summary_stats,
    get_unified_rank_delta,
    DEFAULT_VALUE_BINS,
    DEFAULT_VALUE_LABELS
)

from .draft_flags_calculator import (
    calculate_all_draft_flags,
    calculate_round_percentile,
    calculate_breakout_flag,
    calculate_bust_flag,
    calculate_draft_tier,
    calculate_draft_tier_with_rounds,
    calculate_games_missed,
    calculate_injury_flag,
    calculate_bust_type,
    get_round_thresholds,
    get_flag_summary,
    get_tier_distribution,
    DEFAULT_LATE_ROUND_PERCENTILE,
    DEFAULT_EARLY_ROUND_PERCENTILE,
    DEFAULT_TOP_FINISH_PERCENTILE,
    DEFAULT_BOTTOM_FINISH_PERCENTILE,
    DEFAULT_TIER_PERCENTILES,
    DEFAULT_INJURY_GAMES_MISSED_PCT
)

from .manager_draft_grade_calculator import (
    calculate_manager_draft_grades,
    calculate_pick_quality_score,
    calculate_pick_weight,
    apply_asymmetric_scoring,
    calculate_weighted_manager_score,
    aggregate_manager_draft_stats,
    calculate_manager_draft_percentile,
    assign_manager_draft_grade,
    calculate_manager_draft_score,
    get_manager_draft_leaderboard,
    get_manager_career_grades,
    DEFAULT_MANAGER_PERCENTILE_BINS,
    DEFAULT_MANAGER_GRADE_LABELS,
    DEFAULT_ROUND_WEIGHT_DECAY,
    DEFAULT_EARLY_MISS_PENALTY,
    DEFAULT_EARLY_HIT_BONUS,
    DEFAULT_LATE_MISS_PENALTY,
    DEFAULT_LATE_HIT_BONUS
)

from .starter_designation_calculator import (
    calculate_starter_designation,
    calculate_position_draft_rank,
    load_roster_structure,
    get_starter_slots,
    get_starter_designation_summary,
    get_spar_by_starter_designation,
    DEFAULT_ROSTER,
    FLEX_ELIGIBLE_POSITIONS
)

from .bench_insurance_calculator import (
    calculate_bench_insurance_metrics,
    calculate_starter_failure_rates,
    calculate_bench_activation_metrics,
    calculate_insurance_value,
    calculate_manager_bench_patterns,
    get_optimal_bench_composition,
    get_position_bench_discount,
    calculate_bench_value_by_rank,
    get_bench_value_for_rank
)

__all__ = [
    # Draft type utilities (shared detection logic)
    'detect_draft_type_for_year',
    'detect_draft_type_per_year',
    'add_draft_type_column',
    'get_cost_column_for_year',
    'get_peer_group_column',
    'get_rank_delta_column_for_year',
    'detect_budget_for_year',
    'get_normalized_cost',
    'get_weight_method_for_year',
    'summarize_draft_types',
    'AUCTION_TYPES',
    'SNAKE_TYPES',
    # SPAR calculator
    'calculate_all_draft_metrics',
    'calculate_draft_spar',
    'calculate_draft_roi',
    'normalize_draft_cost',
    'calculate_additional_draft_metrics',
    # Draft grade calculator
    'calculate_draft_grades',
    'calculate_spar_percentile',
    'assign_draft_grade',
    'get_grade_distribution',
    'get_grade_summary_stats',
    'DEFAULT_PERCENTILE_BINS',
    'DEFAULT_GRADE_LABELS',
    # Value tier calculator
    'calculate_value_tiers',
    'calculate_rank_delta',
    'assign_value_tier',
    'detect_draft_type',
    'get_rank_delta_column',
    'get_value_tier_distribution',
    'get_value_tier_summary_stats',
    'get_unified_rank_delta',
    'DEFAULT_VALUE_BINS',
    'DEFAULT_VALUE_LABELS',
    # Draft flags calculator
    'calculate_all_draft_flags',
    'calculate_round_percentile',
    'calculate_breakout_flag',
    'calculate_bust_flag',
    'calculate_draft_tier',
    'calculate_draft_tier_with_rounds',
    'calculate_games_missed',
    'calculate_injury_flag',
    'calculate_bust_type',
    'get_round_thresholds',
    'get_flag_summary',
    'get_tier_distribution',
    'DEFAULT_LATE_ROUND_PERCENTILE',
    'DEFAULT_EARLY_ROUND_PERCENTILE',
    'DEFAULT_TOP_FINISH_PERCENTILE',
    'DEFAULT_BOTTOM_FINISH_PERCENTILE',
    'DEFAULT_TIER_PERCENTILES',
    'DEFAULT_INJURY_GAMES_MISSED_PCT',
    # Manager draft grade calculator
    'calculate_manager_draft_grades',
    'calculate_pick_quality_score',
    'calculate_pick_weight',
    'apply_asymmetric_scoring',
    'calculate_weighted_manager_score',
    'aggregate_manager_draft_stats',
    'calculate_manager_draft_percentile',
    'assign_manager_draft_grade',
    'calculate_manager_draft_score',
    'get_manager_draft_leaderboard',
    'get_manager_career_grades',
    'DEFAULT_MANAGER_PERCENTILE_BINS',
    'DEFAULT_MANAGER_GRADE_LABELS',
    'DEFAULT_ROUND_WEIGHT_DECAY',
    'DEFAULT_EARLY_MISS_PENALTY',
    'DEFAULT_EARLY_HIT_BONUS',
    'DEFAULT_LATE_MISS_PENALTY',
    'DEFAULT_LATE_HIT_BONUS',
    # Starter designation calculator
    'calculate_starter_designation',
    'calculate_position_draft_rank',
    'load_roster_structure',
    'get_starter_slots',
    'get_starter_designation_summary',
    'get_spar_by_starter_designation',
    'DEFAULT_ROSTER',
    'FLEX_ELIGIBLE_POSITIONS',
    # Bench insurance calculator
    'calculate_bench_insurance_metrics',
    'calculate_starter_failure_rates',
    'calculate_bench_activation_metrics',
    'calculate_insurance_value',
    'calculate_manager_bench_patterns',
    'get_optimal_bench_composition',
    'get_position_bench_discount',
    'calculate_bench_value_by_rank',
    'get_bench_value_for_rank',
]
