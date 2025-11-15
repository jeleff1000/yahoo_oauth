# Multi-League Data Dictionary

A comprehensive glossary of all columns across all data sources in the multi-league fantasy football system.

**Purpose:**
- Document what each column means
- Identify primary/foreign keys for merging
- Track data lineage (where columns come from)
- Prevent naming conflicts
- Enable clean cross-file merges

**Last Updated:** 2025-01-01

**Recent Changes (2025-01-01):**
- 🔧 **MAJOR REFACTORING:** Modularized codebase for reusability
  - Created `multi_league/core/data_normalization.py` - Centralized data type handling
  - Created `multi_league/core/script_runner.py` - Script orchestration utilities
  - Created `multi_league/data_fetchers/aggregators.py` - File aggregation functions
  - Created `multi_league/core/yahoo_league_settings.py` - Unified league settings fetcher
- 🔧 **LEAGUE SETTINGS CONSOLIDATED:** ONE API call, ONE comprehensive JSON file
  - New output: `league_settings_{year}_{league_key}.json` (replaces 3+ fragmented files)
  - Deleted redundant modules: `pull_scoring_rules.py`, `yahoo_settings.py`
- ✅ All column definitions unchanged - refactoring was non-breaking
- ✅ Data normalization now uses centralized `normalize_numeric_columns()` function

**Previous Changes (2025-10-28):**
- ✅ Fixed `league_id` population (was NULL, now properly populated from LeagueContext)
- ✅ Extended player data coverage to 1999-2025 (509K+ rows vs 115K previously)
- ✅ Added 34 playoff odds columns to matchup data (p_playoffs, p_bye, x*_seed, x*_win, power_rating)
- ✅ All documented columns now properly created by pipeline

Table of Contents

Primary Keys & Foreign Keys

Player Data (Yahoo + NFL Merged)

Matchup Data

Transaction Data

Draft Data

Merge Strategies

Primary Keys & Foreign Keys

Understanding how to join data across different sources.

Primary Keys by Table
Table	Primary Key	Description
player	(league_id, yahoo_player_id, year, week)	Unique player-week combination (Yahoo)
player	(league_id, player, manager, year, week)	Alternative (name-based)
matchup	(league_id, manager, year, week)	Unique manager-week combination
transaction	(league_id, transaction_id) or (league_id, player, manager, year, week, type)	Transaction records
draft	(league_id, player, manager, year, round, pick)	Draft picks
Foreign Keys for Joining
Join Type	Left Table	Right Table	Join Keys
Player → Matchup	player	matchup	(league_id, manager, year, week)
Player → Player (opponent)	player	player	(league_id, opponent, year, week) as (league_id, manager, year, week)
Matchup → Matchup (opponent)	matchup	matchup	(league_id, opponent, year, week) as (league_id, manager, year, week)
Transaction → Player	transaction	player	(league_id, player, year, week)
Draft → Player	draft	player	(league_id, player, manager, year)
Player Data (Yahoo + NFL Merged)

Source Files:

yahoo_fantasy_data_v2.py - Yahoo player data

nfl_offense_stats_v2.py - NFL stats (includes DEF)

yahoo_nfl_merge_v2.py - Merged data

Identity Columns
Column	Type	Source	Description	Primary Key
league_id	string	Yahoo	League identifier (constant across years)	✓ (with all keys)
yahoo_player_id	string	Yahoo	Yahoo's unique player ID	✓ (with league_id, year, week)
NFL_player_id	string	NFL	NFL's unique player ID (nflverse)
player	string	Both	Player full name (cleaned, normalized)	✓ (with league_id, manager, year, week)
player_last_name	string	Both	Normalized last name (for matching)
player_year	string	Derived	Concatenated player + year (no spaces)
manager	string	Yahoo	Manager/team owner name	✓ (with league_id, year, week)
manager_year	string	Derived	Concatenated manager + year (no spaces)

**Note on `league_id`:** Added 2025-10-26 for multi-league isolation. Value is CONSTANT from `ctx.league_id` (e.g., "449.l.198278"), NOT the year-specific Yahoo league_key. Only present in Yahoo-sourced tables (player, matchup, draft, transaction). NFL data (offense/defense) does NOT have league_id.	
Time Columns
Column	Type	Source	Description	Primary Key
year	int	Both	NFL season year	✓
week	int	Both	NFL week number (1-18)	✓
season_type	string	NFL	Regular (REG) or Playoff (POST)	
Position Columns
Column	Type	Source	Description
yahoo_position	string	Yahoo	Yahoo's position designation
fantasy_position	string	Yahoo	Fantasy roster position (QB, RB, WR, TE, FLEX, etc.)
nfl_position	string	NFL	NFL official position
position_group	string	NFL	Position grouping (offense, defense, special teams)
Team Columns
Column	Type	Source	Description
team	string	Yahoo	Player's NFL team (Yahoo format)
nfl_team	string	NFL	Player's NFL team (3-letter code)
opponent	string	Yahoo	Fantasy opponent manager name
opponent_nfl_team	string	NFL	NFL opponent team
opponent_year	string	Derived	Concatenated opponent + year (no spaces)
Scoring Columns
Column	Type	Source	Description
points	float	Yahoo	Fantasy points earned (league-specific scoring)
fantasy_points_zero_ppr	float	NFL	Standard (0 PPR) fantasy points
fantasy_points_ppr	float	NFL	Full PPR fantasy points
fantasy_points_half_ppr	float	NFL	Half PPR fantasy points
Passing Stats
Column	Type	Source	Description
pass_yds	int	NFL	Passing yards
pass_td	int	NFL	Passing touchdowns
passing_interceptions	int	NFL	Interceptions thrown
completions	int	NFL	Pass completions
attempts	int	NFL	Pass attempts
passing_air_yards	float	NFL	Air yards (depth of target)
passing_yards_after_catch	float	NFL	YAC on completions
passing_first_downs	int	NFL	First downs via passing
passing_epa	float	NFL	Expected Points Added (passing)
passing_cpoe	float	NFL	Completion % Over Expected
passing_2pt_conversions	int	NFL	2-point conversion passes
pacr	float	NFL	Pass Air Conversion Ratio
Rushing Stats
Column	Type	Source	Description
rush_att	int	NFL	Rushing attempts
rush_yds	int	NFL	Rushing yards
rush_td	int	NFL	Rushing touchdowns
rushing_fumbles	int	NFL	Fumbles (rushing)
rushing_fumbles_lost	int	NFL	Fumbles lost (rushing)
rushing_first_downs	int	NFL	First downs via rushing
rushing_epa	float	NFL	Expected Points Added (rushing)
rushing_2pt_conversions	int	NFL	2-point conversion rushes
Receiving Stats
Column	Type	Source	Description
rec	int	NFL	Receptions
targets	int	NFL	Times targeted
rec_yds	int	NFL	Receiving yards
rec_td	int	NFL	Receiving touchdowns
receiving_fumbles	int	NFL	Fumbles (receiving)
receiving_fumbles_lost	int	NFL	Fumbles lost (receiving)
receiving_air_yards	float	NFL	Air yards (targets)
receiving_yards_after_catch	float	NFL	Yards after catch
receiving_first_downs	int	NFL	First downs via receiving
receiving_epa	float	NFL	Expected Points Added (receiving)
receiving_2pt_conversions	int	NFL	2-point conversion catches
racr	float	NFL	Receiver Air Conversion Ratio
target_share	float	NFL	% of team targets
air_yards_share	float	NFL	% of team air yards
wopr	float	NFL	Weighted Opportunity Rating
Aggregate Stats
Column	Type	Source	Description
2-pt	int	Derived	Total 2-pt conversions (pass + rush + rec)
fum_lost	int	Derived	Total fumbles lost (rush + rec)
Defensive/Special Teams Stats
Column	Type	Source	Description
ret_td	int	NFL	Return touchdowns (special teams)
def_tackles_solo	int	NFL	Solo tackles
def_tackles_with_assist	int	NFL	Tackles with assist
def_tackle_assists	int	NFL	Tackle assists
def_tackles_for_loss	int	NFL	Tackles for loss
def_tackles_for_loss_yards	float	NFL	TFL yards
def_fumbles_forced	int	NFL	Forced fumbles
def_sacks	int	NFL	Sacks
def_sack_yards	float	NFL	Sack yards
def_qb_hits	int	NFL	QB hits
def_interceptions	int	NFL	Interceptions (defensive)
def_interception_yards	int	NFL	INT return yards
def_pass_defended	int	NFL	Passes defended
def_tds	int	NFL	Defensive touchdowns
def_fumbles	int	NFL	Fumbles recovered
def_safeties	int	NFL	Safeties
Kicking Stats
Column	Type	Source	Description
fg_made	int	NFL	Field goals made
fg_att	int	NFL	Field goal attempts
fg_miss	int	NFL	Field goals missed
fg_blocked	int	NFL	Field goals blocked
fg_long	int	NFL	Longest field goal
fg_pct	float	NFL	FG percentage
fg_made_0_19	int	NFL	FG made 0-19 yards
fg_made_20_29	int	NFL	FG made 20-29 yards
fg_made_30_39	int	NFL	FG made 30-39 yards
fg_made_40_49	int	NFL	FG made 40-49 yards
fg_made_50_59	int	NFL	FG made 50-59 yards
fg_made_60_	int	NFL	FG made 60+ yards
fg_missed_0_19	int	NFL	FG missed 0-19 yards
fg_missed_20_29	int	NFL	FG missed 20-29 yards
fg_missed_30_39	int	NFL	FG missed 30-39 yards
fg_missed_40_49	int	NFL	FG missed 40-49 yards
fg_missed_50_59	int	NFL	FG missed 50-59 yards
fg_missed_60_	int	NFL	FG missed 60+ yards
fg_yds	int	Derived	Total FG yards (sum of made distances)
fg_made_list	string	NFL	Comma-separated list of made FG distances
fg_missed_list	string	NFL	Comma-separated list of missed FG distances
fg_blocked_list	string	NFL	Comma-separated list of blocked FG distances
fg_made_distance	string	NFL	Made FG distances (raw)
fg_missed_distance	string	NFL	Missed FG distances (raw)
fg_blocked_distance	string	NFL	Blocked FG distances (raw)
pat_made	int	NFL	Extra points made
pat_att	int	NFL	Extra point attempts
pat_missed	int	NFL	Extra points missed
pat_blocked	int	NFL	Extra points blocked
pat_pct	float	NFL	PAT percentage
gwfg_made	int	NFL	Game-winning FG made
gwfg_att	int	NFL	Game-winning FG attempts
gwfg_missed	int	NFL	Game-winning FG missed
gwfg_blocked	int	NFL	Game-winning FG blocked
gwfg_distance	int	NFL	Game-winning FG distance
Miscellaneous Stats
Column	Type	Source	Description
misc_yards	int	NFL	Miscellaneous yards
fumble_recovery_own	int	NFL	Own fumbles recovered
fumble_recovery_yards_own	int	NFL	Own fumble recovery yards
fumble_recovery_opp	int	NFL	Opponent fumbles recovered
fumble_recovery_yards_opp	int	NFL	Opponent fumble recovery yards
fum_rec	int	NFL	Total fumble recoveries
fum_ret_td	int	NFL	Fumble return touchdowns
penalties	int	NFL	Penalties committed
penalty_yards	int	NFL	Penalty yards
punt_returns	int	NFL	Punt returns
punt_return_yards	int	NFL	Punt return yards
kickoff_returns	int	NFL	Kickoff returns
kickoff_return_yards	int	NFL	Kickoff return yards
Metadata Columns
Column	Type	Source	Description
headshot_url	string	NFL	Player headshot image URL
url	string	Yahoo	Yahoo player page URL
Placeholder Columns (for future processing)
Column	Type	Source	Description
kept_next_year	bool	Future	Was player kept for next season
is_keeper_status	bool	Future	Is player keeper-eligible
keeper_price	int	Future	Keeper cost (auction/draft)
avg_points_this_year	float	Future	Season average points
avg_points_next_year	float	Future	Next season projection
avg_cost_next_year	float	Future	Next season projected cost
cost	float	Future	Acquisition cost
faab_bid	float	Future	FAAB bid amount
total_points_next_year	float	Future	Next season projected total
rolling_point_total	float	Future	Cumulative season points
manager_player_all_time_history	float	Future	Manager-player historical avg
manager_position_all_time_history	float	Future	Manager-position historical avg
player_personal_all_time_history	float	Future	Player historical avg (all managers)
position_all_time_history	float	Future	Position historical avg (all managers)
manager_player_season_history	float	Future	Manager-player season avg
manager_position_season_history	float	Future	Manager-position season avg
player_personal_season_history	float	Future	Player season avg
position_season_history	float	Future	Position season avg
*_percentile	float	Future	Percentile ranks for above metrics
Player Stats Enrichment (RECALCULATE WEEKLY)

Source File:

multi_league/transformations/player_stats_v2.py - Player statistics enrichment pipeline

Modules:

modules/scoring_calculator.py - Fantasy points calculation from scoring rules

modules/optimal_lineup.py - League-wide optimal player determination

modules/player_rankings.py - Player ranking systems with tiebreakers

modules/ppg_calculator.py - PPG metrics and rolling averages

Output: Enriched player data with statistics, rankings, and performance metrics

Column	Type	Description
fantasy_points	float	Fantasy points calculated from league-specific scoring rules
is_optimal	bool	True if player is league-wide optimal for their position this week
optimal_points	float	Maximum possible points for manager's roster this week
lineup_efficiency	float	Percentage of optimal points achieved (actual/optimal * 100)
bench_points	float	Total points from rostered but not started players
player_personal_week_rank	int	Rank within this player's all weekly performances (1 = best)
player_personal_week_pct	float	Percentile within player's weekly performances (0-100, higher = better)
player_personal_season_rank	int	Rank within this player's all seasons (1 = best season)
player_personal_season_pct	float	Percentile within player's seasons (0-100, higher = better)
position_week_rank	int	Rank among all players at this position this week (1 = best)
position_week_pct	float	Percentile among position this week (0-100, higher = better)
position_season_rank	int	Rank among all players at this position this season (1 = best)
position_season_pct	float	Percentile among position this season (0-100, higher = better)
position_alltime_rank	int	Rank among all players at this position all-time (1 = best)
position_alltime_pct	float	Percentile among position all-time (0-100, higher = better)
manager_player_week_rank	int	Rank among this manager's players this week (1 = best)
manager_player_week_pct	float	Percentile among manager's players this week (0-100, higher = better)
manager_player_season_rank	int	Rank among manager's players this season (1 = best)
manager_player_season_pct	float	Percentile among manager's players this season (0-100, higher = better)
manager_player_alltime_rank	int	Rank among manager's all-time players (1 = best)
manager_player_alltime_pct	float	Percentile among manager's all-time players (0-100, higher = better)
season_ppg	float	Points per game this season
season_games	int	Number of games played this season
alltime_ppg	float	Points per game across all seasons
alltime_games	int	Number of games played across all seasons
rolling_3_avg	float	Rolling 3-game average points
rolling_5_avg	float	Rolling 5-game average points
weighted_ppg	float	Exponentially weighted PPG (recent games weighted higher)
ppg_trend	float	Difference between season_ppg and alltime_ppg (positive = improving)
consistency_score	float	Coefficient of variation (std/mean * 100, lower = more consistent)

Ranking Logic:

All ranks use ordinal method (1, 2, 3, 4...) with ties broken by deterministic ordering

Lower rank number = better performance (rank 1 = best)

Higher percentile = better performance (100 = top performer)

Personal ranks compare player to their own history

Position ranks compare within position across league

Manager ranks compare within manager's roster history

PPG Metrics:

season_ppg: Average points per game in current season only

alltime_ppg: Average points per game across entire career

rolling_3_avg: Simple moving average of last 3 games

rolling_5_avg: Simple moving average of last 5 games

weighted_ppg: Exponentially weighted moving average (decay_factor=0.9)

ppg_trend: Difference showing if player is performing above/below career average

consistency_score: Lower values indicate consistent performers, higher values indicate boom/bust

Optimal Lineup Logic:

is_optimal: Player is in top N at their position based on roster settings (e.g., top 2 RBs)

optimal_points: Sum of optimal players' points for manager's roster

lineup_efficiency: What percentage of optimal points did manager actually score

bench_points: Points left on bench (rostered but not started)

Matchup Context Columns (From Matchup Import)

Source: multi_league/transformations/matchup_to_player_v2.py

These columns add game outcome context to each player performance.

Column	Type	Description
win	int	1 if manager won this week, 0 otherwise
loss	int	1 if manager lost this week, 0 otherwise
team_points	float	Manager's total points this week
opponent_points	float	Opponent's total points this week
margin	float	Point differential (team - opponent)
is_playoffs	int	1 if playoff game, 0 otherwise
is_consolation	int	1 if consolation game, 0 otherwise
cumulative_week	int	Cross-season week number derived from the season year and week (zero-padded week + year). Always stored as Int64 and used to construct manager_week and player_week keys.
team_made_playoffs	int	1 if manager made playoffs this season
quarterfinal	int	1 if manager reached quarterfinals (auto-init to 0, set manually after playoffs)
semifinal	int	1 if manager reached semifinals (auto-init to 0, set manually after playoffs)
champion	int	1 if manager won championship (auto-init to 0, set manually after championship)
sacko	int	1 if manager finished last place (auto-init to 0, set manually after season)
weekly_rank	int	Manager's rank within league this week (1 = highest score)
teams_beat_this_week	int	Number of league teams manager would have beaten this week
above_league_median	int	1 if scored above league median this week

Use Cases:

Identify clutch players (perform better in playoffs)

Find players who excel in winning weeks

Contextualize player performance by game importance

Analyze player performance in close games vs blowouts

Draft Context Columns (From Draft Import)

Source: multi_league/transformations/draft_to_player_v2.py

These columns add draft information to player data for ROI analysis.

Column	Type	Description
round	int	Draft round number (1-N)
pick	int	Pick number within round
overall_pick	int	Overall pick number in draft (1-N)
cost	float	Auction cost or draft value
is_keeper_status	int	1 if player was a keeper, 0 otherwise
draft_type	string	Type of draft (auction, snake, keeper)

Join Key: (yahoo_player_id, year) - Season-level join

Use Cases:

Calculate draft ROI (season points / cost)

Analyze value by draft position

Compare keeper vs freshly drafted players

Identify late-round steals and early-round busts

Track positional draft trends over time

Transaction Performance Columns (From Transaction Import)

Source: multi_league/transformations/player_to_transactions_v2.py

These columns add player performance context to transaction data, focusing on rest of season performance after the transaction.

Total Columns: 17 (includes position rank before, at, and after transaction)

Column	Type	Description
position	string	Player position at transaction time
nfl_team	string	Player's NFL team at transaction time
points_at_transaction	float	Fantasy points in week of transaction
ppg_before_transaction	float	Average PPG in 4 weeks before transaction
weeks_before	int	Number of weeks in before window
ppg_after_transaction	float	Average PPG in 4 weeks after transaction
total_points_after_4wks	float	Total points in 4 weeks after transaction
weeks_after	int	Number of weeks in after window
total_points_rest_of_season	float	Total points from transaction week to season end
ppg_rest_of_season	float	Average PPG for rest of season after transaction
weeks_rest_of_season	int	Number of weeks remaining after transaction
position_rank_at_transaction	int	Position rank in week of transaction (weekly rank)
position_rank_before_transaction	int	Position rank based on total points before transaction
position_rank_after_transaction	int	Position rank based on rest of season points
position_total_players	int	Total players at position that week
points_per_faab_dollar	float	Total rest of season points / FAAB bid
transaction_quality_score	int	Heuristic score (1-5 for adds, -1 to 3 for drops)

Join Key: (yahoo_player_id, year, cumulative_week) - Week-level exact match

Use Cases:

Identify waiver wire gems (high ppg_rest_of_season for low FAAB)

Measure transaction timing (did you add at the right time?)

Calculate points_per_faab_dollar for budget efficiency

Find breakout players (high ppg_after vs ppg_before)

Identify bad drops (players who performed well after being dropped)

Key Insight: Focus on future performance (after transaction) rather than sunk costs (before transaction).

Keeper Economics Columns (From Keeper Economics)

Source: multi_league/transformations/keeper_economics_v2.py

These columns calculate keeper prices and next-year value for keeper league analysis.

Column	Type	Description
cost	float	Original acquisition cost (draft or trade)
is_keeper_status	int	1 if player was kept this year, 0 if freshly drafted
max_faab_bid	float	Maximum FAAB bid spent on player this season
keeper_price	int	Calculated keeper price for next year
kept_next_year	int	1 if player was actually kept next year, 0 otherwise
total_points_next_year	float	Total fantasy points in following season

Join Key: (yahoo_player_id, year) - Season-level join

Keeper Price Formula:

# Base calculation
if is_keeper:
    base_price = cost * 1.5 + 7.5
else:
    base_price = cost

# Consider FAAB
half_faab = max_faab_bid / 2.0

# Final price (minimum 1)
keeper_price = max(base_price, half_faab, 1)


Use Cases:

Calculate keeper ROI (total_points_next_year / keeper_price)

Identify best keeper values (high points, low price)

Track keeper inflation over time

Analyze FAAB impact on keeper pricing

Compare freshly drafted vs kept player performance

Matchup Data

Source Files:

multi_league/data_fetchers/weekly_matchup_data_v2.py - Raw matchup data

multi_league/transformations/cumulative_stats_v2.py - Enriched with analytics

Output Files:

matchup_data/matchup.parquet - Matchup data with cumulative stats

matchup_data/matchup.csv - CSV version

Identity Columns
Column	Type	Description	Primary Key
manager	string	Manager name	✓ (with year, week)
team_name	string	Team name	
opponent	string	Opponent manager name	Foreign Key → manager
year	int	Season year	✓
week	int	Week number	✓
matchup_key	string	Canonical matchup identifier (alphabetically sorted teams)	For self-joins
matchup_id	string	Unique perspective identifier (matchup_key + manager)	

matchup_key Format:

# Format: "{team1}__vs__{team2}__{year}__{week}"
# Teams are sorted alphabetically to ensure same key for both perspectives

def create_matchup_key(manager: str, opponent: str, year: int, week: int) -> str:
    teams = sorted([manager, opponent])
    return f"{teams[0]}__vs__{teams[1]}__{year}__{week}"

# Example: "Alice" vs "Bob" in 2024 Week 5 → "Alice__vs__Bob__2024__5"
# Both perspectives (Alice's row and Bob's row) have same matchup_key


Usage for Self-Joins:

# Get both sides of matchup in one row
df_combined = df.merge(
    df[['matchup_key', 'manager', 'team_points', 'win']],
    on='matchup_key',
    suffixes=('', '_opp')
)
# Now each row has manager + opponent data together!
# df_combined has: manager, team_points, manager_opp, team_points_opp, etc.

Scoring Columns
Column	Type	Description
team_points	float	Manager's points this week
team_projected_points	float	Manager's projected points
opponent_points	float	Opponent's points this week
opponent_projected_points	float	Opponent's projected points
margin	float	Point differential (team - opponent)
total_matchup_score	float	Combined points (team + opponent)
Win/Loss Columns
Column	Type	Description
win	int	1 if won this week, 0 otherwise
loss	int	1 if lost this week, 0 otherwise
close_margin	int	1 if margin <= 10 points, 0 otherwise
Projection Metrics
Column	Type	Description
proj_wins	int	1 if projected to win, 0 otherwise
proj_losses	int	1 if projected to lose, 0 otherwise
proj_score_error	float	Actual - Projected points
abs_proj_score_error	float	Absolute projection error
above_proj_score	int	1 if outperformed projection
below_proj_score	int	1 if underperformed projection
expected_spread	float	Projected point differential
expected_odds	float	Win probability (0-1) based on projection
win_vs_spread	int	1 if beat the spread
lose_vs_spread	int	1 if lost vs the spread
underdog_wins	int	1 if won as underdog (negative spread)
favorite_losses	int	1 if lost as favorite (positive spread)
League Comparison Metrics
Column	Type	Description
weekly_mean	float	Manager's season average points
weekly_median	float	Manager's season median points
league_weekly_mean	float	League average points this week
league_weekly_median	float	League median points this week
above_league_median	int	1 if scored above league median
below_league_median	int	1 if scored below league median
teams_beat_this_week	int	Number of teams you would have beaten
opponent_teams_beat_this_week	int	Number of teams opponent would have beaten
Cumulative Records (RECALCULATE WEEKLY)

Source: multi_league/transformations/modules/cumulative_records.py

These columns track running totals and streaks across a manager's career.

Column	Type	Description	Update Frequency
cumulative_wins	Int64	All-time total wins (includes playoffs/consolation)	RECALCULATE WEEKLY
cumulative_losses	Int64	All-time total losses (includes playoffs/consolation)	RECALCULATE WEEKLY
wins_to_date	Int64	Season wins to date (excludes consolation)	RECALCULATE WEEKLY
losses_to_date	Int64	Season losses to date (excludes consolation)	RECALCULATE WEEKLY
points_scored_to_date	float	Season points to date (excludes consolation)	RECALCULATE WEEKLY
win_streak	Int64	Current active win streak	RECALCULATE WEEKLY
loss_streak	Int64	Current active loss streak	RECALCULATE WEEKLY

Note: Streaks reset on ties (when neither win nor loss occurs).

Weekly Metrics (RECALCULATE WEEKLY)

Source: multi_league/transformations/modules/weekly_metrics.py

These columns measure league-relative performance for each week.

Column	Type	Description	Update Frequency
teams_beat_this_week	Int64	Number of league teams you would have beaten this week	RECALCULATE WEEKLY
above_league_median	Int64	1 if scored above league median this week, 0 otherwise	RECALCULATE WEEKLY
below_league_median	Int64	1 if scored below league median this week, 0 otherwise	RECALCULATE WEEKLY
weekly_rank	Int64	Rank within league this week (1 = highest score)	RECALCULATE WEEKLY
league_weekly_mean	float	League average points this week	RECALCULATE WEEKLY
league_weekly_median	float	League median points this week	RECALCULATE WEEKLY
Head-to-Head Records (RECALCULATE WEEKLY)

Source: multi_league/transformations/modules/head_to_head.py

Format: w_vs_{manager_token} and l_vs_{manager_token}

Column Pattern	Type	Description	Update Frequency
w_vs_{manager}	Int64	Cumulative wins vs specific manager	RECALCULATE WEEKLY
l_vs_{manager}	Int64	Cumulative losses vs specific manager	RECALCULATE WEEKLY

Example: w_vs_joe, l_vs_joe, w_vs_john_smith

Note: Manager token = manager.strip().lower() with whitespace/non-alphanumeric replaced by _

Token Generation:

def _mgr_token(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"\s+", "_", s)  # Replace whitespace
    s = re.sub(r"[^a-z0-9_]+", "", s)  # Remove non-alphanumeric
    s = re.sub(r"_+", "_", s).strip("_")  # Collapse multiple underscores
    return s or "na"

Season Rankings (MIXED)

Source: multi_league/transformations/modules/season_rankings.py

Contains both SET-AND-FORGET columns (calculated once after championship) and RECALCULATE WEEKLY columns.

SET-AND-FORGET Columns (championship_complete=True only):

Column	Type	Description	Update Frequency
final_wins	Int64	Total wins for the season (including playoffs)	SET-AND-FORGET
final_losses	Int64	Total losses for the season (including playoffs)	SET-AND-FORGET
final_regular_wins	Int64	Regular season wins only (excludes playoffs/consolation)	SET-AND-FORGET
final_regular_losses	Int64	Regular season losses only (excludes playoffs/consolation)	SET-AND-FORGET
season_mean	float	Average points per game (season-level)	SET-AND-FORGET
season_median	float	Median points per game (season-level)	SET-AND-FORGET
manager_season_ranking	Int64	Final season rank (by wins, then total points)	SET-AND-FORGET

RECALCULATE WEEKLY Columns:

Column	Type	Description	Update Frequency
manager_all_time_ranking	Int64	All-time rank across all managers (by total career wins)	RECALCULATE WEEKLY
manager_all_time_ranking_percentile	float	Percentile rank (0-100)	RECALCULATE WEEKLY
league_all_time_ranking	Int64	League-wide all-time rank	RECALCULATE WEEKLY
league_all_time_ranking_percentile	float	League percentile (0-100)	RECALCULATE WEEKLY
Playoff Odds (RECALCULATE WEEKLY)

Source: multi_league/transformations/playoff_odds_v2.py

Playoff probabilities calculated using Monte Carlo simulation (10,000 simulations per week). All columns are recalculated weekly.

Playoff Probabilities:

Column	Type	Description	Update Frequency
p_playoffs	float	Probability of making playoffs (0-100%)	RECALCULATE WEEKLY
p_bye	float	Probability of getting first-round bye (0-100%)	RECALCULATE WEEKLY
p_semis	float	Probability of reaching semifinals (0-100%)	RECALCULATE WEEKLY
p_final	float	Probability of reaching championship game (0-100%)	RECALCULATE WEEKLY
p_champ	float	Probability of winning championship (0-100%)	RECALCULATE WEEKLY

Expected Outcomes:

Column	Type	Description	Update Frequency
exp_final_wins	float	Expected wins at end of season	RECALCULATE WEEKLY
exp_final_pf	float	Expected points at end of season	RECALCULATE WEEKLY
avg_seed	float	Expected playoff seed (1-10)	RECALCULATE WEEKLY
power_rating	float	Team strength rating (normalized by inflation_rate)	RECALCULATE WEEKLY

Seed Distributions:

Column Pattern	Type	Description	Update Frequency
x1_seed through x10_seed	float	Probability of finishing as each seed (0-100%)	RECALCULATE WEEKLY

Example: x1_seed = 25.5 means 25.5% chance of finishing as 1st seed

Win Distributions:

Column Pattern	Type	Description	Update Frequency
x0_win through x14_win	float	Probability of finishing with N wins (0-100%)	RECALCULATE WEEKLY

Example: x10_win = 18.3 means 18.3% chance of finishing with exactly 10 wins

Simulation Methodology:

Uses empirical Bayes shrinkage to estimate team strength (mu/sigma)

Bootstrap sampling from recent games when available

Recency weighting with 10-week half-life

Blends simulation-based seed prediction with historical kernel estimation

Respects actual playoff results and only simulates remaining games

Expected Records (RECALCULATE WEEKLY)

Source: multi_league/transformations/expected_record_v2.py

Schedule-independent expected records calculated using 100,000 Monte Carlo simulations. Answers the question: "What would your record be if you played random schedules?"

Performance-Based Metrics (shuffle_*):

Simulates random schedules using actual team_points to measure performance independent of opponent quality.

Column Pattern	Type	Description	Update Frequency
shuffle_{w}_win (w=0-14)	float	Probability of finishing with exactly W wins (0-100%)	RECALCULATE WEEKLY
shuffle_{s}_seed (s=1-10)	float	Probability of finishing as seed S (0-100%)	RECALCULATE WEEKLY
shuffle_avg_wins	float	Expected wins with random schedules	RECALCULATE WEEKLY
shuffle_avg_seed	float	Expected playoff seed with random schedules	RECALCULATE WEEKLY
shuffle_avg_playoffs	float	Probability of making playoffs (0-100%)	RECALCULATE WEEKLY
shuffle_avg_bye	float	Probability of getting first-round bye (0-100%)	RECALCULATE WEEKLY
wins_vs_shuffle_wins	float	Actual wins - expected wins (measures schedule luck)	RECALCULATE WEEKLY
seed_vs_shuffle_seed	float	Actual seed - expected seed (measures seeding luck)	RECALCULATE WEEKLY

Opponent Difficulty Metrics (opp_shuffle_*):

Simulates random schedules using opponent_points to measure schedule strength/luck.

Column Pattern	Type	Description	Update Frequency
opp_shuffle_{w}_win (w=0-14)	float	Probability of W "easy weeks" (lower opponent_points)	RECALCULATE WEEKLY
opp_shuffle_{s}_seed (s=1-10)	float	Probability of seed S based on opponent difficulty	RECALCULATE WEEKLY
opp_shuffle_avg_wins	float	Expected "easy weeks" (higher = easier schedule)	RECALCULATE WEEKLY
opp_shuffle_avg_seed	float	Expected seed based on schedule difficulty	RECALCULATE WEEKLY
opp_shuffle_avg_playoffs	float	Playoff probability based on schedule strength	RECALCULATE WEEKLY
opp_shuffle_avg_bye	float	Bye probability based on schedule strength	RECALCULATE WEEKLY
opp_pts_week_rank	Int64	Weekly opponent difficulty rank (1 = hardest)	RECALCULATE WEEKLY
opp_pts_week_pct	float	Opponent difficulty percentile (100 = hardest)	RECALCULATE WEEKLY

Interpretation:

wins_vs_shuffle_wins > 0: Lucky schedule (won more than expected based on performance)

wins_vs_shuffle_wins < 0: Unlucky schedule (won less than expected)

opp_shuffle_avg_wins: Higher values = easier opponents faced

opp_pts_week_pct = 90: Faced opponent in top 10% difficulty that week

Simulation Details:

100,000 simulations per week

Round-robin schedule generation with randomization

No repeat matchups in first 5 weeks

Max 2 meetings between any pair

Coin flip for ties

Player Aggregation Columns (From Player Import)

Source: multi_league/transformations/player_to_matchup_v2.py

These columns aggregate player-level stats to provide manager lineup context.

Column	Type	Description
optimal_points	float	Maximum possible points from optimal lineup this week
bench_points	float	Total points left on bench (rostered but not started)
lineup_efficiency	float	Percentage of optimal achieved (team_points / optimal * 100)
optimal_ppg_season	float	Average optimal points per game this season
rolling_optimal_points	float	Cumulative optimal points (season to date)
total_optimal_points	float	Total optimal points for full season
optimal_points_all_time	float	Cumulative optimal points across all seasons
optimal_win	int	1 if would have won with optimal lineup vs opponent optimal
optimal_loss	int	1 if would have lost with optimal lineup vs opponent optimal
opponent_optimal_points	float	Opponent's optimal points this week
total_player_points	float	Sum of all rostered player points (verification column)
players_rostered	int	Count of rostered players this week
players_started	int	Count of started players this week

Use Cases:

Measure lineup management skill (lineup_efficiency)

Identify managers who consistently leave points on bench

Compare actual record vs optimal record (coaching effect)

Track roster construction quality over time

Identify weeks where lineup decisions cost/won games

Calculations:

optimal_points: Aggregated from player-level optimal lineup determination

lineup_efficiency: (team_points / optimal_points) * 100

optimal_win: 1 if (own optimal > opponent optimal), 0 otherwise

optimal_ppg_season: Total optimal points / weeks played

rolling_optimal_points: Cumulative sum of optimal_points within season

Playoff/Meta Columns
Column	Type	Description
is_playoffs	int	1 if playoff week, 0 otherwise
is_consolation	int	1 if consolation bracket, 0 otherwise
week_start	string	Week start date (Yahoo format)
week_end	string	Week end date (Yahoo format)
Matchup Grade Columns
Column	Type	Description
grade	string	Matchup grade (A+, A, B+, etc.)
gpa	float	GPA equivalent of grade (4.0 scale)
matchup_recap_title	string	Yahoo recap title
matchup_recap_url	string	Yahoo recap URL
Team Metadata Columns
Column	Type	Description
url	string	Team page URL
image_url	string	Team logo URL
division_id	string	Division ID (if applicable)
waiver_priority	float	Waiver wire priority (1 = first)
faab_balance	float	Remaining FAAB budget
number_of_moves	float	Total transactions (cumulative)
number_of_trades	float	Total trades (cumulative)
auction_budget_spent	float	Auction dollars spent (draft)
auction_budget_total	float	Total auction budget (draft)
has_draft_grade	string	Whether team has draft grade
coverage_value	float	Coverage value (Yahoo metric)
value	float	Team value (Yahoo metric)
Felo Rating Columns
Column	Type	Description
felo_score	float	Felo rating (Elo for fantasy)
felo_tier	string	Felo tier (GOLD, SILVER, BRONZE, etc.)
win_probability	float	Win probability for this matchup
Transaction Data

Source File: multi_league/data_fetchers/transactions_v2.py

Output Files:

transaction_data/transactions.parquet - All transaction data

transaction_data/transactions.csv - All transaction data (CSV)

Transaction data captures all roster moves including adds, drops, trades, and waivers. Designed for seamless joins with player and matchup data.

Primary Keys

Transaction Primary Key: (transaction_id) or (yahoo_player_id, manager, year, week, transaction_type)

Foreign Keys:

yahoo_player_id → links to player.yahoo_player_id

(manager, year, week) → links to matchup data

manager_year, manager_week, player_year, player_week → composite keys for quick joins

Identity Columns
Column	Type	Description	Primary Key	Example
transaction_id	string	Yahoo transaction unique ID	✓	nfl.l.123456.tr.1234
yahoo_player_id	string	Yahoo player ID (for joins)	✓ (with manager, year, week, type)	33376
player_name	string	Player full name (original Yahoo)		Patrick Mahomes
player_key	string	Full Yahoo player key		461.p.33376
manager	string	Manager who made transaction	Foreign Key	John Smith

Note on player_name: Preserved as-is from Yahoo (no cleaning). Use yahoo_player_id for joins to player data.

Time Columns
Column	Type	Description	Example
year	int	Season year	2024
week	int	NFL week number (1-18)	5
cumulative_week	int	Cross-season week number derived from year and week (e.g., year 2024 week 5 → 52024). Always stored as Int64 and used in manager_week and player_week join keys.	145
week_start	datetime	Week start date	2024-10-05
week_end	datetime	Week end date	2024-10-09
timestamp	string	Yahoo timestamp (seconds since epoch)	1728432000
human_readable_timestamp	string	Human-readable timestamp	OCT 08 2024 03:00:00 PM
Transaction Details
Column	Type	Description	Example Values
transaction_type	string	Type of player movement	add, drop, trade
source_type	string	Where player came from	freeagents, waivers, team
destination	string	Where player went	team, waivers, dropped
status	string	Transaction status	successful, failed, pending
faab_bid	int	FAAB bid amount (0 if free agent)	15, 0
Composite Keys (for joins)
Column	Type	Description	Example
manager_week	string	Manager + cumulative_week (no spaces)	JohnSmith145
manager_year	string	Manager + year (no spaces)	JohnSmith2024
player_week	string	Player name + cumulative_week (no spaces)	PatrickMahomes145
player_year	string	Player name + year (no spaces)	PatrickMahomes2024
Common Queries
# Join transactions to player stats
df = transactions.merge(
    player_stats,
    on=['yahoo_player_id', 'year', 'week'],
    how='left'
)

# Find all FAAB bids for a manager
faab_bids = transactions[
    (transactions['manager'] == 'John Smith') &
    (transactions['faab_bid'] > 0)
]

# Weekly transaction counts by manager
weekly_activity = transactions.groupby(['manager', 'year', 'week']).size()

Draft Data

Source File: multi_league/data_fetchers/draft_data_v2.py

Output Files:

draft_data/draft_data_{year}.csv - Single year draft data

draft_data/draft_data_{year}.parquet - Single year (Parquet)

draft_data/draft_data_all_years.csv - Combined all years

draft_data/draft_data_all_years.parquet - Combined (Parquet)

Draft data captures every pick in the fantasy draft, including auction costs, keeper status, and draft analysis metrics from Yahoo.

Primary Keys

Draft Primary Key: (yahoo_player_id, manager, year)

Alternative: (pick, year) for overall pick number

Foreign Keys:

yahoo_player_id → links to player.yahoo_player_id

(manager, year) → links to manager's seasonal data

manager_year → composite key for quick joins

Identity Columns
Column	Type	Description	Source	Example
year	int	Draft year	Yahoo	2024
pick	int	Overall pick number (1-N)	Yahoo	15
round	int	Draft round number	Yahoo	2
team_key	string	Yahoo team key	Yahoo	nfl.l.123456.t.1
manager	string	Manager who drafted (with overrides applied)	Yahoo + Context	John Smith
yahoo_player_id	string	Yahoo player ID (unique identifier)	Yahoo	12345

Note on manager: This field applies manager_name_overrides from LeagueContext to normalize nicknames to standardized names.

Draft Pick Details
Column	Type	Description	Source	Example
player	string	Player full name	Yahoo	Patrick Mahomes
yahoo_position	string	Position as Yahoo	Yahoo	QB, RB, WR/RB
nfl_team	string	NFL team abbreviation	Yahoo	KC, BUF
cost	float	Auction cost (null if snake draft)	Yahoo	45.0, null
Draft Analysis Metrics

These metrics come from Yahoo's draft analysis API and provide league-wide draft trends:

Column	Type	Description	Source	Example
avg_pick	float	Average pick in Yahoo leagues	Yahoo Analysis	15.3
avg_round	float	Average round in Yahoo leagues	Yahoo Analysis	2.1
avg_cost	float	Average auction cost in Yahoo	Yahoo Analysis	42.5
percent_drafted	float	% of Yahoo leagues drafted in	Yahoo Analysis	98.5
preseason_avg_pick	float	Preseason average pick	Yahoo Analysis	12.0
preseason_avg_round	float	Preseason average round	Yahoo Analysis	1.8
preseason_avg_cost	float	Preseason average cost	Yahoo Analysis	48.0
preseason_percent_drafted	float	Preseason % drafted	Yahoo Analysis	99.2

Note: Preseason metrics reflect expectations before the season started. Post-draft metrics reflect actual draft behavior.

Keeper Information
Column	Type	Description	Source	Example
is_keeper_status	string	Keeper status	Yahoo	"K" if keeper, "" otherwise
is_keeper_cost	string	Cost to keep player	Yahoo	"35", ""
savings	float	Savings vs avg cost (calculated)	Calculated	7.5 (avg_cost - keeper_cost)

Keeper Logic:

savings = avg_cost - is_keeper_cost (if both present)

Positive savings = good keeper value (drafted below average)

Negative savings = bad keeper value (overpaid vs average)

Derived Columns
Column	Type	Description	Calculation	Example
player_year	string	Player + year (no spaces)	player.replace(" ", "") + year	PatrickMahomes2024
manager_year	string	Manager + year (no spaces)	manager.replace(" ", "") + year	JohnSmith2024
cost_bucket	int	Cost tier within position/year	Grouped by position/year, buckets of 3	1, 2, 3

Cost Bucket Logic:

Groups players by (yahoo_position, year)

Sorts by cost ascending

Assigns buckets: picks 1-3 = bucket 1, picks 4-6 = bucket 2, etc.

Useful for analyzing draft strategy by position tier

Example Draft Data
year,pick,round,team_key,manager,yahoo_player_id,cost,player,yahoo_position,avg_pick,avg_round,avg_cost,percent_drafted,is_keeper_status,is_keeper_cost,savings,player_year,manager_year,nfl_team,cost_bucket
2024,1,1,nfl.l.123456.t.1,John Smith,12345,55.0,Patrick Mahomes,QB,8.2,1.1,52.3,99.8,,,2.7,PatrickMahomes2024,JohnSmith2024,KC,1
2024,2,1,nfl.l.123456.t.2,Jane Doe,23456,54.0,Christian McCaffrey,RB,2.1,1.0,58.9,100.0,,,4.9,ChristianMcCaffrey2024,JaneDoe2024,SF,1
2024,15,2,nfl.l.123456.t.1,John Smith,34567,48.0,Travis Kelce,TE,15.3,2.1,45.2,,,,TravisKelce2024,JohnSmith2024,KC,1
2024,16,2,nfl.l.123456.t.2,Jane Doe,45678,35.0,Tyreek Hill,WR,12.8,1.8,47.3,,,12.3,TyreekHill2024,JaneDoe2024,MIA,2
2024,50,5,nfl.l.123456.t.1,John Smith,56789,,Defensive Player,DEF,98.5,10.2,,65.2,,,DefensivePlayer2024,JohnSmith2024,SF,

Data Quality Notes

Null Values:

cost will be null for snake drafts (all non-auction leagues)

is_keeper_status and is_keeper_cost are empty strings for non-keeper picks

savings is null when keeper cost or avg cost unavailable

Cost Bucket Assignment:

Only assigned when cost > 0 and yahoo_position is not empty

Calculated AFTER merging draft picks with analysis data

Missing for players without valid cost data

Manager Name Normalization:

Raw Yahoo nicknames are mapped via manager_name_overrides from LeagueContext

Example: "--hidden--" → "Ilan" (from KMFFL league)

Position Handling:

Yahoo uses flexible positions like "WR/RB" for flex-eligible players

yahoo_position preserves original Yahoo format

Use nfl_position from player data for strict position classification

Merge Examples
Add Draft Data to Player Data
# Join draft info to weekly player stats
merged = player_df.merge(
    draft_df[['yahoo_player_id', 'year', 'pick', 'round', 'cost', 'is_keeper_status']],
    on=['yahoo_player_id', 'year'],
    how='left'
)
# Result: Player rows now have draft pick, round, cost, keeper status

Add Season Stats to Draft Data
# Aggregate player stats by season
season_stats = player_df.groupby(['yahoo_player_id', 'year']).agg({
    'points': 'sum',
    'pass_yds': 'sum',
    'rush_td': 'sum',
    'rec': 'sum'
}).reset_index()

# Join to draft data
merged = draft_df.merge(
    season_stats,
    on=['yahoo_player_id', 'year'],
    how='left',
    suffixes=('', '_season')
)
# Result: Draft rows now have season totals for ROI analysis

Draft Value Analysis
# Calculate draft ROI (points per dollar)
draft_value = draft_df.merge(
    season_stats[['yahoo_player_id', 'year', 'points']],
    on=['yahoo_player_id', 'year'],
    how='left'
)

draft_value['points_per_dollar'] = draft_value['points'] / draft_value['cost']
draft_value['value_over_adp'] = draft_value['points'] - (draft_value['avg_pick'] * 10)  # Rough heuristic

# Find best values by cost bucket
best_values = (
    draft_value[draft_value['cost_bucket'].notna()]
    .groupby(['yahoo_position', 'cost_bucket'])
    .apply(lambda g: g.nlargest(5, 'points_per_dollar'))
)

Merge Strategies
Common Merge Patterns
1. Add Matchup Data to Player Data

Goal: Add wins/losses to player data

# Left: player data
# Right: matchup data
# Join on: (manager, year, week)

merged = player_df.merge(
    matchup_df[['manager', 'year', 'week', 'win', 'loss', 'team_points', 'opponent_points']],
    on=['manager', 'year', 'week'],
    how='left',
    suffixes=('', '_matchup')
)


Result: Each player row now has manager's W/L record for that week

2. Add Opponent Matchup Data to Player Data

Goal: Add opponent's stats to player data

# Left: player data
# Right: matchup data
# Join on: player.opponent = matchup.manager (same year, week)

merged = player_df.merge(
    matchup_df[['manager', 'year', 'week', 'team_points', 'teams_beat_this_week']],
    left_on=['opponent', 'year', 'week'],
    right_on=['manager', 'year', 'week'],
    how='left',
    suffixes=('', '_opp')
)


Result: Each player row now has opponent's stats

3. Add Player Stats to Matchup Data

Goal: Aggregate player points by manager-week

# Aggregate player data by manager-week
player_agg = player_df.groupby(['manager', 'year', 'week']).agg({
    'points': 'sum',
    'player': 'count'  # number of players
}).reset_index()

player_agg.columns = ['manager', 'year', 'week', 'total_player_points', 'roster_count']

# Merge to matchup data
merged = matchup_df.merge(
    player_agg,
    on=['manager', 'year', 'week'],
    how='left'
)


Result: Matchup data now has aggregated player stats

4. Add Head-to-Head Records to Player Data

Goal: Add W/L vs specific opponent

# Get opponent's manager token
import re
opponent_token = re.sub(r'\W+', '_', opponent_name.strip().lower())

# Merge specific W/L columns
merged = player_df.merge(
    matchup_df[['manager', 'year', 'week', f'w_vs_{opponent_token}', f'l_vs_{opponent_token}']],
    on=['manager', 'year', 'week'],
    how='left'
)


Result: Player data has W/L record vs specific opponent

5. Add Transaction Context to Player Data

Goal: Mark when player was acquired

# Left: player data
# Right: transaction data (type='add')
# Join on: (player, manager, year, week)

merged = player_df.merge(
    transaction_df[transaction_df['type'] == 'add'][['player', 'manager', 'year', 'week', 'faab_bid']],
    on=['player', 'manager', 'year', 'week'],
    how='left',
    suffixes=('', '_txn')
)


Result: Player data shows when/how player was acquired

Merge Key Naming Conventions

To avoid conflicts and enable clean merges:

Always use consistent key names:

manager (not team_owner, owner, etc.)

year (not season, season_year)

week (not week_num, week_number)

player (not player_name, full_name)

Use suffixes for duplicate columns:

_matchup - from matchup data

_opp - opponent's data

_txn - from transaction data

_draft - from draft data

_nfl - from NFL source

_yahoo - from Yahoo source

Use foreign key prefixes:

opponent + _column for opponent's data

from_manager / to_manager for trade data

Data Validation Before Merge
# Check for duplicates in keys
assert not df.duplicated(subset=['manager', 'year', 'week']).any(), "Duplicate keys found!"

# Check key columns exist
required_keys = ['manager', 'year', 'week']
assert all(k in df.columns for k in required_keys), f"Missing key columns"

# Check key dtypes match
assert player_df['year'].dtype == matchup_df['year'].dtype, "Year dtype mismatch"

# Check for nulls in keys
assert not df[required_keys].isnull().any().any(), "Null values in keys"

Column Naming Conventions
Pattern Standards

Statistical Columns:

Format: {stat_type}_{stat_name}

Examples: pass_yds, rush_td, rec_yds

Derived Columns:

Format: {description} or {stat}_derived

Examples: margin, teams_beat_this_week, 2-pt

Metadata Columns:

Format: {entity}_{attribute}

Examples: team_name, matchup_recap_url, auction_budget_total

Aggregated Columns:

Format: {stat}_{aggregation}

Examples: weekly_mean, league_weekly_median, total_points_next_year

Boolean/Flag Columns:

Format: is_{condition} or has_{attribute}

Examples: is_playoffs, is_keeper_status, has_draft_grade

Relationship Columns:

Format: {relation}_{column}

Examples: opponent_points, opponent_nfl_team

Reserved Prefixes

manager_ - Manager-specific data

team_ - Team-specific data

player_ - Player-specific data

opponent_ - Opponent's data

league_ - League-wide aggregates

def_ - Defensive stats

pass_, rush_, rec_ - Offensive stat categories

fg_, pat_ - Kicking stats

w_vs_, l_vs_ - Head-to-head records

Quick Reference: Join Keys
# Player ← Matchup (add W/L to player)
on=['manager', 'year', 'week']

# Player ← Player (add opponent player data)
left_on=['opponent', 'year', 'week']
right_on=['manager', 'year', 'week']

# Matchup ← Matchup (add opponent matchup data)
left_on=['opponent', 'year', 'week']
right_on=['manager', 'year', 'week']

# Player ← Transaction (add transaction context)
on=['player', 'manager', 'year', 'week']  # or just 'transaction_id'

# Player ← Draft (add draft info)
on=['player', 'manager', 'year']

# Player ← Player (season aggregates)
on=['player', 'year']  # drops week for season totals

Updates Log
Date	Update	Changed By
2025-10-20	Initial creation	Claude
2025-10-20	Added transformation columns (cumulative_records, weekly_metrics, head_to_head, season_rankings)	Claude
2025-10-20	Added matchup_key documentation and self-join examples	Claude
2025-10-20	Documented SET-AND-FORGET vs RECALCULATE WEEKLY distinction	Claude
2025-10-21	Added playoff_odds columns with full Monte Carlo simulation documentation	Claude
2025-10-21	Documented power_rating, seed distributions, and win distributions	Claude
2025-10-21	Added expected_record columns (shuffle_* and opp_shuffle_*) with 100K simulations	Claude
2025-10-21	Documented schedule-independent performance and opponent difficulty metrics	Claude
2025-10-21	Added player_stats_v2 enrichment columns (rankings, PPG, optimal lineup, consistency)	Claude
2025-10-21	Documented modular player stats transformation with 4 focused modules	Claude
2025-10-21	Added cross-import transformations (matchup_to_player_v2, player_to_matchup_v2)	Claude
2025-10-21	Documented 15 matchup context columns added to player data	Claude
2025-10-21	Documented 13 player aggregation columns added to matchup data	Claude
2025-10-21	Added draft_to_player_v2 transformation (6 draft context columns)	Claude
2025-10-21	Added player_to_transactions_v2 transformation (17 transaction performance columns)	Claude
2025-10-21	Added keeper_economics_v2 transformation (6 keeper economics columns)	Claude
2025-10-21	Documented keeper price formula and ROI analysis patterns	Claude
2025-10-21	Enhanced transaction columns with position rank before/after transaction (17 total)	Claude
2025-10-21	Added aggregate_player_season_v2.py for lean season/career datasets	Claude
2025-10-21	Created initial_import_v2.py orchestrator (complete historical import)	Claude
2025-10-21	Created weekly_import_v2.py orchestrator (automated weekly updates)	Claude
2025-10-21	Created preseason_import_v2.py orchestrator (post-draft setup)	Claude
2025-10-20	Updated header with refactoring changes	Claude
		

To update this document:

Add new columns when creating new data sources

Update merge strategies when new join patterns emerge

Document any column renames or deprecations

Add examples of successful merges
---

## Recent Column Additions (2025-10-27)

The following columns were added or enhanced to ensure pipeline compatibility and multi-league support:

### player.parquet - New/Enhanced Columns

| Column | Type | Source | Description | Added |
|--------|------|--------|-------------|-------|
| **position** | string | Derived | **Unified position column** (Required by player_stats_v2.py). Prefers `fantasy_position` → `yahoo_position` → `nfl_position`. Used for optimal lineup determination and position-based rankings. | 2025-10-27 |
| **league_id** | string | Yahoo OAuth | **Multi-league isolation key**. Format: `{game_id}.l.{league_id}` (e.g., `449.l.198278`). Added to ALL Yahoo-sourced player rows to enable multi-tenant SaaS architecture. | 2025-10-27 |

**Usage Notes**:
- `position`: Use this for all position-based queries and groupings. Do NOT use fantasy_position/yahoo_position/nfl_position directly unless you need specific source distinction.
- `league_id`: CRITICAL for multi-league queries. Always filter by league_id when querying player data in multi-tenant environment.

**Example**:
```sql
-- Get optimal RBs for a specific league
SELECT player, fantasy_points, manager
FROM player
WHERE league_id = '449.l.198278'
  AND position = 'RB'
  AND is_optimal = true
  AND year = 2024
  AND week = 5
ORDER BY fantasy_points DESC;
```

---

### matchup.parquet - New/Enhanced Columns

| Column | Type | Source | Description | Added |
|--------|------|--------|-------------|-------|
| **inflation_rate** | float | Calculated | **Year-over-year scoring normalization factor**. Calculated as `(year_avg_points / base_year_avg_points)`. Base year (earliest year in data) = 1.0. Used by playoff_odds_v2.py to normalize power ratings across different scoring eras. Example: If 2024 averages 120 pts/game and 2014 averaged 100 pts/game, 2024's inflation_rate = 1.2. | 2025-10-27 |

**Calculation**:
```python
year_means = df.groupby("year")["team_points"].mean()
base_year = year_means.index.min()
base_mean = year_means.loc[base_year]
inflation_rate[year] = year_means[year] / base_mean
```

**Usage Notes**:
- Use for cross-year comparisons of team strength
- Essential for playoff simulations that span multiple seasons
- Helps identify whether high scores are due to rule changes or superior performance

**Example**:
```sql
-- Compare real vs inflation-adjusted scores
SELECT year, 
       AVG(team_points) as avg_raw_score,
       AVG(team_points / inflation_rate) as avg_adjusted_score,
       inflation_rate
FROM matchup
WHERE is_playoffs = 0 AND is_consolation = 0
GROUP BY year, inflation_rate
ORDER BY year;
```

---

### Nearest-Year Fallback Logic

**Problem Solved**: Missing scoring rules or roster settings for specific years no longer cause failures.

**Implementation**:
- **scoring_calculator.py**: If exact year's scoring rules don't exist, uses closest available year
- **optimal_lineup.py**: If exact year's roster settings don't exist, uses closest available year

**Algorithm**:
```python
# Example: 2017 rules missing, but 2016 and 2018 exist
available_years = [2014, 2015, 2016, 2018, 2019]
target_year = 2017

closest_year = min(available_years, key=lambda y: abs(y - target_year))
# Returns: 2016 or 2018 (both distance=1, min returns first found)
```

**Log Messages**:
```
[scoring] Using 2016 rules for year 2017 (exact rules not found)
[roster] Using 2018 roster settings for year 2017 (exact settings not found)
```

---

## Pipeline Compatibility Matrix

| Script | Requires position | Requires league_id | Requires inflation_rate |
|--------|-------------------|--------------------|-----------------------|
| player_stats_v2.py | ✅ YES | ❌ No | ❌ No |
| playoff_odds_v2.py | ❌ No | ✅ YES | ✅ YES |
| player_to_matchup_v2.py | ❌ No | ✅ YES | ❌ No |
| player_to_transactions_v2.py | ❌ No | ✅ YES | ❌ No |
| cumulative_stats_v2.py | ❌ No | ✅ YES | ❌ Creates it |

**Multi-League Isolation Requirements**:
- ALL queries accessing player/matchup/draft/transactions data MUST filter by `league_id`
- MotherDuck tables MUST partition by `league_id`
- Streamlit UI MUST pass user's `league_id` to all queries

---

## Validation Queries

### Check league_id coverage:

```sql
-- Player data
SELECT 
    COUNT(*) as total_rows,
    SUM(CASE WHEN league_id IS NOT NULL THEN 1 ELSE 0 END) as with_league_id,
    ROUND(100.0 * SUM(CASE WHEN league_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as coverage_pct
FROM player;

-- Expected: 100% coverage for Yahoo players, NULL for NFL-only players
```

### Check position coverage:

```sql
-- Should show distribution across QB/RB/WR/TE/DEF
SELECT position, COUNT(*) as count
FROM player
WHERE league_id IS NOT NULL
GROUP BY position
ORDER BY count DESC;
```

### Check inflation_rate calculation:

```sql
-- Should show progression over years
SELECT year, 
       ROUND(AVG(inflation_rate), 3) as avg_inflation,
       ROUND(AVG(team_points), 2) as avg_score
FROM matchup
WHERE is_playoffs = 0
GROUP BY year
ORDER BY year;
```
