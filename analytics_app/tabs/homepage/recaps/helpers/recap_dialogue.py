"""
Recap Dialogue Configuration
=============================
This file contains ALL dialogue for weekly recaps and player recaps.
Edit the text here to customize what users see - no code changes needed!

Each dialogue entry uses {placeholders} that get filled with actual data.
"""

# ============================================================================
# WEEKLY RECAP DIALOGUE - Matchup Outcomes
# ============================================================================
# Key: (win, above_proj, proj_wins, win_ats)
# - win: 1 = won, 0 = lost
# - above_proj: 1 = beat your projection, 0 = missed it
# - proj_wins: 1 = you were favored, 0 = you were underdog
# - win_ats: 1 = beat the spread, 0 = didn't cover

WEEKLY_RECAP_OUTCOMES = {
    # ========================================================================
    # LOSSES (win = 0)
    # ========================================================================
    (0, 0, 0, 0): {
        "title": "💀 Complete Meltdown",
        "text": (
            "The haters said you couldn't do it and the haters were right. Shout out to the haters. "
            "We expected you to lose, but not like this. When you saw {opponent} was a {inv_odds_pct} favorite, "
            "you guys just decided to pack it in, losing by {margin_pos_str} compared to the {spread_flip_str} point spread going into the week. "
            "Maybe a players-only meeting can sort out this debacle."
        ),
        "emoji": "💀",
        "color": "red",
    },
    (0, 0, 0, 1): {
        "title": "🤷 Moral Victory?",
        "text": (
            "{opponent} knew he could go easy on you this week and you proved him right. "
            "Sure your {margin_pos_str} point loss was closer than the {spread_flip_str} spread going into the week, but you still lost. "
            "You missed your personal projection by {abs_proj_err_str}. "
            "You lost but at least you kept it close? Not much good to take from this one. They are who you thought they were — and you let them off the hook."
        ),
        "emoji": "🤷",
        "color": "orange",
    },
    (0, 0, 1, 0): {
        "title": "💔 Heartbreaker",
        "text": (
            "Now I know this one hurts. You had a {inv_odds_pct} chance of winning this matchup going into the week and then… well, I don't need to tell you what happened. "
            "You lost by {margin_pos_str} when you were expected to win by {spread_str}. "
            "You even missed your personal projections by {abs_proj_err_str}. Scrap that entire game plan because you won't make it far playing like this."
        ),
        "emoji": "💔",
        "color": "red",
    },
    (0, 1, 0, 0): {
        "title": "⚔️ Noble Defeat",
        "text": (
            "You gave it your best but {opponent}'s best was better. "
            "You exceeded your projected score by {proj_err_pos_str}. Goliath beat David this week, but at least you gave it your all."
        ),
        "emoji": "⚔️",
        "color": "blue",
    },
    (0, 1, 0, 1): {
        "title": "🫱 Went Down Swinging",
        "text": (
            "You came out fighting this week. The projections had you losing by {spread_flip_str} but you got this game pretty close — "
            "only losing by {margin_pos_str}. You exceeded your projections by {proj_err_pos_str}; just not enough to pull off the upset."
        ),
        "emoji": "🫱",
        "color": "orange",
    },
    (0, 1, 1, 0): {
        "title": "😤 Deserved Better",
        "text": (
            "{opponent} had something to prove this week. You deserved better. "
            "You beat your projection by {proj_err_pos_str} but still lost a game where you were favored. Just remember: defense wins championships."
        ),
        "emoji": "😤",
        "color": "orange",
    },
    # ========================================================================
    # WINS (win = 1)
    # ========================================================================
    (1, 0, 0, 1): {
        "title": "🍀 Lucky Break",
        "text": (
            "Sometimes your opponent decides to help you out. Despite a {odds_pct} chance of winning coming into the week and "
            "missing your expected point total by {abs_proj_err_str}, you still came away with the victory. Way to steal one!"
        ),
        "emoji": "🍀",
        "color": "green",
    },
    (1, 0, 1, 0): {
        "title": "💪 Ugly Win",
        "text": (
            "The great teams can win ugly. Sure you only won by {margin_pos_str} compared to the {spread_str} margin entering the week, "
            "but hey — a win is a win."
        ),
        "emoji": "💪",
        "color": "green",
    },
    (1, 0, 1, 1): {
        "title": "😅 Close Call",
        "text": (
            "Did you and {opponent} agree to take it easy on each other this week? "
            "You missed your projected score by {abs_proj_err_str} but still came away with the {margin_pos_str} point victory. Way to survive and advance."
        ),
        "emoji": "😅",
        "color": "green",
    },
    (1, 1, 0, 1): {
        "title": "🔥 Giant Slayer",
        "text": (
            "Everyone doubted you, but you pulled through. You won despite the odds makers only giving you a {odds_pct} chance of winning at the beginning of the week. "
            "You exceeded your projected score by {proj_err_pos_str} and stepped up for a big win."
        ),
        "emoji": "🔥",
        "color": "green",
    },
    (1, 1, 1, 0): {
        "title": "✊ Gritty Victory",
        "text": (
            "{opponent} gave you everything they had, but it wasn't enough to stop your boys. "
            "Sure, you only won by {margin_pos_str} — but close only counts in horseshoes and grenades."
        ),
        "emoji": "✊",
        "color": "green",
    },
    (1, 1, 1, 1): {
        "title": "🏆 Dominant Performance",
        "text": (
            "You exceeded your lofty expectations. You had a {odds_pct} chance of winning and won by {margin_pos_str}; "
            "we had you pegged for a {spread_str} point favorite going into the week."
        ),
        "emoji": "🏆",
        "color": "green",
    },
}

# ============================================================================
# CONTEXTUAL MESSAGES - Additional flavor text
# ============================================================================

CONTEXTUAL_MESSAGES = {
    "close_game": {
        "win": "That was a nail-biter! You won by just {margin_pos_str} points. Your heart rate is probably still elevated.",
        "loss": "So close yet so far. Lost by just {margin_pos_str} points. That's gotta sting.",
    },
    "blowout": {
        "win": "Absolutely demolished {opponent} by {margin_pos_str} points. No mercy!",
        "loss": "Ouch. Got blown out by {margin_pos_str} points. Might want to look away from this one.",
    },
    "league_performance": {
        "above_median": "You beat the league median this week ({weekly_median:.1f} pts), scoring {team_points:.1f}. Rock solid.",
        "below_median": "You fell below the league median this week ({weekly_median:.1f} pts), only scoring {team_points:.1f}. Room for improvement.",
        "dominated": "You would've beaten {teams_beat} other teams this week. Dominant performance!",
        "struggled": "You only would've beaten {teams_beat} other teams this week. Yikes.",
    },
    # ========================================================================
    # OPTIMAL LINEUP CONTEXT
    # ========================================================================
    "optimal_lineup": {
        "left_points": {
            "major": "You left {optimal_diff:.1f} points on the bench this week! That's {optimal_pct:.0f}% of your optimal score. Ouch!",
            "massive": "You left {optimal_diff:.1f} points on the bench this week! That's {optimal_pct:.0f}% of your optimal score. Ouch!",
            "significant": "You left {optimal_diff:.1f} points on your bench. Better lineup management could've changed this outcome.",
            "minor_win": "Pretty close to optimal! Only {optimal_diff:.1f} points left on the bench — and you still came away with the W! Nice work.",
            "minor_loss": "Pretty close to optimal — only {optimal_diff:.1f} points left on the bench. If you'd started them, this one might have swung the other way.",
            "minor": "Pretty close to optimal! Only {optimal_diff:.1f} points left on the bench.",
            "perfect": "PERFECT LINEUP! You started your best possible team. That's what we like to see!",
            "swing_loss": "You lost by {margin_pos_str} but had {optimal_diff:.1f} points on your bench — a lineup change would've swung the matchup. Brutal.",
            "near_swing": "Left {optimal_diff:.1f} on the bench and lost by {margin_pos_str}. That's one move away from a win.",
        },
        "optimal_win": "If you had set your optimal lineup, you would've scored {optimal_points:.1f} and {optimal_result}!",
        "optimal_record": "Your optimal lineup record this season is {optimal_wins}-{optimal_losses}. Your actual record is {wins}-{losses}. {optimal_comment}",
    },
    # ========================================================================
    # STREAKS & MOMENTUM
    # ========================================================================
    "streaks": {
        "winning_streak": {
            "hot": "🔥 {winning_streak} straight wins! You're on FIRE!",
            "rolling": "You're riding a {winning_streak}-game winning streak. Keep it rolling!",
            "started": "Back-to-back wins! A streak is born.",
        },
        "losing_streak": {
            "crisis": "😱 {losing_streak} straight losses. This is a full-blown crisis!",
            "slump": "You've lost {losing_streak} in a row. Time to shake things up.",
            "trouble": "Two straight losses. Time to stop the bleeding.",
        },
    },
    # ========================================================================
    # WEEKLY RANKING CONTEXT
    # ========================================================================
    "weekly_ranking": {
        "best_week": "🏆 This was your BEST SCORING WEEK of the season! Previous high was {prev_best:.1f}.",
        "worst_week": "💩 This was your worst week of the season. Your previous low was {prev_worst:.1f}.",
        "top_scorer": "👑 You were the #1 scorer in the league this week with {team_points:.1f} points!",
        "bottom_scorer": "You finished {opp_pts_week_rank} out of {league_size} teams this week in scoring.",
        "percentile_high": "Your {team_points:.1f} points ranked in the {opp_pts_week_pct:.0f}th percentile for the week. Elite performance!",
        "percentile_mid": "You scored {team_points:.1f}, putting you in the {opp_pts_week_pct:.0f}th percentile this week.",
        "percentile_low": "Your {team_points:.1f} points ranked in the {opp_pts_week_pct:.0f}th percentile. Bottom tier this week.",
    },
    # ========================================================================
    # PLAYOFF IMPLICATIONS
    # ========================================================================
    "playoff_race": {
        "locked_in": "You've clinched a playoff spot! Now it's all about seeding.",
        "safe": "Looking good! You'd make the playoffs if the season ended today (#{playoff_seed_to_date} seed).",
        "bubble": "You're on the playoff bubble at #{playoff_seed_to_date}. Every game matters now!",
        "must_win": "This is a MUST-WIN situation. You're fighting for your playoff life!",
        "eliminated": "Playoffs are out of reach, but pride is still on the line.",
        "bye_week_locked": "You've locked up a first-round bye! Rest those starters mentally.",
        "bye_week_chase": "Win this week and you're in position for a bye week!",
    },
    # ========================================================================
    # FELO RATING & POWER RANKINGS
    # ========================================================================
    "power_rankings": {
        "elite": "Your FELO score of {felo_score:.0f} puts you in the {felo_tier} tier. You're a {power_rating:.1f} power rating juggernaut!",
        "rising": "Your power rating climbed to {power_rating:.1f} this week. You're trending up at the right time!",
        "falling": "Your power rating dropped to {power_rating:.1f}. Need to turn things around.",
        "dominant": "You have the highest power rating in the league at {power_rating:.1f}. Fear the champ!",
    },
    # ========================================================================
    # HEAD-TO-HEAD RECORDS
    # ========================================================================
    "h2h_records": {
        "dominated_opponent": "You're now {h2h_wins}-{h2h_losses} all-time against {opponent}. You own them!",
        "rivalry": "This evens your all-time record with {opponent} at {h2h_wins}-{h2h_losses}. True rivals!",
        "owned": "{opponent} now leads the all-time series {opp_h2h_wins}-{opp_h2h_losses}. Revenge is sweet... when you get it.",
        "first_meeting": "This was your first ever matchup with {opponent}. {h2h_result}!",
        "season_sweep": "You've beaten {opponent} twice this season! Season sweep complete!",
    },
    # ========================================================================
    # SCHEDULE LUCK & WINS ANALYSIS
    # ========================================================================
    "schedule_luck": {
        "very_lucky": "You've been EXTREMELY lucky with your schedule. You have {exp_final_wins:.1f} expected wins based on points, but actually have {wins_to_date} wins. That's {wins_diff:+.1f} extra wins from an easy schedule!",
        "lucky": "Lucky you! Your schedule has gifted you {wins_diff:+.1f} extra wins this season. You'd have {exp_final_wins:.1f} wins with an average schedule.",
        "unlucky": "The schedule hasn't been kind. You should have {exp_final_wins:.1f} wins based on your scoring, but you only have {wins_to_date}. That's {wins_diff:.1f} wins lost to bad luck!",
        "very_unlucky": "BRUTAL schedule luck! You've lost {wins_diff:.1f} wins to tough matchups despite strong scoring.",
        "fair": "Your schedule has been fair. Your {wins_to_date} wins align with your {exp_final_wins:.1f} expected wins.",
    },
    "alternate_schedules": {
        "playoffs_most": "In {shuffle_avg_playoffs:.0%} of possible schedules, you'd make the playoffs. You're in a good spot!",
        "playoffs_some": "In {shuffle_avg_playoffs:.0%} of schedules, you'd be playoff-bound. It's close!",
        "playoffs_few": "Only {shuffle_avg_playoffs:.0%} of possible schedules have you in the playoffs. Need to score more points!",
        "avg_wins": "Across all possible schedules, you'd average {shuffle_avg_wins:.1f} wins (you have {wins_to_date}).",
        "avg_seed": "Your average playoff seed across all schedules would be #{shuffle_avg_seed:.1f} (currently #{playoff_seed_to_date}).",
        "bye_chance": "You'd get a bye week in {shuffle_avg_bye:.0%} of schedules.",
    },
    # ========================================================================
    # SEASON MILESTONES
    # ========================================================================
    "milestones": {
        "first_win": "🎉 Your first win of the season! Feels good, doesn't it?",
        "clinch_playoffs": "🎊 YOU'RE GOING TO THE PLAYOFFS! Clinched with this win!",
        "clinch_bye": "🌟 FIRST ROUND BYE SECURED! You're one of the top dogs!",
        "championship": "🏆 LEAGUE CHAMPION! The trophy is yours!",
        "sacko_avoid": "You avoided the Sacko! Not last place feels pretty good, right?",
        "sacko_claim": "💩 Congratulations? You've secured last place and the Sacko.",
        "500_record": "Back to .500! All square at {wins_to_date}-{losses_to_date}.",
        "above_500": "You're above .500 for the first time! {wins_to_date}-{losses_to_date} feels nice.",
        "playoff_elimination": "Your playoff hopes are mathematically eliminated. Play for pride!",
    },
    # ========================================================================
    # GRADE & GPA CONTEXT
    # ========================================================================
    "performance_grade": {
        "a_plus": "💯 A+ grade with a {gpa:.2f} GPA this week! Perfect execution!",
        "a_range": "📚 Solid {grade} grade (GPA: {gpa:.2f}). Dean's list performance!",
        "b_range": "📖 {grade} grade this week. GPA of {gpa:.2f}. Room to improve but respectable.",
        "c_range": "📝 {grade} grade. Your {gpa:.2f} GPA is... average. You're the middle of the pack.",
        "d_range": "📉 {grade}? GPA of {gpa:.2f}? You're on academic probation, my friend.",
        "f_range": "❌ {grade}. GPA: {gpa:.2f}. You failed this week. Time to hit the books!",
    },
    # ========================================================================
    # FAAB & ROSTER MOVES
    # ========================================================================
    "roster_management": {
        "active_manager": "You made {number_of_moves} roster moves this week. Working that waiver wire!",
        "inactive_manager": "Zero moves this week? Sometimes staying put is the right call... sometimes.",
        "faab_rich": "You still have ${faab_balance} in FAAB. Plenty of dry powder for the stretch run!",
        "faab_broke": "You're down to ${faab_balance} in FAAB. Budget is tight!",
        "trade_active": "You've made {number_of_trades} trades this season. Wheeling and dealing!",
        "trade_inactive": "No trades yet this season. Standing pat or nobody wants your players?",
    },
}

# ============================================================================
# PLAYER WEEKLY RECAP DIALOGUE - Enhanced with Percentiles
# ============================================================================

PLAYER_PERFORMANCE = {
    # ========================================================================
    # PERCENTILE-BASED PERFORMANCE
    # ========================================================================
    "percentile_performance": {
        "elite_week": "{player} had an ELITE week, scoring in the {percentile:.0f}th percentile among all {position} performances this season!",
        "great_week": "{player} balled out! {percentile:.0f}th percentile performance for a {position} this week.",
        "solid_week": "{player} had a solid outing ({percentile:.0f}th percentile among {position}s).",
        "mediocre_week": "{player} was mediocre, {percentile:.0f}th percentile for {position}s this week.",
        "poor_week": "{player} struggled with a {percentile:.0f}th percentile showing for a {position}.",
        "awful_week": "{player} was AWFUL. Bottom {percentile:.0f}th percentile among all {position}s. Yikes.",
    },
    # ========================================================================
    # ALL-TIME & HISTORICAL CONTEXT
    # ========================================================================
    "historical_performance": {
        "all_time_best": "🔥 {player} just had their BEST GAME EVER for your team! Previous high: {prev_high:.1f}",
        "season_best": "⭐ {player}'s best game of the season! Beat their previous high of {prev_high:.1f}",
        "all_time_worst": "💩 {player} just had their worst game in your team's history. Brutal.",
        "career_game": "{player} ({points:.1f} pts) is now #{rank} in your all-time single-game performances!",
        "position_record": "{player} set a new record for {position} on your team with {points:.1f} points!",
    },
    # ========================================================================
    # #1 PLAYER ACHIEVEMENTS
    # ========================================================================
    "number_one_player": {
        "league_week": "👑 {player} was the #1 scoring player in the ENTIRE LEAGUE this week with {points:.1f} points!",
        "league_season": "🏆 {player} is currently the #1 player in the league this season!",
        "position_week": "📈 {player} was the top {position} in the league this week!",
        "position_season": "🥇 {player} is the #1 {position} in fantasy this season!",
        "team_week": "⭐ {player} was your MVP this week, leading your team in scoring.",
        "team_season": "👏 {player} has been your most consistent player all season long.",
    },
    # ========================================================================
    # MANAGER/PLAYER/POSITION HISTORY HIGHLIGHTS (percentile-driven)
    # ========================================================================
    "history_highlights": {
        # All messages assume percentile already on a 0-100 scale
        "mgr_player_all_time_top": "For you, this ranks in your top {bucket} weeks with {player} all-time.",
        "mgr_player_season_top": "This is a top {bucket} week with {player} this season.",
        "mgr_pos_all_time_top": "On your roster, this sits in the top {bucket} all-time for a {position}.",
        "mgr_pos_season_top": "For your {position} group this season, this is top {bucket}.",
        "player_personal_all_time_top": "For {player}, this is a top {bucket} career fantasy week.",
        "player_personal_season_top": "{player}'s season form: top {bucket} week.",
        "league_pos_all_time_top": "League-wide, a {position} score like this lands in the top {bucket} historically.",
        "league_pos_season_top": "Across the league this season, this is a top {bucket} {position} performance.",
        "league_player_all_time_top": "Across all managers, this week is a top {bucket} for {player} historically.",
        "league_player_season_top": "This season, {player}'s week ranks in the top {bucket} across the league.",
    },
    # ========================================================================
    # POSITION GROUP PERFORMANCE
    # ========================================================================
    "position_group": {
        "group_dominated": "Your {position} corps dominated this week! Combined for {total_points:.1f} points.",
        "group_struggled": "Your {position}s struggled this week, combining for only {total_points:.1f} points.",
        "rb_committee": "The RB committee approach is working! Your RBs combined for {total_points:.1f}.",
        "wr_depth": "WR depth paying off! {wr_count} receivers scored {total_points:.1f} combined.",
        "te_premium": "Your TE advantage was huge this week! {points:.1f} from the position.",
    },
}

# ============================================================================
# SEASON ANALYSIS DIALOGUE - Enhanced
# ============================================================================

SEASON_ANALYSIS = {
    "current_record": {
        "in_playoffs": "So far your record is {wins}-{losses} and you would be the #{seed} seed in the playoffs if the season ended today. You're in playoff position!",
        "out_of_playoffs": "So far your record is {wins}-{losses} and you would be the #{seed} seed if the season ended today. You are NOT in playoff position.",
        "no_seed": "So far your record is {wins}-{losses}.",
        "dominant": "Your {wins}-{losses} record has you sitting pretty at #{seed}. Crushing it!",
        "struggling": "At {wins}-{losses}, you're #{seed} and fighting to stay relevant.",
    },
    "playoff_probability": {
        "locked": "You're a {playoff_pct} LOCK for the playoffs! Start planning your championship parade.",
        "high": "You have a {playoff_pct} chance of making the playoffs (projected #{avg_seed:.1f} seed). Looking good!",
        "medium": "You're at {playoff_pct} to make the playoffs. It's going to come down to the wire!",
        "low": "Only a {playoff_pct} shot at the playoffs. You'll need some help and some wins.",
        "eliminated": "Mathematically eliminated from playoff contention. There's always next year!",
        "championship_odds": "Your championship odds? {champ_pct}. {champ_comment}",
    },
    "expected_wins": {
        "very_lucky": "You should have {exp_final_wins:.2f} wins based on points scored, but you have {wins}! You've been gifted {extra_wins:.2f} wins by an easy schedule. Enjoy it while it lasts!",
        "lucky": "You should have about {exp_final_wins:.2f} wins, but you have {wins}. That's {extra_wins:.2f} extra wins from favorable matchups!",
        "neutral": "Your {wins} wins align with your expected {exp_final_wins:.2f} wins. Fair schedule so far.",
        "unlucky": "Based on your scoring, you should have {exp_final_wins:.2f} wins, but you only have {wins}. You've been ROBBED of {abs_extra_wins:.2f} wins by a tough schedule!",
        "very_unlucky": "Brutal! You should have {exp_final_wins:.2f} wins but only have {wins}. The schedule has cost you {abs_extra_wins:.2f} wins!",
    },
    "possible_schedules": {
        "dominating": "About {playoff_pct_schedules:.0%} of possible schedules would have you in the playoffs. You're scoring enough to succeed!",
        "struggling": "Only {playoff_pct_schedules:.0%} of schedules put you in playoff position. Need to score more points, not just get lucky!",
        "bubble": "About {playoff_pct_schedules:.0%} of schedules have you playoff-bound. You're on the bubble no matter what!",
    },
    "efficiency": {
        "efficient_wins": "You're {win_eff:.1%} efficient in converting scoring into wins. Making the most of your points!",
        "inefficient_wins": "Only {win_eff:.1%} win efficiency? You're not converting your scoring into W's.",
        "efficient_losses": "Ouch, {loss_eff:.1%} of your potential wins turned into losses. That's rough.",
    },
}

# ============================================================================
# HELPER: Get outcome key from flags
# ============================================================================


def get_outcome_key(win: int, above_proj: int, proj_wins: int, win_ats: int) -> tuple:
    """Convert boolean flags to outcome key tuple."""
    return (win, above_proj, proj_wins, win_ats)


def get_outcome_dialogue(
    win: int, above_proj: int, proj_wins: int, win_ats: int
) -> dict:
    """
    Get the dialogue configuration for a specific outcome.

    Returns dict with keys: title, text, emoji, color
    """
    key = get_outcome_key(win, above_proj, proj_wins, win_ats)
    return WEEKLY_RECAP_OUTCOMES.get(
        key,
        {
            "title": "🤔 Unknown Outcome",
            "text": "Hmm, we're not sure what happened here. The data is incomplete.",
            "emoji": "🤔",
            "color": "gray",
        },
    )


def format_dialogue(template: str, **kwargs) -> str:
    """
    Format a dialogue template with provided values.
    Missing values will show as 'N/A'.
    """
    try:
        return template.format(**kwargs)
    except KeyError:
        # If a key is missing, try to continue with what we have

        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


# ============================================================================
# OUTCOME SUMMARY - Quick reference guide
# ============================================================================


def print_outcome_summary():
    """
    Prints a readable summary of all 16 possible outcomes.
    Useful for understanding what triggers each message.
    """
    print("\n" + "=" * 80)
    print("WEEKLY RECAP OUTCOMES REFERENCE GUIDE")
    print("=" * 80)

    labels = {
        "win": ["LOSS", "WIN"],
        "above_proj": ["Missed Proj", "Beat Proj"],
        "proj_wins": ["Underdog", "Favored"],
        "win_ats": ["Didn't Cover", "Covered Spread"],
    }

    for key, config in sorted(WEEKLY_RECAP_OUTCOMES.items()):
        w, a, pw, ats = key
        print(f"\n{config['emoji']} {config['title']}")
        print(
            f"   Outcome: {labels['win'][w]} | {labels['above_proj'][a]} | {labels['proj_wins'][pw]} | {labels['win_ats'][ats]}"
        )
        print(f"   Color: {config['color']}")
        print(f"   Text: {config['text'][:100]}...")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run this file directly to see all outcomes
    print_outcome_summary()

    # Example usage:
    print("\n\nEXAMPLE USAGE:")
    print("-" * 80)

    # Simulate a dominant win
    outcome = get_outcome_dialogue(win=1, above_proj=1, proj_wins=1, win_ats=1)
    print(f"\n{outcome['emoji']} {outcome['title']}")

    # Format with sample data
    sample_data = {
        "opponent": "Marc",
        "odds_pct": "65%",
        "margin_pos_str": "23.4",
        "spread_str": "18.5",
        "proj_err_pos_str": "12.3",
    }
    print(format_dialogue(outcome["text"], **sample_data))
