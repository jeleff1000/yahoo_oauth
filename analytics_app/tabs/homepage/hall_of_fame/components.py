"""
Hall of Fame Reusable Components

Provides HTML card generators to eliminate duplication across Hall of Fame files.
All components use CSS variables for theme support.
"""

from typing import Optional


def champion_card(
    year: int,
    winner: str,
    runner_up: str,
    score: str,
    winner_pts: Optional[float] = None,
) -> str:
    """
    Generate HTML for a championship result card.

    Args:
        year: Championship year
        winner: Winner's name
        runner_up: Runner-up's name
        score: Score string (e.g., "142.5 - 138.2")
        winner_pts: Optional winner's score for highlighting
    """
    return f"""
        <div class='hof-timeline-card'>
            <div class='timeline-year'>{year}</div>
            <div class='timeline-winner'>🏆 {winner}</div>
            <div class='timeline-details'>def. {runner_up} • {score}</div>
        </div>
    """


def record_card(
    title: str,
    holder: str,
    value: str,
    year: Optional[int] = None,
    context: Optional[str] = None,
) -> str:
    """
    Generate HTML for a record highlight card.

    Args:
        title: Record title (e.g., "Highest Single Week")
        holder: Record holder's name
        value: Record value (e.g., "162.4 pts")
        year: Optional year
        context: Optional additional context
    """
    year_str = f"<span class='record-year'>{year}</span>" if year else ""
    context_str = f"<div class='record-context'>{context}</div>" if context else ""

    return f"""
        <div class='hof-record-card'>
            <div class='hof-record-label'>{title}</div>
            <div class='hof-record-value'>{value}</div>
            <div class='hof-record-detail'>{holder} {year_str}</div>
            {context_str}
        </div>
    """


def game_card(
    winner: str,
    loser: str,
    winner_pts: float,
    loser_pts: float,
    year: int,
    week: int,
    is_playoff: bool = False,
    highlight_stat: Optional[str] = None,
    highlight_label: Optional[str] = None,
) -> str:
    """
    Generate HTML for a game result card.

    Args:
        winner: Winner's name
        loser: Loser's name
        winner_pts: Winner's score
        loser_pts: Loser's score
        year: Game year
        week: Game week
        is_playoff: Whether it's a playoff game
        highlight_stat: Optional stat to highlight (e.g., "280.5 Total")
        highlight_label: Optional label for highlight
    """
    playoff_class = " hof-game-card-playoff" if is_playoff else ""
    highlight_html = ""
    if highlight_stat:
        highlight_html = f"""
            <span class='game-stat' style='color: var(--success, #059669);'>{highlight_stat}</span>
        """

    return f"""
        <div class='hof-game-card{playoff_class}'>
            <div class='game-header'>
                <span class='game-date'>{year} Week {week}</span>
                {highlight_html}
            </div>
            <div style='font-size: 0.95rem;'>
                <div style='margin-bottom: 0.2rem;'>
                    <span class='team-name'>✅ {winner}</span>
                    <span style='float: right;' class='team-score'>{winner_pts:.1f}</span>
                </div>
                <div>
                    <span class='loser-name'>❌ {loser}</span>
                    <span style='float: right;' class='loser-score'>{loser_pts:.1f}</span>
                </div>
            </div>
        </div>
    """


def upset_card(
    favored: str,
    underdog: str,
    favored_pts: float,
    underdog_pts: float,
    favored_proj: float,
    underdog_proj: float,
    year: int,
    week: int,
    proj_diff: float,
    is_playoff: bool = False,
) -> str:
    """
    Generate HTML for an upset game card.

    Args:
        favored: Favored team (lost)
        underdog: Underdog team (won)
        favored_pts: Favored team's actual score
        underdog_pts: Underdog team's actual score
        favored_proj: Favored team's projected score
        underdog_proj: Underdog team's projected score
        year: Game year
        week: Game week
        proj_diff: Projection differential
        is_playoff: Whether it's a playoff game
    """
    playoff_class = " hof-game-card-playoff" if is_playoff else ""

    return f"""
        <div class='hof-game-card{playoff_class}' style='border-left: 3px solid var(--accent, #8B5CF6);'>
            <div class='game-header'>
                <span class='game-date'>{year} Week {week}</span>
                <span class='game-stat' style='color: var(--accent, #8B5CF6);'>Upset by {proj_diff:.1f}</span>
            </div>
            <div style='font-size: 0.9rem;'>
                <div style='margin-bottom: 0.3rem; padding-bottom: 0.3rem; border-bottom: 1px solid var(--border-subtle, rgba(255,255,255,0.2));'>
                    <div class='loser-name' style='font-size: 0.8rem; margin-bottom: 0.2rem;'>Favored (Lost)</div>
                    <span class='team-name'>{favored}</span>
                    <span style='float: right;'>
                        <span class='loser-score' style='font-size: 0.85rem;'>Proj: {favored_proj:.1f}</span>
                        <span class='team-score' style='margin-left: 0.5rem;'>{favored_pts:.1f}</span>
                    </span>
                </div>
                <div>
                    <div style='color: var(--success, #059669); font-size: 0.8rem; margin-bottom: 0.2rem;'>Underdog (Won)</div>
                    <span class='team-name' style='color: var(--success, #059669);'>{underdog}</span>
                    <span style='float: right;'>
                        <span class='loser-score' style='font-size: 0.85rem;'>Proj: {underdog_proj:.1f}</span>
                        <span style='margin-left: 0.5rem; color: var(--success, #059669); font-weight: bold;'>{underdog_pts:.1f}</span>
                    </span>
                </div>
            </div>
        </div>
    """


def rivalry_card(
    team_a: str,
    team_b: str,
    team_a_wins: int,
    team_b_wins: int,
    total_games: int,
    avg_combined: float,
    is_playoff: bool = False,
) -> str:
    """
    Generate HTML for a rivalry matchup card.

    Args:
        team_a: First team name
        team_b: Second team name
        team_a_wins: Team A's wins
        team_b_wins: Team B's wins
        total_games: Total games played
        avg_combined: Average combined score
        is_playoff: Whether these are playoff matchups
    """
    playoff_class = " hof-game-card-playoff" if is_playoff else ""

    return f"""
        <div class='hof-rivalry-card{playoff_class}'>
            <div class='rivalry-header'>
                <span class='rivalry-teams'>{team_a} vs {team_b}</span>
                <span class='rivalry-games'>{total_games} games</span>
            </div>
            <div class='rivalry-stats'>
                <span>{team_a}: <b>{team_a_wins}</b></span>
                <span>{team_b}: <b>{team_b_wins}</b></span>
                <span style='color: var(--text-muted, #9CA3AF);'>Avg: {avg_combined:.1f}</span>
            </div>
        </div>
    """


def narrative_callout(text: str, emoji: str = "💡") -> str:
    """
    Generate HTML for a narrative/fun fact callout.

    Args:
        text: The narrative text
        emoji: Emoji to display
    """
    return f"""
        <div class='hof-narrative-callout'>
            <span class='callout-emoji'>{emoji}</span>
            <span class='callout-text'>{text}</span>
        </div>
    """


def leader_card(
    rank: int,
    name: str,
    primary_stat: str,
    primary_label: str,
    secondary_stats: Optional[dict] = None,
    is_champion: bool = False,
) -> str:
    """
    Generate HTML for a leaderboard entry card.

    Args:
        rank: Position rank (1, 2, 3, etc.)
        name: Player/manager name
        primary_stat: Main stat value
        primary_label: Label for main stat
        secondary_stats: Dict of additional stats {label: value}
        is_champion: Whether to show champion badge
    """
    # Medal for top 3
    if rank == 1:
        medal = "🥇"
        border_color = "#FFD700"
    elif rank == 2:
        medal = "🥈"
        border_color = "#C0C0C0"
    elif rank == 3:
        medal = "🥉"
        border_color = "#CD7F32"
    else:
        medal = f"#{rank}"
        border_color = "var(--border, #E5E7EB)"

    champ_badge = " 🏆" if is_champion else ""

    # Build secondary stats HTML
    secondary_html = ""
    if secondary_stats:
        stats_items = "".join(
            [
                f"<div><div class='season-stat-label'>{label}</div><div class='season-stat-value'>{value}</div></div>"
                for label, value in secondary_stats.items()
            ]
        )
        secondary_html = f"<div style='display: flex; gap: 1rem; margin-top: 0.5rem;'>{stats_items}</div>"

    return f"""
        <div class='hof-season-card' style='border-left: 4px solid {border_color};'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                <span class='season-rank'>{medal}</span>
            </div>
            <div class='season-manager'>{name}{champ_badge}</div>
            <div style='margin-top: 0.25rem;'>
                <div class='season-stat-label'>{primary_label}</div>
                <div class='season-stat-value stat-highlight'>{primary_stat}</div>
            </div>
            {secondary_html}
        </div>
    """


def season_card(
    rank: int,
    manager: str,
    year: int,
    total_points: float,
    wins: int,
    losses: int,
    ppg: float,
    is_champion: bool = False,
) -> str:
    """
    Generate HTML for a top season card.

    Args:
        rank: Position rank
        manager: Manager name
        year: Season year
        total_points: Total points scored
        wins: Number of wins
        losses: Number of losses
        ppg: Points per game
        is_champion: Whether this season won championship
    """
    secondary_stats = {"Record": f"{wins}-{losses}", "PPG": f"{ppg:.1f}"}

    return leader_card(
        rank=rank,
        name=f"{manager} ({year})",
        primary_stat=f"{total_points:.1f}",
        primary_label="Total Points",
        secondary_stats=secondary_stats,
        is_champion=is_champion,
    )


def week_card(
    manager: str,
    year: int,
    week: int,
    points: float,
    result: str,
    is_playoff: bool = False,
) -> str:
    """
    Generate HTML for a top week performance card.

    Args:
        manager: Manager name
        year: Season year
        week: Week number
        points: Points scored
        result: 'W' or 'L'
        is_playoff: Whether it was a playoff game
    """
    playoff_class = " hof-week-card-playoff" if is_playoff else ""
    result_color = "#059669" if result == "W" else "#DC2626"

    return f"""
        <div class='hof-week-card{playoff_class}'>
            <div class='week-info'>
                <div>
                    <span class='week-manager'>{manager}</span>
                    <span class='week-meta'> • {year} Wk{week}</span>
                </div>
                <div>
                    <span class='week-score'>{points:.1f}</span>
                    <span style='color: {result_color}; font-weight: bold; margin-left: 0.5rem;'>
                        {result}
                    </span>
                </div>
            </div>
        </div>
    """
