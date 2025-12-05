#!/usr/bin/env python3
"""
Yahoo Fantasy API Helpers

This module handles all Yahoo Fantasy Sports API operations including:
- Making API calls
- Fetching user games and leagues
- Fetching team/manager data
- Discovering league seasons
"""

import time
from typing import Optional
import requests


def yahoo_api_call(
    access_token: str,
    endpoint: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """
    Make authenticated call to Yahoo Fantasy API with retry logic.

    Uses exponential backoff for transient failures and rate limits.

    Args:
        access_token: Yahoo OAuth access token
        endpoint: API endpoint (e.g., "users;use_login=1/games")
        max_retries: Maximum number of retry attempts (default 3)
        base_delay: Base delay in seconds for exponential backoff (default 1.0)

    Returns:
        JSON response from the API

    Raises:
        requests.HTTPError: If all retries are exhausted
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/{endpoint}"

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30)

            # Success
            if resp.status_code == 200:
                return resp.json()

            # Rate limited - wait and retry
            if resp.status_code == 429:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                retry_after = resp.headers.get('Retry-After')
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                print(f"Rate limited (429), waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
                time.sleep(delay)
                continue

            # Server errors (5xx) - retry with backoff
            if 500 <= resp.status_code < 600:
                delay = base_delay * (2 ** attempt)
                print(f"Server error ({resp.status_code}), waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
                time.sleep(delay)
                continue

            # Client errors (4xx except 429) - don't retry, raise immediately
            resp.raise_for_status()

        except requests.exceptions.Timeout:
            delay = base_delay * (2 ** attempt)
            print(f"Request timeout, waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
            time.sleep(delay)
            last_exception = requests.exceptions.Timeout(f"Timeout for {endpoint}")
            continue

        except requests.exceptions.ConnectionError as e:
            delay = base_delay * (2 ** attempt)
            print(f"Connection error, waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
            time.sleep(delay)
            last_exception = e
            continue

        except requests.exceptions.HTTPError as e:
            # Non-retryable HTTP errors
            raise

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise requests.exceptions.HTTPError(f"Failed after {max_retries} retries for {endpoint}")


def get_user_games(access_token: str):
    """Get all games (sports/seasons) for the authenticated user."""
    return yahoo_api_call(access_token, "users;use_login=1/games?format=json")


def get_user_football_leagues(access_token: str, game_key: str):
    """Get NFL leagues for a specific game/season."""
    return yahoo_api_call(access_token, f"users;use_login=1/games;game_keys={game_key}/leagues?format=json")


def get_league_teams(access_token: str, league_key: str, year: int = None) -> list[dict]:
    """
    Fetch all teams/managers for a league.
    Returns list of dicts with team_name, manager_name, team_key, year.
    """
    try:
        # Request teams with managers sub-resource
        data = yahoo_api_call(access_token, f"league/{league_key}/teams/managers?format=json")
        teams = []

        league_data = data.get("fantasy_content", {}).get("league", [])
        if len(league_data) > 1:
            teams_data = league_data[1].get("teams", {})
            for key in teams_data:
                if key == "count":
                    continue

                team_entry = teams_data[key].get("team", [])
                team_name = "Unknown Team"
                manager_name = None

                # Parse team info - it's a nested structure
                for part in team_entry:
                    if isinstance(part, list):
                        for item in part:
                            if isinstance(item, dict):
                                if "name" in item:
                                    team_name = item["name"]
                                if "managers" in item:
                                    # Extract manager nickname
                                    mgrs = item["managers"]
                                    if isinstance(mgrs, list) and mgrs:
                                        mgr_data = mgrs[0].get("manager", {})
                                        manager_name = mgr_data.get("nickname") or mgr_data.get("manager_id")
                                    elif isinstance(mgrs, dict):
                                        for mk in mgrs:
                                            if mk != "count":
                                                mgr_data = mgrs[mk].get("manager", {})
                                                manager_name = mgr_data.get("nickname") or mgr_data.get("manager_id")
                                                break
                    elif isinstance(part, dict):
                        if "name" in part:
                            team_name = part["name"]
                        if "managers" in part:
                            mgrs = part["managers"]
                            if isinstance(mgrs, list) and mgrs:
                                mgr_data = mgrs[0].get("manager", {})
                                manager_name = mgr_data.get("nickname") or mgr_data.get("manager_id")

                # Only add if we have a valid manager_name (could be --hidden-- or actual name)
                teams.append({
                    "team_key": "",
                    "team_name": team_name,
                    "manager_name": manager_name if manager_name else "Unknown",
                    "year": year,
                })

        return teams
    except Exception as e:
        # Don't warn for old years that might not exist
        return []


def get_league_teams_all_years(access_token: str, league_name: str, games_data: dict) -> list[dict]:
    """
    Fetch teams/managers across all years for a league.
    Returns list of dicts with team_name, manager_name, team_key, year.
    """
    all_teams = []
    football_games = extract_football_games(games_data)

    for game in football_games:
        game_key = game.get("game_key")
        year = game.get("season")

        try:
            # Get leagues for this game/year
            leagues_data = get_user_football_leagues(access_token, game_key)
            leagues = (
                leagues_data.get("fantasy_content", {})
                .get("users", {}).get("0", {}).get("user", [])[1]
                .get("games", {}).get("0", {}).get("game", [])[1]
                .get("leagues", {})
            )

            for key in leagues:
                if key == "count":
                    continue
                league = leagues[key].get("league", [])[0]
                if league.get("name") == league_name:
                    league_key = league.get("league_key")
                    teams = get_league_teams(access_token, league_key, year)
                    all_teams.extend(teams)
                    break

        except Exception:
            continue

    return all_teams


def find_hidden_managers(teams: list[dict]) -> list[dict]:
    """Find teams with hidden manager names (only --hidden-- pattern)"""
    hidden_teams = []

    for team in teams:
        mgr_name = (team.get("manager_name") or "").strip()
        # Only flag actual "--hidden--" managers, not unknown/empty
        if mgr_name == "--hidden--" or mgr_name.lower() == "hidden":
            hidden_teams.append(team)

    return hidden_teams


def extract_football_games(games_data):
    """Extract NFL games from Yahoo API response, sorted by season descending (latest first)."""
    football_games = []
    try:
        games = games_data.get("fantasy_content", {}).get("users", {}).get("0", {}).get("user", [])[1].get("games", {})
        for key in games:
            if key == "count":
                continue
            game = games[key].get("game")
            if isinstance(game, list):
                game = game[0]
            if game and game.get("code") == "nfl":
                football_games.append({
                    "game_key": game.get("game_key"),
                    "season": game.get("season"),
                    "name": game.get("name"),
                })
    except Exception:
        pass
    # Sort by season descending so latest year appears first in dropdown
    football_games.sort(key=lambda g: int(g.get("season", 0)), reverse=True)
    return football_games


def seasons_for_league_name(access_token: str, all_games: list[dict], target_league_name: str) -> list[str]:
    """Find all seasons where this league exists"""
    seasons = set()
    for g in all_games:
        game_key = g.get("game_key")
        season = str(g.get("season", "")).strip()
        if not game_key or not season:
            continue
        try:
            leagues_data = get_user_football_leagues(access_token, game_key)
            leagues = (
                leagues_data.get("fantasy_content", {})
                .get("users", {}).get("0", {}).get("user", [])[1]
                .get("games", {}).get("0", {}).get("game", [])[1]
                .get("leagues", {})
            )
            for key in leagues:
                if key == "count":
                    continue
                league = leagues[key].get("league", [])[0]
                name = league.get("name")
                if name == target_league_name:
                    seasons.add(season)
                    break
        except Exception:
            pass
    return sorted(seasons)


def get_all_leagues_with_years(access_token: str, all_games: list[dict]) -> list[dict]:
    """
    Get all unique leagues across all seasons with their year ranges.

    Returns list of dicts:
        - name: league name
        - years: sorted list of years [2015, 2016, ..., 2025]
        - year_range: formatted string like "(2015-2025)" or "(2024)"
        - display_name: "League Name (2015-2025)"
        - latest_league_key: league_key from most recent season
        - latest_season: most recent season
        - num_teams: team count from most recent season
    """
    from collections import defaultdict

    # Map league_name -> list of (season, league_key, num_teams)
    league_info = defaultdict(list)

    for g in all_games:
        game_key = g.get("game_key")
        season = str(g.get("season", "")).strip()
        if not game_key or not season:
            continue
        try:
            leagues_data = get_user_football_leagues(access_token, game_key)
            leagues = (
                leagues_data.get("fantasy_content", {})
                .get("users", {}).get("0", {}).get("user", [])[1]
                .get("games", {}).get("0", {}).get("game", [])[1]
                .get("leagues", {})
            )
            for key in leagues:
                if key == "count":
                    continue
                league = leagues[key].get("league", [])[0]
                name = league.get("name")
                league_key = league.get("league_key")
                num_teams = league.get("num_teams")
                if name:
                    league_info[name].append({
                        "season": int(season),
                        "league_key": league_key,
                        "num_teams": num_teams,
                    })
        except Exception:
            pass

    # Build result list
    result = []
    for name, seasons_list in league_info.items():
        # Sort by season descending
        seasons_list.sort(key=lambda x: x["season"], reverse=True)
        years = sorted([s["season"] for s in seasons_list])

        # Format year range
        if len(years) == 1:
            year_range = f"({years[0]})"
        else:
            year_range = f"({years[0]}-{years[-1]})"

        latest = seasons_list[0]
        result.append({
            "name": name,
            "years": years,
            "year_range": year_range,
            "display_name": f"{name} {year_range}",
            "latest_league_key": latest["league_key"],
            "latest_season": latest["season"],
            "num_teams": latest["num_teams"],
        })

    # Sort by latest season descending, then name
    result.sort(key=lambda x: (-x["latest_season"], x["name"]))
    return result
