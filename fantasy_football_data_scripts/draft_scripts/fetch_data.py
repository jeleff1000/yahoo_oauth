import re
import requests
import time
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
import yahoo_fantasy_api as yfa
from draft_result import DraftResult
from imports_and_utils import clean_name, apply_manual_mapping

# Add parent directory to path for oauth_utils import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oauth_utils import create_oauth2

# ---- Timeout/Retry modeled after yahoo_fantasy_data ----
class APITimeoutError(Exception):
    """Signal a recoverable API timeout so caller can resume or save partials."""
    pass

class RecoverableAPIError(APITimeoutError):
    """Transient API failures (rate-limit, 403/429, 'Request denied', empty XML, etc.)."""
    pass

_XML_NS_RE = re.compile(r' xmlns="[^"]+"')

def fetch_url(url, oauth, max_retries: int = 6, backoff: float = 0.5) -> ET.Element:
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = oauth.session.get(url, timeout=30)
            try:
                r.raise_for_status()
            except requests.HTTPError as he:
                code = getattr(he.response, "status_code", None)
                if code in (429, 403, 502, 503, 504):
                    if attempt == max_retries - 1:
                        raise RecoverableAPIError(f"HTTP {code} on {url}") from he
                    time.sleep(backoff * (2 ** attempt))
                    continue
                raise
            text = (r.text or "").strip()
            if not text or "Request denied" in text:
                if attempt == max_retries - 1:
                    raise RecoverableAPIError(f"Empty or denied response from {url}")
                time.sleep(backoff * (2 ** attempt))
                continue
            xmlstring = _XML_NS_RE.sub("", text, count=1)
            return ET.fromstring(xmlstring)
        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise APITimeoutError(f"Timeout fetching {url}") from e
            time.sleep(backoff * (2 ** attempt))
        except (requests.RequestException, ET.ParseError, ValueError) as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise RecoverableAPIError(f"Transient error fetching {url}: {e}") from e
            time.sleep(backoff * (2 ** attempt))
    if isinstance(last_exc, requests.exceptions.Timeout):
        raise APITimeoutError(f"Timeout fetching {url}") from last_exc
    raise RecoverableAPIError(f"Unknown fetch_url failure for {url}: {last_exc}")

# ---- existing funcs, now using fetch_url ----

def fetch_draft_data(oauth, league_id):
    league = yfa.League(oauth, league_id)
    draft_results = league.draft_results()
    return [DraftResult(result) for result in draft_results]

def fetch_team_and_player_mappings(oauth, league_id):
    team_key_to_manager = {}
    player_id_to_name = {}
    player_id_to_team = {}
    stat_mapping = {}

    # league settings (with retries/backoff)
    root = fetch_url(f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_id}/settings", oauth)
    for stat in root.findall("league/settings/stat_categories/stats/stat"):
        stat_id = stat.findtext("stat_id")
        display_name = stat.findtext("display_name")
        if stat_id and display_name:
            stat_mapping[stat_id] = display_name

    # week-1 rosters for nicknames + some player/team seeds
    for i in range(1, 11):
        try:
            root = fetch_url(
                f"https://fantasysports.yahooapis.com/fantasy/v2/team/{league_id}.t.{i}/roster;week=1/players/stats",
                oauth
            )
        except (APITimeoutError, RecoverableAPIError):
            continue

        nickname = (root.findtext("team/managers/manager/nickname") or "unknown").strip()
        team_key = (root.findtext("team/team_key") or "").strip()
        if team_key:
            team_key_to_manager[team_key] = nickname

        players = root.findall("team/roster/players/player")
        player_ids = [p.findtext("player_id") for p in players if p.find("player_id") is not None]
        names = [p.findtext("name/full") for p in players if p.find("name/full") is not None]
        cleaned_names = [apply_manual_mapping(clean_name(n)) for n in names]
        team_abbrs = [p.findtext("editorial_team_abbr") for p in players if p.find("editorial_team_abbr") is not None]

        for pid, cname, tabbr in zip(player_ids, cleaned_names, team_abbrs):
            if pid and pid not in player_id_to_name:
                player_id_to_name[pid] = cname
                player_id_to_team[pid] = tabbr

    # all players (paginated)
    start = 0
    while True:
        try:
            root = fetch_url(
                f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_id}/players;start={start}",
                oauth
            )
        except (APITimeoutError, RecoverableAPIError):
            break

        players = root.findall("league/players/player")
        if not players:
            break

        for player in players:
            pid = player.findtext("player_id")
            if pid and pid not in player_id_to_name:
                nm = player.findtext("name/full")
                cleaned = apply_manual_mapping(clean_name(nm))
                player_id_to_name[pid] = cleaned
                team_abbr = player.findtext("editorial_team_abbr")
                player_id_to_team[pid] = team_abbr

        start += len(players)

    return team_key_to_manager, player_id_to_name, player_id_to_team, stat_mapping

def fetch_draft_analysis(oauth, league_id, year):
    draft_analysis = []
    start = 0
    while True:
        try:
            root = fetch_url(
                f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_id}/players;start={start}/draft_analysis",
                oauth
            )
        except (APITimeoutError, RecoverableAPIError):
            break

        players = root.findall("league/players/player")
        if not players:
            break

        for player in players:
            name = player.findtext("name/full")
            cleaned_name = apply_manual_mapping(clean_name(name))
            player_data = {
                "year": year,
                "player_key": player.findtext("player_key"),
                "player_id": player.findtext("player_id"),
                "name_full": cleaned_name,
                "primary_position": player.findtext("primary_position"),
                "average_pick": player.findtext("draft_analysis/average_pick"),
                "average_round": player.findtext("draft_analysis/average_round"),
                "average_cost": player.findtext("draft_analysis/average_cost"),
                "percent_drafted": player.findtext("draft_analysis/percent_drafted"),
                "preseason_average_pick": player.findtext("draft_analysis/preseason_average_pick"),
                "preseason_average_round": player.findtext("draft_analysis/preseason_average_round"),
                "preseason_average_cost": player.findtext("draft_analysis/preseason_average_cost"),
                "preseason_percent_drafted": player.findtext("draft_analysis/preseason_percent_drafted"),
                "is_keeper_status": (player.findtext("is_keeper/status") or ""),
                "is_keeper_cost": (player.findtext("is_keeper/cost") or "")
            }
            draft_analysis.append(player_data)

        start += len(players)

    return draft_analysis
