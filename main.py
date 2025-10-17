#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import io
import json
import base64
import zipfile
import urllib.parse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st

# =====================================================================================
# Config & Secrets
# =====================================================================================

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = (APP_ROOT / "fantasy_football_data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load client secrets (from env first, then st.secrets)
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID") or (st.secrets.get("YAHOO_CLIENT_ID") if hasattr(st, "secrets") else None)
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET") or (st.secrets.get("YAHOO_CLIENT_SECRET") if hasattr(st, "secrets") else None)
REDIRECT_URI = os.environ.get("YAHOO_REDIRECT_URI") or (st.secrets.get("YAHOO_REDIRECT_URI") if hasattr(st, "secrets") else "https://localhost/")

# MotherDuck token (NO UI! only env or secrets)
MD_TOKEN = os.environ.get("MOTHERDUCK_TOKEN") or (st.secrets.get("MOTHERDUCK_TOKEN") if hasattr(st, "secrets") else None)

# Yahoo OAuth endpoints
AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

# Yahoo Fantasy endpoints
YF_BASE = "https://fantasysports.yahooapis.com/fantasy/v2"

# =====================================================================================
# Helpers
# =====================================================================================

def _slug_db_name(name: str) -> str:
    """Create a safe DB name from league name for MD."""
    s = (name or "leaguehistory").lower()
    for ch in " -./\\:()[]{}'\"":
        s = s.replace(ch, "_")
    s = "_".join(filter(None, s.split("_")))
    return s[:63]

def _save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _load_json(path: Path) -> Optional[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _dump_json(path: Path, obj: dict) -> None:
    _save_text(path, json.dumps(obj, indent=2, ensure_ascii=False))

def _now_utc() -> datetime:
    return datetime.utcnow()

def _to_epoch(dt: datetime) -> int:
    return int(dt.timestamp())

# =====================================================================================
# Yahoo OAuth
# =====================================================================================

def build_auth_url(state: Optional[str] = None, scope: str = "fspt-w"):
    assert CLIENT_ID and REDIRECT_URI, "Missing CLIENT_ID or REDIRECT_URI"
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "language": "en-us",
        "scope": scope
    }
    if state:
        params["state"] = state
    return AUTH_URL + "?" + urllib.parse.urlencode(params)

def exchange_code_for_token(code: str) -> dict:
    assert CLIENT_ID and CLIENT_SECRET and REDIRECT_URI, "Missing Yahoo OAuth config"
    auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
        "code": code
    }
    resp = requests.post(TOKEN_URL, headers=headers, data=data, timeout=60)
    resp.raise_for_status()
    return resp.json()

def refresh_access_token(refresh_token: str) -> dict:
    assert CLIENT_ID and CLIENT_SECRET and REDIRECT_URI, "Missing Yahoo OAuth config"
    auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "refresh_token",
        "redirect_uri": REDIRECT_URI,
        "refresh_token": refresh_token
    }
    resp = requests.post(TOKEN_URL, headers=headers, data=data, timeout=60)
    resp.raise_for_status()
    return resp.json()

def token_expiry_time(token_data: dict) -> datetime:
    # Yahoo returns "expires_in" seconds; compute wall time
    issued = st.session_state.get("token_issued_at") or _now_utc()
    return issued + timedelta(seconds=int(token_data.get("expires_in", 3600)))

def token_is_expired(token_data: dict) -> bool:
    expires_at = token_expiry_time(token_data)
    # refresh a bit early
    return _now_utc() >= (expires_at - timedelta(seconds=60))

def ensure_access_token(token_data: dict) -> dict:
    """Ensure token is valid. If expired and we have refresh_token, refresh it."""
    if not token_data:
        raise RuntimeError("Missing token_data")
    if not token_is_expired(token_data):
        return token_data
    rt = token_data.get("refresh_token")
    if not rt:
        raise RuntimeError("Token expired and refresh_token not available.")
    new_tok = refresh_access_token(rt)
    # carry forward refresh_token if not sent again
    if "refresh_token" not in new_tok and "refresh_token" in token_data:
        new_tok["refresh_token"] = token_data["refresh_token"]
    st.session_state["token_data"] = new_tok
    st.session_state["token_issued_at"] = _now_utc()
    return new_tok

# =====================================================================================
# Yahoo Fantasy API utilities
# =====================================================================================

def yahoo_api_call(access_token: str, path: str, params: dict | None = None) -> requests.Response:
    url = f"{YF_BASE}/{path}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    return requests.get(url, headers=headers, params=params, timeout=60)

def fetch_user_guid(access_token: str) -> str:
    resp = yahoo_api_call(access_token, "users;use_login=1/games?format=json")
    resp.raise_for_status()
    data = resp.json()
    # Grab the user's GUID safely
    users = data.get("fantasy_content", {}).get("users", {})
    # The Yahoo JSON is nested; try a few paths
    # If not found, raise helpful error.
    for k, v in users.items():
        if isinstance(v, dict) and "user" in v:
            user = v["user"][0]
            guid = user[0].get("guid")
            if guid:
                return guid
    raise RuntimeError("Unable to determine Yahoo user GUID from response.")

def fetch_user_leagues(access_token: str, season: Optional[int] = None) -> list[dict]:
    # When season is not provided, Yahoo returns all current games/leagues
    # The structure is nested; we’ll parse for fantasy leagues.
    resp = yahoo_api_call(access_token, "users;use_login=1/games;game_keys=nfl/leagues?format=json")
    resp.raise_for_status()
    data = resp.json()
    leagues_out: list[dict] = []

    games = data.get("fantasy_content", {}).get("users", {}).get("0", {}).get("user", [None, {}])[1].get("games", {})
    # Walk the nested JSON carefully
    for _, game_obj in games.items():
        if not isinstance(game_obj, dict) or "game" not in game_obj:
            continue
        game = game_obj["game"][0]
        game_key = game.get("game_key")
        game_code = game.get("code")
        game_season = game.get("season")
        leagues = game_obj["game"][1].get("leagues", {})
        if not isinstance(leagues, dict):
            continue
        for _, lg in leagues.items():
            if not isinstance(lg, dict) or "league" not in lg:
                continue
            league_node = lg["league"][0]
            league_key = league_node.get("league_key")
            name = league_node.get("name")
            num_teams = league_node.get("num_teams")
            # filter by requested season if provided
            if season and str(game_season) != str(season):
                continue
            leagues_out.append({
                "name": name,
                "league_key": league_key,
                "game_key": game_key,
                "season": game_season,
                "game_code": game_code,
                "num_teams": num_teams,
            })
    return leagues_out

# =====================================================================================
# Persist token & league selection
# =====================================================================================

OAUTH_STORE = APP_ROOT / ".oauth" / "yahoo_token.json"
LEAGUE_STORE = APP_ROOT / ".oauth" / "league_selection.json"

def save_oauth_token(token_data: dict, selected_league: dict | None) -> Path:
    OAUTH_STORE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_data": {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "token_type": token_data.get("token_type"),
            "expires_in": token_data.get("expires_in"),
        },
        "issued_at": _to_epoch(st.session_state.get("token_issued_at") or _now_utc()),
        "selected_league": selected_league or {},
    }
    _dump_json(OAUTH_STORE, payload)
    return OAUTH_STORE

def load_saved_state() -> tuple[dict | None, dict | None]:
    token_blob = _load_json(OAUTH_STORE)
    league_blob = _load_json(LEAGUE_STORE)
    token_data = token_blob.get("token_data") if token_blob else None
    if token_blob and "issued_at" in token_blob:
        st.session_state["token_issued_at"] = datetime.utcfromtimestamp(token_blob["issued_at"])
    selected_league = league_blob.get("league") if league_blob else None
    return token_data, selected_league

def save_league_selection(league: dict) -> None:
    LEAGUE_STORE.parent.mkdir(parents=True, exist_ok=True)
    _dump_json(LEAGUE_STORE, {"league": league})

# =====================================================================================
# Initial Import Runner
# =====================================================================================

def run_initial_import() -> bool:
    """
    Runs initial_import.py as a subprocess so it can use the env vars we've set.
    Returns True on success.
    """
    script = APP_ROOT / "initial_import.py"
    if not script.exists():
        st.error("initial_import.py not found next to main.py")
        return False

    env = os.environ.copy()
    # Respect MD env already configured above.
    cmd = [sys.executable, str(script)]
    st.info("Starting initial data import…")
    try:
        out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        st.success("Initial import completed.")
        with st.expander("Import logs"):
            st.code(out.stdout or "(no stdout)")
            if out.stderr:
                st.code(out.stderr)
        return True
    except subprocess.CalledProcessError as e:
        st.error("Initial import failed.")
        with st.expander("Import logs (error)"):
            st.code(e.stdout or "(no stdout)")
            st.code(e.stderr or "(no stderr)")
        return False

# =====================================================================================
# UI
# =====================================================================================

st.set_page_config(page_title="KMFFL Stats", page_icon="🦆", layout="wide")

st.title("KMFFL Stats")
st.caption("Yahoo + NFL merged stats • Automatic local/MD storage • No token prompts")

# Sidebar status
with st.sidebar:
    st.header("Status")
    if MD_TOKEN:
        st.success("🦆 MotherDuck detected: will write directly to your personal MD.")
    else:
        st.info("No MotherDuck token configured; data will be saved locally.")

# Load any saved state on boot
if "boot_loaded" not in st.session_state:
    token_data, selected_league = load_saved_state()
    if token_data:
        st.session_state["token_data"] = token_data
        if "token_issued_at" not in st.session_state:
            st.session_state["token_issued_at"] = _now_utc()
    if selected_league:
        st.session_state["selected_league"] = selected_league
    st.session_state["boot_loaded"] = True

# Step 1: OAuth
st.header("1) Connect Yahoo")
if not CLIENT_ID or not CLIENT_SECRET:
    st.error("Missing Yahoo CLIENT_ID / CLIENT_SECRET. Set env vars or st.secrets.")
else:
    # If we have a code param in the URL (Streamlit Cloud sometimes provides via query params)
    qp = st.query_params
    auth_code = qp.get("code", [None])[0] if isinstance(qp.get("code"), list) else qp.get("code", None)

    if "token_data" not in st.session_state:
        if auth_code:
            try:
                tok = exchange_code_for_token(auth_code)
                st.session_state["token_data"] = tok
                st.session_state["token_issued_at"] = _now_utc()
                st.success("Yahoo connected!")
                # Clear code from URL
                st.query_params.clear()
            except Exception as e:
                st.error(f"OAuth exchange failed: {e}")
        else:
            auth_link = build_auth_url()
            st.markdown(
                f"[Click here to authorize Yahoo Fantasy access]({auth_link})",
                unsafe_allow_html=True
            )
    else:
        # Ensure the token is fresh
        try:
            st.session_state["token_data"] = ensure_access_token(st.session_state["token_data"])
            st.success("Yahoo connected (token valid).")
        except Exception as e:
            st.warning(f"Token invalid/expired: {e}")
            st.session_state.pop("token_data", None)
            st.rerun()

# Step 2: Choose League
st.header("2) Select League")
if "token_data" in st.session_state:
    tok = st.session_state["token_data"]
    access_token = tok.get("access_token")
    try:
        leagues = fetch_user_leagues(access_token)
    except Exception as e:
        st.error(f"Failed to list leagues: {e}")
        leagues = []

    if leagues:
        # Display leagues in a selectbox by name (season)
        display_names = [f"{lg['name']} (season {lg['season']})" for lg in leagues]
        default_idx = 0

        # If already selected before, pre-select
        pre_sel = st.session_state.get("selected_league")
        if pre_sel:
            try:
                default_idx = next(i for i, lg in enumerate(leagues) if lg["league_key"] == pre_sel.get("league_key"))
            except StopIteration:
                default_idx = 0

        chosen = st.selectbox("Choose your league", display_names, index=default_idx)
        if chosen:
            idx = display_names.index(chosen)
            selected_league = leagues[idx]
            st.session_state["selected_league"] = selected_league
            save_league_selection(selected_league)

            st.success(f"Selected league: {selected_league['name']} ({selected_league['season']})")

# Step 3: Persist OAuth + Import
st.header("3) Save & Import")
if "token_data" in st.session_state and "selected_league" in st.session_state:
    selected_league = st.session_state["selected_league"]

    # Compute DB name from league for MD
    league_name_slug = _slug_db_name(selected_league["name"])
    md_db = league_name_slug

    # Configure env (no UI prompts)
    if MD_TOKEN:
        os.environ["MOTHERDUCK_TOKEN"] = MD_TOKEN
        os.environ["MD_DIRECT_UPLOAD"] = "1"
        os.environ["MD_DB"] = md_db
    else:
        os.environ.pop("MOTHERDUCK_TOKEN", None)
        os.environ.pop("MD_DIRECT_UPLOAD", None)
        os.environ.pop("MD_DB", None)

    # Save token + league
    oauth_file = save_oauth_token(st.session_state["token_data"], selected_league)
    st.caption(f"OAuth saved to: `{oauth_file}`")

    # Buttons
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        if st.button("Run Initial Import", type="primary"):
            run_initial_import()
    with col2:
        if st.button("Re-run Import (Overwrite)"):
            run_initial_import()

# Optional: Helper section
with st.expander("ℹ️ What gets imported?"):
    st.write(
        """
        - Your Yahoo league data is fetched with your OAuth token.
        - The local scripts merge/clean stats.
        - If a MotherDuck token is configured in the environment or `st.secrets`, 
          results are written directly to your personal MotherDuck database named after the league (slug).
        - Otherwise, results are saved locally.
        """
    )

# Footer
st.write("---")
st.caption("KMFFL Stats • Built with Streamlit")
