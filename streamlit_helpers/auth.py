#!/usr/bin/env python3
"""
OAuth Authentication Helpers for Yahoo Fantasy API

This module handles all OAuth-related operations including:
- Building authorization URLs
- Exchanging codes for tokens
- Saving tokens to files and MotherDuck
"""

import os
import base64
import json
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests

# OAuth Configuration
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID")
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("REDIRECT_URI", "https://leaguehistory.streamlit.app")

AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

# Try to get from streamlit secrets as fallback
try:
    import streamlit as st
    if not CLIENT_ID:
        CLIENT_ID = st.secrets.get("YAHOO_CLIENT_ID", None)
    if not CLIENT_SECRET:
        CLIENT_SECRET = st.secrets.get("YAHOO_CLIENT_SECRET", None)
except Exception:
    # Catch all exceptions including StreamlitSecretNotFoundError
    pass

# Directory for OAuth files
ROOT_DIR = Path(__file__).parent.parent.resolve()
OAUTH_DIR = ROOT_DIR / "oauth"


def get_auth_header() -> str:
    """Generate Basic auth header for Yahoo OAuth."""
    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


def build_authorize_url(state: str | None = None) -> str:
    """Build Yahoo OAuth authorization URL."""
    params = {"client_id": CLIENT_ID, "redirect_uri": REDIRECT_URI, "response_type": "code"}
    if state:
        params["state"] = state
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for access/refresh tokens."""
    headers = {"Authorization": get_auth_header(), "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "authorization_code", "redirect_uri": REDIRECT_URI, "code": code}
    resp = requests.post(TOKEN_URL, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()


def save_oauth_token(token_data: dict, league_info: dict | None = None) -> Path:
    """
    Save OAuth token to a JSON file for the league import scripts.

    The file is saved to oauth/Oauth_{league_name}.json or oauth/Oauth.json
    Returns the path to the saved file.
    """
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)

    # Determine filename based on league info
    if league_info and league_info.get("name"):
        safe_name = league_info["name"].replace(" ", "_").replace("-", "_").lower()
        oauth_filename = f"Oauth_{safe_name}.json"
    else:
        oauth_filename = "Oauth.json"

    oauth_path = OAUTH_DIR / oauth_filename

    # Add metadata to token
    token_with_meta = {
        **token_data,
        "consumer_key": CLIENT_ID,
        "consumer_secret": CLIENT_SECRET,
        "token_time": datetime.now(timezone.utc).isoformat(),
    }

    if league_info:
        token_with_meta["league_info"] = league_info

    # Write to file
    with open(oauth_path, "w") as f:
        json.dump(token_with_meta, f, indent=2)

    return oauth_path


def save_token_to_motherduck(token_data: dict, league_info: Optional[dict] = None) -> Optional[str]:
    """
    Save OAuth token to MotherDuck ops.oauth_tokens table.

    This allows GitHub Actions to retrieve tokens for scheduled imports.
    Returns the token_id if successful, None otherwise.
    """
    motherduck_token = os.environ.get("MOTHERDUCK_TOKEN")
    if not motherduck_token:
        try:
            import streamlit as st
            motherduck_token = st.secrets.get("MOTHERDUCK_TOKEN", "")
        except Exception:
            # Catch all exceptions including StreamlitSecretNotFoundError
            pass

    if not motherduck_token:
        return None

    try:
        import duckdb
        con = duckdb.connect("md:")

        # Create ops schema and table if they don't exist
        con.execute("CREATE SCHEMA IF NOT EXISTS ops")
        con.execute("""
            CREATE TABLE IF NOT EXISTS ops.oauth_tokens (
                token_id TEXT PRIMARY KEY,
                league_name TEXT,
                league_key TEXT,
                access_token TEXT,
                refresh_token TEXT,
                token_type TEXT,
                expires_in INTEGER,
                consumer_key TEXT,
                consumer_secret TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        # Generate token_id
        import uuid
        token_id = str(uuid.uuid4())[:8]

        now = datetime.now(timezone.utc)

        con.execute("""
            INSERT INTO ops.oauth_tokens
            (token_id, league_name, league_key, access_token, refresh_token,
             token_type, expires_in, consumer_key, consumer_secret, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            token_id,
            league_info.get("name") if league_info else None,
            league_info.get("league_key") if league_info else None,
            token_data.get("access_token"),
            token_data.get("refresh_token"),
            token_data.get("token_type"),
            token_data.get("expires_in"),
            CLIENT_ID,
            CLIENT_SECRET,
            now,
            now
        ])

        con.close()
        return token_id

    except Exception as e:
        print(f"[AUTH] Warning: Could not save token to MotherDuck: {e}")
        return None
