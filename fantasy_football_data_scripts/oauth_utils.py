#!/usr/bin/env python3
"""
OAuth utilities - Supports both yahoo-oauth and manual token management
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional


def find_oauth_file() -> Optional[Path]:
    """
    Find OAuth.json file in priority order:
    1. OAUTH_PATH environment variable (absolute path)
    2. oauth/Oauth.json relative to current working directory
    3. ../../oauth/Oauth.json relative to this script
    4. Search up the directory tree for oauth/Oauth.json
    """
    # Priority 1: Environment variable (set by main.py)
    env_path = os.environ.get("OAUTH_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists():
            return p
        print(f"Warning: OAUTH_PATH set but file not found: {p}")

    # Priority 2: CWD relative
    cwd_oauth = Path.cwd() / "oauth" / "Oauth.json"
    if cwd_oauth.exists():
        return cwd_oauth

    # Priority 3: Script relative (for backward compatibility)
    try:
        script_dir = Path(__file__).resolve().parent
        script_oauth = script_dir / ".." / ".." / "oauth" / "Oauth.json"
        script_oauth = script_oauth.resolve()
        if script_oauth.exists():
            return script_oauth
    except NameError:
        pass

    # Priority 4: Search up the tree
    current = Path.cwd()
    for _ in range(5):  # Search up to 5 levels
        candidate = current / "oauth" / "Oauth.json"
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:  # Reached root
            break
        current = parent

    return None


def create_oauth2(oauth_path: Optional[str | Path] = None):
    """
    Create OAuth2 session for Yahoo Fantasy API.

    Supports both:
    - yahoo-oauth library (Format 1: with consumer_key/consumer_secret)
    - Manual token management (Format 2: generic OAuth2)

    Args:
        oauth_path: Optional path to OAuth.json. If None, will search for it.

    Returns:
        OAuth2 session object compatible with yahoo-fantasy-api

    Raises:
        FileNotFoundError: If OAuth.json cannot be found
        ValueError: If OAuth.json is invalid
    """
    # Find OAuth file
    if oauth_path:
        oauth_file = Path(oauth_path).resolve()
        if not oauth_file.exists():
            raise FileNotFoundError(f"OAuth file not found: {oauth_file}")
    else:
        oauth_file = find_oauth_file()
        if not oauth_file:
            # Provide helpful error message
            searched = []
            if os.environ.get("OAUTH_PATH"):
                searched.append(f"  - OAUTH_PATH env: {os.environ.get('OAUTH_PATH')}")
            searched.append(f"  - CWD: {Path.cwd() / 'oauth' / 'Oauth.json'}")
            try:
                script_dir = Path(__file__).resolve().parent
                searched.append(f"  - Script relative: {script_dir / '..' / '..' / 'oauth' / 'Oauth.json'}")
            except NameError:
                pass

            raise FileNotFoundError(
                "OAuth.json not found. Searched locations:\n" + "\n".join(searched) +
                "\n\nPlease ensure you've authenticated and the OAuth.json file exists."
            )

    print(f"Using OAuth file: {oauth_file}")

    # Load OAuth data
    with open(oauth_file, 'r') as f:
        oauth_data = json.load(f)

    # Try yahoo-oauth library first (Format 1)
    if 'consumer_key' in oauth_data or 'consumer_secret' in oauth_data:
        try:
            from yahoo_oauth import OAuth2
            oauth = OAuth2(None, None, from_file=str(oauth_file))
            return oauth
        except Exception as e:
            print(f"Warning: yahoo-oauth library failed: {e}")

    # Fall back to manual token management (Format 2)
    # Create a minimal OAuth2-compatible object
    class ManualOAuth2:
        def __init__(self, token_data):
            self.access_token = token_data.get('access_token')
            self.refresh_token = token_data.get('refresh_token')
            self.token_time = token_data.get('token_time', 0)
            self.token_type = token_data.get('token_type', 'bearer')

            if not self.access_token:
                raise ValueError("No access_token found in OAuth.json")

            # Create a session-like object
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.access_token}'
            })

        def refresh_access_token(self):
            """Placeholder - refresh logic would go here"""
            print("Warning: Token refresh not implemented for manual OAuth")
            return False

    return ManualOAuth2(oauth_data)