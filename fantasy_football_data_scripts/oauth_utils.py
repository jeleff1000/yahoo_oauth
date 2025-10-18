#!/usr/bin/env python3
"""
OAuth Utility Module

Handles loading OAuth credentials from multiple JSON formats:
- Format 1 (legacy): Flat structure with token fields at root
- Format 2 (new): Nested structure with token_data object and optional league_info

This module provides a unified interface for all scripts to access OAuth credentials.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from yahoo_oauth import OAuth2


def _resolve_oauth_path(candidate: str | Path | None) -> Path:
    """
    Resolve an oauth path robustly:
      1) If candidate provided and exists, use it
      2) If OAUTH_PATH env var set and exists, use it
      3) If repo-local oauth/Oauth.json exists (two levels up), use that
      4) Otherwise return the candidate Path (may not exist)
    """
    # Normalize
    if candidate:
        p = Path(candidate)
        if p.exists():
            return p.resolve()
    # Env override
    env_p = os.environ.get("OAUTH_PATH")
    if env_p:
        p = Path(env_p)
        if p.exists():
            return p.resolve()
    # Repo-local fallback (this file lives in fantasy_football_data_scripts)
    try:
        repo_root = Path(__file__).resolve().parents[1]
        repo_oauth = repo_root / "oauth" / "Oauth.json"
        if repo_oauth.exists():
            return repo_oauth.resolve()
    except Exception:
        pass
    # Last resort: return Path(candidate or '')
    return Path(candidate or '')


def load_oauth_json(oauth_path: str | Path) -> Dict[str, Any]:
    """
    Load OAuth JSON and normalize to Format 1 structure.

    Supports two formats:

    Format 1 (legacy/flat):
    {
      "access_token": "...",
      "refresh_token": "...",
      "consumer_key": "...",
      "consumer_secret": "...",
      "token_type": "bearer",
      "token_time": 1234567890.123,
      "guid": null
    }

    Format 2 (new/nested, without email):
    {
      "league_info": {
        "league_key": "331.l.492605",
        "name": "League Name",
        "num_teams": 10,
        "season": "2024"
      },
      "token_data": {
        "access_token": "...",
        "refresh_token": "...",
        "consumer_key": "...",
        "consumer_secret": "...",
        "token_type": "bearer",
        "expires_in": 3600,
        "token_time": 1234567890.123,
        "guid": null
      },
      "timestamp": "2025-10-15T18:57:57.398804"
    }

    Returns:
        Dict with Format 1 structure (flat) for compatibility with OAuth2
    """
    oauth_path = _resolve_oauth_path(oauth_path)

    if not oauth_path or not oauth_path.exists():
        raise FileNotFoundError(f"OAuth file not found: {oauth_path}")

    with open(oauth_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if it's Format 2 (has token_data key)
    if "token_data" in data:
        # Extract token_data and flatten to Format 1
        token_data = data["token_data"]

        # Return flattened structure
        return {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "consumer_key": token_data.get("consumer_key"),
            "consumer_secret": token_data.get("consumer_secret"),
            "token_type": token_data.get("token_type", "bearer"),
            "token_time": token_data.get("token_time"),
            "guid": token_data.get("guid"),
        }

    # Format 1 - already flat, return as-is
    return data


def get_league_info(oauth_path: str | Path) -> Optional[Dict[str, Any]]:
    """
    Extract league_info from OAuth JSON if present (Format 2 only).

    Returns:
        Dict with league_info or None if not present
    """
    oauth_path = _resolve_oauth_path(oauth_path)

    if not oauth_path or not oauth_path.exists():
        return None

    with open(oauth_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("league_info")


def create_oauth2(oauth_path: str | Path) -> OAuth2:
    """
    Create OAuth2 instance from either format of OAuth JSON.

    Args:
        oauth_path: Path to Oauth.json file

    Returns:
        Configured OAuth2 instance
    """
    oauth_path_resolved = _resolve_oauth_path(oauth_path)
    if not oauth_path_resolved or not oauth_path_resolved.exists():
        raise FileNotFoundError(f"OAuth file not found (after fallback attempts): {oauth_path_resolved}")

    oauth_data = load_oauth_json(oauth_path_resolved)
    oauth = OAuth2(None, None, from_file=str(oauth_path_resolved))

    # Refresh token if needed
    if not oauth.token_is_valid():
        oauth.refresh_access_token()

    return oauth


def save_oauth_format2(
    oauth_path: str | Path,
    token_data: Dict[str, Any],
    league_info: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None
) -> None:
    """
    Save OAuth data in Format 2 (nested structure).

    Args:
        oauth_path: Path to save Oauth.json
        token_data: Token credentials dict
        league_info: Optional league information
        timestamp: Optional timestamp string
    """
    from datetime import datetime

    oauth_path = Path(oauth_path)

    from typing import Dict
    data: Dict[str, Any] = {
        "token_data": token_data
    }

    if league_info:
        data["league_info"] = league_info

    if timestamp:
        data["timestamp"] = timestamp
    else:
        data["timestamp"] = datetime.now().isoformat()

    with open(oauth_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def migrate_format1_to_format2(
    oauth_path: str | Path,
    league_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Migrate OAuth file from Format 1 to Format 2.

    Args:
        oauth_path: Path to Oauth.json file
        league_info: Optional league information to add
    """
    from datetime import datetime

    oauth_path = Path(oauth_path)
    oauth_data = load_oauth_json(oauth_path)

    # Create Format 2 structure
    save_oauth_format2(
        oauth_path,
        token_data=oauth_data,
        league_info=league_info,
        timestamp=datetime.now().isoformat()
    )
