#!/usr/bin/env python3
"""
OAuth utilities for Yahoo Fantasy Football scripts.
Provides robust OAuth file discovery across different environments.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2


def _resolve_oauth_path(hint: str | None = None) -> Path:
    """
    Find OAuth.json file using multiple fallback strategies.

    Priority:
    1. OAUTH_PATH environment variable
    2. hint parameter (if provided)
    3. Current working directory
    4. Repository root (detected via common markers)
    5. Common relative paths from script location
    """
    candidates = []

    # Priority 1: Environment variable
    if os.environ.get("OAUTH_PATH"):
        candidates.append(Path(os.environ["OAUTH_PATH"]))

    # Priority 2: Explicit hint
    if hint:
        candidates.append(Path(hint))

    # Priority 3: CWD and its parent hierarchy
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents)[:3]:
        candidates.append(parent / "oauth" / "Oauth.json")
        candidates.append(parent / "Oauth.json")

    # Priority 4: Detect repo root via common markers
    try:
        # Look for .git, main.py, or requirements.txt as repo indicators
        for parent in [cwd] + list(cwd.parents)[:5]:
            if any((parent / marker).exists() for marker in [".git", "main.py", "requirements.txt"]):
                candidates.append(parent / "oauth" / "Oauth.json")
                break
    except Exception:
        pass

    # Priority 5: Relative to this script file
    try:
        this_file = Path(__file__).resolve()
        script_dir = this_file.parent
        for i in range(3):  # Check up to 3 levels up
            candidates.append(script_dir / "oauth" / "Oauth.json")
            candidates.append(script_dir / "Oauth.json")
            script_dir = script_dir.parent
    except Exception:
        pass

    # Priority 6: Common Streamlit Cloud paths
    for base in ["/mount/src", "/workspace", "/app"]:
        try:
            base_path = Path(base)
            if base_path.exists():
                # Look for repo-like subdirectories
                for subdir in base_path.iterdir():
                    if subdir.is_dir():
                        candidates.append(subdir / "oauth" / "Oauth.json")
        except Exception:
            pass

    # Return first existing candidate
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        except Exception:
            continue

    # If nothing found, return the most likely default
    default = cwd / "oauth" / "Oauth.json"
    return default


def create_oauth2(oauth_path: str | Path | None = None) -> OAuth2:
    """
    Create OAuth2 object with robust path resolution.

    Args:
        oauth_path: Optional explicit path to OAuth.json

    Returns:
        Configured OAuth2 object

    Raises:
        FileNotFoundError: If OAuth.json cannot be found
        ValueError: If OAuth.json is malformed
    """
    resolved_path = _resolve_oauth_path(str(oauth_path) if oauth_path else None)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"OAuth.json not found at {resolved_path}. "
            f"Checked locations:\n"
            f"  - OAUTH_PATH env: {os.environ.get('OAUTH_PATH', '(not set)')}\n"
            f"  - CWD: {Path.cwd()}\n"
            f"  - Resolved: {resolved_path}"
        )

    # Validate OAuth file format
    try:
        with open(resolved_path, 'r') as f:
            data = json.load(f)

        # Support both Format 1 (token_data wrapper) and Format 2 (flat)
        if "token_data" in data:
            # Format 1: {"token_data": {...}}
            token_data = data["token_data"]
        else:
            # Format 2: {...} (flat)
            token_data = data

        # Ensure required keys exist
        required_keys = ["access_token", "refresh_token", "consumer_key", "consumer_secret"]
        missing = [k for k in required_keys if k not in token_data]
        if missing:
            raise ValueError(f"OAuth.json missing required keys: {missing}")

        # Check if the file is already in yahoo-oauth format (has top-level access_token)
        # If so, use it directly. Otherwise, rewrite it to the expected format.
        if "access_token" not in data:
            # Need to flatten the structure for yahoo-oauth
            temp_path = resolved_path.parent / f".{resolved_path.name}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(token_data, f, indent=2)
            temp_path.replace(resolved_path)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {resolved_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading {resolved_path}: {e}")

    # Create OAuth2 object - yahoo-oauth expects the file path, not the data
    # and it will read consumer_key/consumer_secret from the file
    try:
        oauth = OAuth2(None, None, from_file=str(resolved_path))
        return oauth
    except AttributeError as e:
        # If yahoo-oauth can't find consumer_key, the file format is wrong
        # Try reading and reconstructing
        with open(resolved_path, 'r') as f:
            file_data = json.load(f)

        # Extract token_data if wrapped
        if "token_data" in file_data:
            token_info = file_data["token_data"]
        else:
            token_info = file_data

        # Ensure it has all required fields at the top level
        if "consumer_key" in token_info and "consumer_secret" in token_info:
            # Rewrite file with correct structure
            with open(resolved_path, 'w') as f:
                json.dump(token_info, f, indent=2)

            # Try again
            oauth = OAuth2(None, None, from_file=str(resolved_path))
            return oauth
        else:
            raise ValueError(f"OAuth.json missing consumer_key or consumer_secret: {e}")


def get_oauth_data(oauth_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and return OAuth data as a dictionary.

    Args:
        oauth_path: Optional explicit path to OAuth.json

    Returns:
        Dictionary containing OAuth credentials
    """
    resolved_path = _resolve_oauth_path(str(oauth_path) if oauth_path else None)

    if not resolved_path.exists():
        raise FileNotFoundError(f"OAuth.json not found at {resolved_path}")

    with open(resolved_path, 'r') as f:
        data = json.load(f)

    # Normalize to Format 1 (with token_data wrapper)
    if "token_data" in data:
        return data
    else:
        return {"token_data": data}


# Convenience function for quick testing
def test_oauth_resolution():
    """Test OAuth file resolution and print diagnostic info"""
    print("OAuth Resolution Diagnostic")
    print("=" * 60)
    print(f"CWD: {Path.cwd()}")
    print(f"OAUTH_PATH env: {os.environ.get('OAUTH_PATH', '(not set)')}")
    print()

    try:
        resolved = _resolve_oauth_path()
        print(f"✅ Resolved path: {resolved}")
        print(f"   Exists: {resolved.exists()}")

        if resolved.exists():
            with open(resolved, 'r') as f:
                data = json.load(f)
            print(f"   Format: {'Format 1 (wrapped)' if 'token_data' in data else 'Format 2 (flat)'}")

            token_data = data.get("token_data", data)
            has_keys = all(
                k in token_data for k in ["access_token", "refresh_token", "consumer_key", "consumer_secret"])
            print(f"   Valid: {has_keys}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_oauth_resolution()