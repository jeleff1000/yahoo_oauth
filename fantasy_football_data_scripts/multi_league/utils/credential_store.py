#!/usr/bin/env python3
"""
Credential Store - Secure storage and retrieval of per-league OAuth credentials.

Stores encrypted refresh tokens in MotherDuck so weekly updates can run
automatically without user intervention.

Security notes:
- Refresh tokens are encrypted using Fernet (AES-128-CBC)
- Encryption key is stored as a GitHub secret (CREDENTIAL_ENCRYPTION_KEY)
- Only refresh_token is stored (not access_token, which expires quickly)
- Consumer key/secret come from GitHub secrets, not stored per-league
"""

import os
import json
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Try to import cryptography for encryption
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


def get_encryption_key() -> Optional[bytes]:
    """
    Get the encryption key from environment variable.

    The key should be a Fernet-compatible key (32 url-safe base64-encoded bytes).
    Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

    Store as GitHub secret: CREDENTIAL_ENCRYPTION_KEY
    """
    key = os.environ.get("CREDENTIAL_ENCRYPTION_KEY")
    if key:
        return key.encode() if isinstance(key, str) else key
    return None


def encrypt_token(token: str, key: bytes = None) -> Optional[str]:
    """
    Encrypt a token for storage.

    Returns base64-encoded encrypted string, or None if encryption unavailable.
    """
    if not CRYPTO_AVAILABLE:
        print("[WARN] cryptography not installed, cannot encrypt token")
        return None

    key = key or get_encryption_key()
    if not key:
        print("[WARN] CREDENTIAL_ENCRYPTION_KEY not set, cannot encrypt")
        return None

    try:
        f = Fernet(key)
        encrypted = f.encrypt(token.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    except Exception as e:
        print(f"[ERROR] Encryption failed: {e}")
        return None


def decrypt_token(encrypted: str, key: bytes = None) -> Optional[str]:
    """
    Decrypt a stored token.

    Returns decrypted token string, or None if decryption fails.
    """
    if not CRYPTO_AVAILABLE:
        print("[WARN] cryptography not installed, cannot decrypt token")
        return None

    key = key or get_encryption_key()
    if not key:
        print("[WARN] CREDENTIAL_ENCRYPTION_KEY not set, cannot decrypt")
        return None

    try:
        f = Fernet(key)
        encrypted_bytes = base64.urlsafe_b64decode(encrypted.encode())
        decrypted = f.decrypt(encrypted_bytes)
        return decrypted.decode()
    except Exception as e:
        print(f"[ERROR] Decryption failed: {e}")
        return None


def store_league_credentials(
    database_name: str,
    refresh_token: str,
    league_id: str,
    league_name: str,
    motherduck_token: str = None
) -> bool:
    """
    Store encrypted credentials for a league in MotherDuck.

    Creates a 'league_credentials' table if it doesn't exist.

    Args:
        database_name: MotherDuck database name
        refresh_token: Yahoo OAuth refresh token
        league_id: Yahoo league ID
        league_name: Human-readable league name
        motherduck_token: MotherDuck auth token (defaults to env var)

    Returns:
        True if successful, False otherwise
    """
    import duckdb

    motherduck_token = motherduck_token or os.environ.get("MOTHERDUCK_TOKEN")
    if not motherduck_token:
        print("[ERROR] MOTHERDUCK_TOKEN not set")
        return False

    # Encrypt the refresh token
    encrypted_token = encrypt_token(refresh_token)
    if not encrypted_token:
        print("[ERROR] Failed to encrypt refresh token")
        return False

    try:
        conn = duckdb.connect(f"md:{database_name}?motherduck_token={motherduck_token}")

        # Create credentials table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS league_credentials (
                league_id VARCHAR PRIMARY KEY,
                league_name VARCHAR,
                encrypted_refresh_token VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        # Upsert credentials
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO league_credentials (league_id, league_name, encrypted_refresh_token, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (league_id) DO UPDATE SET
                encrypted_refresh_token = EXCLUDED.encrypted_refresh_token,
                updated_at = EXCLUDED.updated_at
        """, [league_id, league_name, encrypted_token, now, now])

        conn.close()
        print(f"[OK] Stored credentials for {league_name} ({league_id})")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to store credentials: {e}")
        return False


def retrieve_league_credentials(
    database_name: str,
    league_id: str = None,
    motherduck_token: str = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieve credentials for a league from MotherDuck.

    Args:
        database_name: MotherDuck database name
        league_id: Specific league ID (optional, gets first if not specified)
        motherduck_token: MotherDuck auth token (defaults to env var)

    Returns:
        Dict with 'refresh_token', 'league_id', 'league_name' or None
    """
    import duckdb

    motherduck_token = motherduck_token or os.environ.get("MOTHERDUCK_TOKEN")
    if not motherduck_token:
        print("[ERROR] MOTHERDUCK_TOKEN not set")
        return None

    try:
        conn = duckdb.connect(f"md:{database_name}?motherduck_token={motherduck_token}")

        # Check if credentials table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]

        if 'league_credentials' not in table_names:
            print(f"[WARN] No league_credentials table in {database_name}")
            conn.close()
            return None

        # Get credentials
        if league_id:
            result = conn.execute("""
                SELECT league_id, league_name, encrypted_refresh_token
                FROM league_credentials
                WHERE league_id = ?
            """, [league_id]).fetchone()
        else:
            result = conn.execute("""
                SELECT league_id, league_name, encrypted_refresh_token
                FROM league_credentials
                ORDER BY updated_at DESC
                LIMIT 1
            """).fetchone()

        conn.close()

        if not result:
            print(f"[WARN] No credentials found for league_id={league_id}")
            return None

        league_id, league_name, encrypted_token = result

        # Decrypt the token
        refresh_token = decrypt_token(encrypted_token)
        if not refresh_token:
            print("[ERROR] Failed to decrypt refresh token")
            return None

        return {
            'league_id': league_id,
            'league_name': league_name,
            'refresh_token': refresh_token
        }

    except Exception as e:
        print(f"[ERROR] Failed to retrieve credentials: {e}")
        return None


def create_oauth_file_from_stored(
    database_name: str,
    output_path: str = "oauth/Oauth.json",
    league_id: str = None
) -> bool:
    """
    Convenience function: retrieve stored credentials and write OAuth file.

    Args:
        database_name: MotherDuck database name
        output_path: Where to write the OAuth JSON file
        league_id: Specific league ID (optional)

    Returns:
        True if OAuth file was created successfully
    """
    creds = retrieve_league_credentials(database_name, league_id)
    if not creds:
        return False

    # Get consumer credentials from environment
    consumer_key = os.environ.get("YAHOO_CLIENT_ID")
    consumer_secret = os.environ.get("YAHOO_CLIENT_SECRET")

    if not consumer_key or not consumer_secret:
        print("[WARN] YAHOO_CLIENT_ID or YAHOO_CLIENT_SECRET not set")

    oauth_data = {
        'consumer_key': consumer_key,
        'consumer_secret': consumer_secret,
        'refresh_token': creds['refresh_token'],
        # Force token refresh on first use
        'token_time': 0,
    }

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(oauth_data, f, indent=2)

    print(f"[OK] Created OAuth file at {output_path}")
    return True


if __name__ == "__main__":
    # CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="Credential Store CLI")
    parser.add_argument("action", choices=["store", "retrieve", "create-oauth"])
    parser.add_argument("--database", required=True, help="MotherDuck database name")
    parser.add_argument("--league-id", help="League ID")
    parser.add_argument("--league-name", help="League name (for store)")
    parser.add_argument("--refresh-token", help="Refresh token (for store)")
    parser.add_argument("--output", default="oauth/Oauth.json", help="Output path (for create-oauth)")

    args = parser.parse_args()

    if args.action == "store":
        if not args.refresh_token or not args.league_id:
            print("ERROR: --refresh-token and --league-id required for store")
            exit(1)
        success = store_league_credentials(
            args.database,
            args.refresh_token,
            args.league_id,
            args.league_name or args.league_id
        )
        exit(0 if success else 1)

    elif args.action == "retrieve":
        creds = retrieve_league_credentials(args.database, args.league_id)
        if creds:
            print(json.dumps({k: v for k, v in creds.items() if k != 'refresh_token'}, indent=2))
            print(f"refresh_token: {'*' * 20} (hidden)")
        exit(0 if creds else 1)

    elif args.action == "create-oauth":
        success = create_oauth_file_from_stored(args.database, args.output, args.league_id)
        exit(0 if success else 1)
