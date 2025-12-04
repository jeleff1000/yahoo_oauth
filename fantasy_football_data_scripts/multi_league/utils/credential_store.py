"""
Credential Store for Weekly Updates

Stores encrypted OAuth refresh tokens in MotherDuck for automatic weekly updates.
"""

import os
import json
from typing import Optional

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


def get_encryption_key() -> Optional[str]:
    """Get encryption key from environment."""
    return os.environ.get('CREDENTIAL_ENCRYPTION_KEY')


def encrypt_token(token: str, key: str) -> str:
    """Encrypt a token using Fernet symmetric encryption."""
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography package not installed")

    f = Fernet(key.encode() if isinstance(key, str) else key)
    return f.encrypt(token.encode()).decode()


def decrypt_token(encrypted_token: str, key: str) -> str:
    """Decrypt a token using Fernet symmetric encryption."""
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography package not installed")

    f = Fernet(key.encode() if isinstance(key, str) else key)
    return f.decrypt(encrypted_token.encode()).decode()


def store_league_credentials(
    league_id: str,
    league_name: str,
    refresh_token: str,
    database_name: str,
    encryption_key: Optional[str] = None,
    motherduck_token: Optional[str] = None
) -> bool:
    """
    Store encrypted credentials for a league in MotherDuck.

    Args:
        league_id: Yahoo league ID
        league_name: League name
        refresh_token: OAuth refresh token to store
        database_name: MotherDuck database name
        encryption_key: Fernet encryption key (from env if not provided)
        motherduck_token: MotherDuck token (from env if not provided)

    Returns:
        True if successful, False otherwise
    """
    if not DUCKDB_AVAILABLE:
        print("Warning: duckdb not available - cannot store credentials")
        return False

    if not CRYPTO_AVAILABLE:
        print("Warning: cryptography not available - cannot encrypt credentials")
        return False

    key = encryption_key or get_encryption_key()
    if not key:
        print("Warning: No encryption key available")
        return False

    md_token = motherduck_token or os.environ.get('MOTHERDUCK_TOKEN')
    if not md_token:
        print("Warning: No MotherDuck token available")
        return False

    try:
        # Encrypt the refresh token
        encrypted = encrypt_token(refresh_token, key)

        # Store in MotherDuck
        os.environ['MOTHERDUCK_TOKEN'] = md_token
        con = duckdb.connect("md:")

        # Create credentials table if not exists
        con.execute("""
            CREATE TABLE IF NOT EXISTS ops.league_credentials (
                league_id TEXT PRIMARY KEY,
                league_name TEXT,
                database_name TEXT,
                encrypted_refresh_token TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert or update
        con.execute("""
            INSERT OR REPLACE INTO ops.league_credentials
            (league_id, league_name, database_name, encrypted_refresh_token, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [league_id, league_name, database_name, encrypted])

        con.close()
        print(f"✅ Stored credentials for {league_name} ({league_id})")
        return True

    except Exception as e:
        print(f"Error storing credentials: {e}")
        return False


def get_league_credentials(
    league_id: str,
    encryption_key: Optional[str] = None,
    motherduck_token: Optional[str] = None
) -> Optional[str]:
    """
    Retrieve and decrypt credentials for a league.

    Args:
        league_id: Yahoo league ID
        encryption_key: Fernet encryption key (from env if not provided)
        motherduck_token: MotherDuck token (from env if not provided)

    Returns:
        Decrypted refresh token or None if not found
    """
    if not DUCKDB_AVAILABLE or not CRYPTO_AVAILABLE:
        return None

    key = encryption_key or get_encryption_key()
    md_token = motherduck_token or os.environ.get('MOTHERDUCK_TOKEN')

    if not key or not md_token:
        return None

    try:
        os.environ['MOTHERDUCK_TOKEN'] = md_token
        con = duckdb.connect("md:")

        result = con.execute("""
            SELECT encrypted_refresh_token
            FROM ops.league_credentials
            WHERE league_id = ?
        """, [league_id]).fetchone()

        con.close()

        if result:
            return decrypt_token(result[0], key)
        return None

    except Exception as e:
        print(f"Error retrieving credentials: {e}")
        return None
