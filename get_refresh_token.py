#!/usr/bin/env python3
"""
Get Yahoo OAuth Refresh Token

This script handles Yahoo OAuth authentication locally by:
1. Opening the Yahoo auth URL in your browser
2. You log in and authorize
3. Yahoo shows you a code (since redirect is 'oob')
4. You paste the code here
5. Script exchanges it for tokens and saves to oauth/Oauth.json

Usage:
    python get_refresh_token.py
"""

import os
import json
import webbrowser
from pathlib import Path
from urllib.parse import urlencode
import requests

def main():
    print("=" * 60)
    print("Yahoo OAuth Token Generator")
    print("=" * 60)

    # Try to get credentials from environment or .env file
    client_id = os.environ.get('YAHOO_CLIENT_ID')
    client_secret = os.environ.get('YAHOO_CLIENT_SECRET')

    # Try to load from streamlit secrets
    if not client_id or not client_secret:
        secrets_path = Path('.streamlit/secrets.toml')
        if secrets_path.exists():
            print("Loading credentials from .streamlit/secrets.toml...")
            content = secrets_path.read_text()
            for line in content.split('\n'):
                if 'YAHOO_CLIENT_ID' in line and '=' in line:
                    client_id = line.split('=')[1].strip().strip('"\'')
                if 'YAHOO_CLIENT_SECRET' in line and '=' in line:
                    client_secret = line.split('=')[1].strip().strip('"\'')

    # Prompt if still missing
    if not client_id:
        print("\nEnter your Yahoo Client ID (from developer.yahoo.com):")
        client_id = input("> ").strip()

    if not client_secret:
        print("\nEnter your Yahoo Client Secret:")
        client_secret = input("> ").strip()

    if not client_id or not client_secret:
        print("❌ Client ID and Secret are required")
        return

    print(f"\n✓ Using Client ID: {client_id[:20]}...")

    # Build authorization URL
    auth_url = "https://api.login.yahoo.com/oauth2/request_auth"
    params = {
        'client_id': client_id,
        'redirect_uri': 'oob',  # Out-of-band - shows code on screen
        'response_type': 'code',
        'scope': 'openid',
    }

    full_url = f"{auth_url}?{urlencode(params)}"

    print("\n" + "=" * 60)
    print("STEP 1: Authorize with Yahoo")
    print("=" * 60)
    print("\nOpening browser to Yahoo login...")
    print("If browser doesn't open, visit this URL manually:\n")
    print(full_url)

    webbrowser.open(full_url)

    print("\n" + "=" * 60)
    print("STEP 2: Enter the authorization code")
    print("=" * 60)
    print("\nAfter logging in, Yahoo will show you a code.")
    print("Copy and paste that code here:\n")

    auth_code = input("Authorization code> ").strip()

    if not auth_code:
        print("❌ No code entered")
        return

    print("\n" + "=" * 60)
    print("STEP 3: Exchanging code for tokens")
    print("=" * 60)

    # Exchange authorization code for tokens
    token_url = "https://api.login.yahoo.com/oauth2/get_token"

    try:
        response = requests.post(
            token_url,
            data={
                'grant_type': 'authorization_code',
                'code': auth_code,
                'redirect_uri': 'oob',
                'client_id': client_id,
                'client_secret': client_secret,
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ Token exchange failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return

        tokens = response.json()

        access_token = tokens.get('access_token')
        refresh_token = tokens.get('refresh_token')
        expires_in = tokens.get('expires_in', 3600)

        if not access_token or not refresh_token:
            print(f"❌ Missing tokens in response: {tokens}")
            return

        print(f"✅ Got access_token: {access_token[:30]}...")
        print(f"✅ Got refresh_token: {refresh_token[:30]}...")

    except Exception as e:
        print(f"❌ Error exchanging code: {e}")
        return

    print("\n" + "=" * 60)
    print("STEP 4: Saving tokens")
    print("=" * 60)

    # Save to oauth/Oauth.json
    import time
    oauth_data = {
        'consumer_key': client_id,
        'consumer_secret': client_secret,
        'access_token': access_token,
        'refresh_token': refresh_token,
        'token_time': time.time(),
        'token_type': 'bearer',
    }

    oauth_dir = Path('oauth')
    oauth_dir.mkdir(exist_ok=True)
    oauth_file = oauth_dir / 'Oauth.json'

    with open(oauth_file, 'w') as f:
        json.dump(oauth_data, f, indent=2)

    print(f"✅ Saved to {oauth_file}")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nYour refresh_token (copy this for GitHub workflow):\n")
    print("-" * 60)
    print(refresh_token)
    print("-" * 60)
    print("\nNext steps:")
    print("1. Copy the refresh_token above")
    print("2. Go to GitHub → Actions → Refresh OAuth Tokens")
    print("3. Paste the token and run the workflow")
    print("4. Also update YAHOO_MASTER_REFRESH_TOKEN secret with this token")


if __name__ == "__main__":
    main()
