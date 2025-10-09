import os
import streamlit as st
import requests
from requests_oauthlib import OAuth1Session
from datetime import datetime, timedelta

# Load your client credentials from env vars or secrets
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID")
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET")

# OAuth 1.0a endpoints
REQUEST_TOKEN_URL = "https://api.login.yahoo.com/oauth/v2/get_request_token"
AUTHORIZE_URL = "https://api.login.yahoo.com/oauth/v2/request_auth"
ACCESS_TOKEN_URL = "https://api.login.yahoo.com/oauth/v2/get_token"

# Callback URL - using 'oob' for out-of-band (manual code entry)
CALLBACK_URI = "oob"


def get_request_token():
    """Get OAuth 1.0a request token"""
    oauth = OAuth1Session(CLIENT_ID, client_secret=CLIENT_SECRET, callback_uri=CALLBACK_URI)
    fetch_response = oauth.fetch_request_token(REQUEST_TOKEN_URL)
    return fetch_response.get('oauth_token'), fetch_response.get('oauth_token_secret')


def get_authorization_url(request_token):
    """Build the authorization URL"""
    return f"{AUTHORIZE_URL}?oauth_token={request_token}"


def get_access_token(request_token, request_token_secret, verifier):
    """Exchange verifier for access token"""
    oauth = OAuth1Session(
        CLIENT_ID,
        client_secret=CLIENT_SECRET,
        resource_owner_key=request_token,
        resource_owner_secret=request_token_secret,
        verifier=verifier
    )
    oauth_tokens = oauth.fetch_access_token(ACCESS_TOKEN_URL)
    return oauth_tokens.get('oauth_token'), oauth_tokens.get('oauth_token_secret')


def fetch_user_league_data(access_token, access_token_secret):
    """Fetch user's fantasy league data from Yahoo API"""
    oauth = OAuth1Session(
        CLIENT_ID,
        client_secret=CLIENT_SECRET,
        resource_owner_key=access_token,
        resource_owner_secret=access_token_secret
    )
    url = "https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1/games?format=json"
    resp = oauth.get(url)
    resp.raise_for_status()
    return resp.json()


def main():
    st.title("Yahoo Fantasy League History")

    # Debug info
    with st.expander("🔍 Debug Info"):
        st.write("**Client ID configured:**", "✅ Yes" if CLIENT_ID else "❌ No")
        st.write("**Client Secret configured:**", "✅ Yes" if CLIENT_SECRET else "❌ No")
        st.write("**OAuth Method:**", "OAuth 1.0a (out-of-band)")
        st.write("**Session State:**", {k: "..." if "secret" in k.lower() else v for k, v in st.session_state.items()})

    # Check if we have a stored access token
    if "access_token" in st.session_state and "access_token_secret" in st.session_state:
        st.success("🔐 You are authenticated!")

        if st.button("Fetch League Data"):
            with st.spinner("Fetching your fantasy league data..."):
                try:
                    data = fetch_user_league_data(
                        st.session_state.access_token,
                        st.session_state.access_token_secret
                    )
                    st.subheader("Your Fantasy Data")
                    st.json(data)
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    # Step 2: If we have request token, show verifier input
    elif "request_token" in st.session_state:
        st.write("### Step 2: Enter Verification Code")
        st.write("After authorizing the app, Yahoo will show you a verification code.")

        verifier = st.text_input("Enter the verification code from Yahoo:", key="verifier_input")

        if st.button("Submit Verification Code"):
            if verifier:
                with st.spinner("Exchanging verification code for access token..."):
                    try:
                        access_token, access_token_secret = get_access_token(
                            st.session_state.request_token,
                            st.session_state.request_token_secret,
                            verifier
                        )

                        st.session_state.access_token = access_token
                        st.session_state.access_token_secret = access_token_secret

                        # Clean up request token
                        del st.session_state.request_token
                        del st.session_state.request_token_secret

                        st.success("✅ Successfully authenticated!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.error("Please try again or start over.")
            else:
                st.warning("Please enter the verification code.")

        if st.button("Start Over"):
            st.session_state.clear()
            st.rerun()

    # Step 1: Initial authorization
    else:
        st.write("### Step 1: Connect Your Yahoo Fantasy Account")
        st.write("This app uses OAuth 1.0a for secure authentication with Yahoo.")

        if st.button("🔐 Start Authorization"):
            with st.spinner("Getting authorization URL..."):
                try:
                    request_token, request_token_secret = get_request_token()

                    st.session_state.request_token = request_token
                    st.session_state.request_token_secret = request_token_secret

                    auth_url = get_authorization_url(request_token)

                    st.success("✅ Authorization URL generated!")
                    st.write("**Click the link below to authorize:**")
                    st.markdown(f"### [Authorize with Yahoo]({auth_url})")
                    st.write("After authorizing, you'll receive a verification code. Copy it and return here.")

                    st.rerun()

                except Exception as e:
                    st.error(f"Error getting request token: {e}")
                    st.error("Please check your Client ID and Secret are correct.")

        st.divider()

        st.subheader("📋 Setup Instructions")
        st.markdown("""
        **Yahoo Developer Console Setup:**

        1. Make sure your app is set to **"Confidential Client"**
        2. Enable **Fantasy Sports - Read** permissions
        3. For OAuth 1.0a, you don't need to configure a redirect URI
        4. Make sure your app is active/published

        **Streamlit Secrets:**

        Add to your Streamlit Cloud secrets:
        ```toml
        YAHOO_CLIENT_ID = "your_client_id"
        YAHOO_CLIENT_SECRET = "your_client_secret"
        ```
        """)


if __name__ == "__main__":
    main()