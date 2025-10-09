import os
import streamlit as st
import urllib.parse
import requests
from datetime import datetime, timedelta

# Load your client credentials from env vars or secrets
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID")
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET")

# Must match exactly what you register in Yahoo Developer Console
# Streamlit apps don't support custom paths, so we use the root URL
REDIRECT_URI = "https://leaguehistory.streamlit.app/"

AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"


def build_authorize_url(state: str = None) -> str:
    """Build the Yahoo OAuth authorization URL"""
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "language": "en-us"
    }
    if state:
        params["state"] = state
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for access token"""
    data = {
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
        "code": code,
    }
    resp = requests.post(TOKEN_URL, data=data, auth=(CLIENT_ID, CLIENT_SECRET))
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(refresh_token: str) -> dict:
    """Refresh an expired access token"""
    data = {
        "grant_type": "refresh_token",
        "redirect_uri": REDIRECT_URI,
        "refresh_token": refresh_token,
    }
    resp = requests.post(TOKEN_URL, data=data, auth=(CLIENT_ID, CLIENT_SECRET))
    resp.raise_for_status()
    return resp.json()


def fetch_user_league_data(access_token: str):
    """Fetch user's fantasy league data from Yahoo API"""
    headers = {"Authorization": f"Bearer {access_token}"}
    url = "https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1/games?format=json"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def main():
    st.title("Yahoo Fantasy League History")

    # Debug info (remove in production)
    with st.expander("🔍 Debug Info"):
        st.write("**Client ID configured:**", "✅ Yes" if CLIENT_ID else "❌ No")
        st.write("**Client Secret configured:**", "✅ Yes" if CLIENT_SECRET else "❌ No")
        st.write("**Redirect URI:**", REDIRECT_URI)
        st.write("**Current URL:**", st.query_params)

    # Check for errors in URL
    qp = st.query_params
    if "error" in qp:
        st.error(f"❌ OAuth Error: {qp.get('error')}")
        if "error_description" in qp:
            st.error(f"Description: {qp.get('error_description')}")
        st.info("Please try authorizing again.")
        if st.button("Clear Error & Retry"):
            st.query_params.clear()
            st.rerun()
        return

    # Check if Yahoo redirected back with authorization code
    if "code" in qp:
        code = qp["code"]

        with st.spinner("Exchanging authorization code for access token..."):
            try:
                token_data = exchange_code_for_tokens(code)

                access_token = token_data.get("access_token")
                refresh_token = token_data.get("refresh_token")
                expires_in = token_data.get("expires_in", 3600)

                expiry_time = datetime.utcnow() + timedelta(seconds=int(expires_in))

                st.success("✅ Successfully authenticated with Yahoo!")

                # Store tokens in session state (in production, use a database)
                st.session_state.access_token = access_token
                st.session_state.refresh_token = refresh_token
                st.session_state.token_expiry = expiry_time

                # Clear the code from URL
                st.query_params.clear()
                st.rerun()

            except requests.exceptions.HTTPError as e:
                st.error(f"❌ HTTP Error: {e}")
                if e.response is not None:
                    st.error(f"Response: {e.response.text}")
            except Exception as e:
                st.error(f"❌ Error exchanging code for token: {e}")

        return

    # Check if we have a stored access token
    if "access_token" in st.session_state:
        access_token = st.session_state.access_token

        st.success("🔐 You are authenticated!")

        if st.button("Fetch League Data"):
            with st.spinner("Fetching your fantasy league data..."):
                try:
                    data = fetch_user_league_data(access_token)
                    st.subheader("Your Fantasy Data")
                    st.json(data)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 401:
                        st.warning("Token expired. Please re-authenticate.")
                        del st.session_state.access_token
                        st.rerun()
                    else:
                        st.error(f"Error fetching data: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    else:
        # Show the authorization button
        st.write("### Connect Your Yahoo Fantasy Account")
        st.write("Click the button below to authorize this app to access your Yahoo Fantasy Sports data.")

        auth_url = build_authorize_url()

        st.markdown(f"""
        <a href="{auth_url}" target="_self">
            <button style="
                background-color: #6001d2;
                color: white;
                padding: 12px 24px;
                font-size: 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            ">
                🔐 Connect Yahoo Account
            </button>
        </a>
        """, unsafe_allow_html=True)

        st.divider()

        st.subheader("Troubleshooting Tips")
        st.markdown("""
        **If you see an error from Yahoo:**

        1. **Check Redirect URI**: In your Yahoo Developer Console, ensure the redirect URI is EXACTLY:
           ```
           https://leaguehistory.streamlit.app/
           ```
           (With trailing slash, exact match)

        2. **Verify Client Credentials**: Make sure your `YAHOO_CLIENT_ID` and `YAHOO_CLIENT_SECRET` 
           environment variables are set correctly in Streamlit Cloud.

        3. **App Permissions**: In Yahoo Developer Console, ensure your app has:
           - Read permissions enabled
           - Fantasy Sports API access

        4. **App Status**: Your Yahoo app must be in "Published" or "Deployed" status.

        5. **Test in Private/Incognito**: Sometimes cached credentials cause issues.
        """)


if __name__ == "__main__":
    main()