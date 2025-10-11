import os
import streamlit as st
import urllib.parse
import requests
import base64
from datetime import datetime, timedelta

# Load your client credentials
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID") or st.secrets.get("YAHOO_CLIENT_ID", None)
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET") or st.secrets.get("YAHOO_CLIENT_SECRET", None)

# For deployment, use your actual domain. For local testing, use localhost
REDIRECT_URI = os.environ.get("REDIRECT_URI", "https://leaguehistory.streamlit.app")

# OAuth 2.0 endpoints
AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"


def get_auth_header():
    """Create Basic Auth header as per Yahoo's requirements"""
    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


def build_authorize_url(state: str = None) -> str:
    """Build the Yahoo OAuth authorization URL"""
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
    }
    if state:
        params["state"] = state
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for access token"""
    headers = {
        "Authorization": get_auth_header(),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
        "code": code,
    }
    
    resp = requests.post(TOKEN_URL, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(refresh_token: str) -> dict:
    """Refresh an expired access token"""
    headers = {
        "Authorization": get_auth_header(),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "refresh_token",
        "redirect_uri": REDIRECT_URI,
        "refresh_token": refresh_token,
    }
    
    resp = requests.post(TOKEN_URL, headers=headers, data=data)
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
    
    # Check if credentials are loaded
    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("❌ Credentials not configured!")
        st.write("Please add your Yahoo credentials to secrets.")
        return
    
    # Debug info
    with st.expander("🔍 Debug Info"):
        st.write("**Client ID configured:**", "✅ Yes" if CLIENT_ID else "❌ No")
        st.write("**Client Secret configured:**", "✅ Yes" if CLIENT_SECRET else "❌ No")
        st.write("**Redirect URI:**", REDIRECT_URI)
        st.write("**Current URL params:**", dict(st.query_params))
        st.write("**Platform:**", "Hugging Face" if "SPACE_ID" in os.environ else "Other")
    
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
                
                # Store tokens in session state
                st.session_state.access_token = access_token
                st.session_state.refresh_token = refresh_token
                st.session_state.token_expiry = expiry_time
                
                # Clear the code from URL
                st.query_params.clear()
                st.rerun()
                
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ HTTP Error: {e}")
                if e.response is not None:
                    st.error(f"Status Code: {e.response.status_code}")
                    st.error(f"Response: {e.response.text}")
                    
                    # Check if it's a rate limit
                    if e.response.status_code == 429:
                        st.warning("🚫 Rate limited. Try deploying to a different platform or running locally.")
            except Exception as e:
                st.error(f"❌ Error exchanging code for token: {e}")
        
        return
    
    # Check if we have a stored access token
    if "access_token" in st.session_state:
        access_token = st.session_state.access_token
        
        st.success("🔐 You are authenticated!")
        
        # Show token expiry
        if "token_expiry" in st.session_state:
            expiry = st.session_state.token_expiry
            now = datetime.utcnow()
            if now < expiry:
                remaining = (expiry - now).total_seconds() / 60
                st.info(f"Token expires in {remaining:.0f} minutes")
            else:
                st.warning("Token expired. Refreshing...")
                try:
                    new_tokens = refresh_access_token(st.session_state.refresh_token)
                    st.session_state.access_token = new_tokens.get("access_token")
                    st.session_state.refresh_token = new_tokens.get("refresh_token")
                    st.session_state.token_expiry = datetime.utcnow() + timedelta(seconds=new_tokens.get("expires_in", 3600))
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to refresh token: {e}")
                    del st.session_state.access_token
                    st.rerun()
        
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
                        st.error(f"Response: {e.response.text}")
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
        
        # Use Streamlit's link_button
        st.link_button("🔐 Connect Yahoo Account", auth_url)
        
        st.divider()
        
        st.subheader("📋 Setup Instructions")
        st.markdown("""
        **Yahoo Developer Console Setup:**
        
        1. Go to https://developer.yahoo.com/apps/
        2. Create or select your app
        3. Set **Redirect URI** to your app URL (e.g., `https://leaguehistory.streamlit.app`)
        4. Enable **Fantasy Sports - Read** permissions
        5. Set to **Confidential Client**
        
        **If you're getting rate limited:**
        - Yahoo may have blocked cloud hosting IPs
        - Try running locally: `streamlit run main.py`
        - Or deploy to a VPS with dedicated IP
        """)


if __name__ == "__main__":
    main()
