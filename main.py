import streamlit as st
import urllib.parse

CLIENT_ID = "your_yahoo_client_id"
REDIRECT_URI = "https://yourapp.streamlit.app/callback"
AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"

# Build the OAuth URL
params = {
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "response_type": "code",
    "scope": "fspt-r",
}
url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

st.markdown(f"[![Auth with Yahoo](https://developer.yahoo.com/static/img/yahoo.png)]({url})")
