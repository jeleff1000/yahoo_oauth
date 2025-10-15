import os
import streamlit as st
import urllib.parse
import requests
import base64
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# Load your client credentials
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID") or st.secrets.get("YAHOO_CLIENT_ID", None)
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET") or st.secrets.get("YAHOO_CLIENT_SECRET", None)

# Email configuration
ADMIN_EMAIL = "joeyeleff@gmail.com"
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", None)  # Your email
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", None)  # Your email app password

# For deployment - NO TRAILING SLASH
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


def yahoo_api_call(access_token: str, endpoint: str):
    """Make a call to Yahoo Fantasy API"""
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/{endpoint}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def get_user_games(access_token: str):
    """Get all games the user has participated in"""
    return yahoo_api_call(access_token, "users;use_login=1/games?format=json")


def get_user_football_leagues(access_token: str, game_key: str):
    """Get user's leagues for a specific football game"""
    return yahoo_api_call(access_token, f"users;use_login=1/games;game_keys={game_key}/leagues?format=json")


def extract_football_games(games_data):
    """Extract football games from the games data"""
    football_games = []

    try:
        games = games_data.get("fantasy_content", {}).get("users", {}).get("0", {}).get("user", [])[1].get("games", {})

        for key in games:
            if key == "count":
                continue

            game = games[key].get("game")
            if isinstance(game, list):
                game = game[0]

            if game and game.get("code") == "nfl":
                football_games.append({
                    "game_key": game.get("game_key"),
                    "season": game.get("season"),
                    "name": game.get("name"),
                    "is_game_over": game.get("is_game_over"),
                })
    except Exception as e:
        st.error(f"Error parsing games: {e}")

    return football_games


def send_email(user_email, league_info, token_data):
    """Send email to admin with user's league info and tokens"""

    # Prepare data in BOTH formats
    # Format 1: Flat format for your existing scripts
    flat_token_data = {
        "access_token": token_data["access_token"],
        "consumer_key": token_data["consumer_key"],
        "consumer_secret": token_data["consumer_secret"],
        "guid": token_data.get("guid"),
        "refresh_token": token_data["refresh_token"],
        "token_time": token_data["token_time"],
        "token_type": token_data["token_type"]
    }

    # Format 2: Nested format with all info
    nested_data = {
        "user_email": user_email,
        "league_info": league_info,
        "token_data": token_data,
        "timestamp": datetime.now().isoformat()
    }

    if not SMTP_USERNAME or not SMTP_PASSWORD:
        # Fallback: Show both formats to copy manually
        st.warning("⚠️ Email not configured. Please copy this data and send to joeyeleff@gmail.com:")

        st.write("**Format 1: For your OAuth scripts (save as oauth.json):**")
        st.code(json.dumps(flat_token_data, indent=2))

        st.write("**Format 2: Complete submission data:**")
        st.code(json.dumps(nested_data, indent=2))

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download OAuth Token",
                json.dumps(flat_token_data, indent=2),
                file_name=f"oauth_{league_info['league_key']}.json",
                mime="application/json"
            )
        with col2:
            st.download_button(
                "📥 Download Full Data",
                json.dumps(nested_data, indent=2),
                file_name=f"submission_{league_info['league_key']}.json",
                mime="application/json"
            )

        return False

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f"New League History Request - {league_info.get('name', 'Unknown League')}"

        # Email body
        body = f"""
New League History Request

User Email: {user_email}
Request Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

League Information:
- Name: {league_info['name']}
- Season: {league_info['season']}
- League Key: {league_info['league_key']}
- Teams: {league_info['num_teams']}

==========================================
OAUTH TOKEN DATA (for your scripts):
==========================================

{json.dumps(flat_token_data, indent=2)}

==========================================
COMPLETE SUBMISSION DATA:
==========================================

{json.dumps(nested_data, indent=2)}

You can use the OAuth token data directly in your existing scripts.
Token expires in 1 hour, but you can use the refresh_token to get a new one.
        """

        msg.attach(MIMEText(body, 'plain'))

        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        return True

    except Exception as e:
        st.error(f"Error sending email: {e}")
        # Show fallback with both formats
        st.warning("⚠️ Could not send email. Please copy this data:")

        st.write("**Format 1: For your OAuth scripts (save as oauth.json):**")
        st.code(json.dumps(flat_token_data, indent=2))

        st.write("**Format 2: Complete submission data:**")
        st.code(json.dumps(nested_data, indent=2))

        return False


def main():
    st.title("🏈 Yahoo Fantasy Football League History")

    # Check if credentials are loaded
    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("❌ Credentials not configured!")
        return

    # Check for errors in URL
    qp = st.query_params
    if "error" in qp:
        st.error(f"❌ OAuth Error: {qp.get('error')}")
        if "error_description" in qp:
            st.error(f"Description: {qp.get('error_description')}")
        if st.button("Clear Error & Retry"):
            st.query_params.clear()
            st.rerun()
        return

    # Check if Yahoo redirected back with authorization code
    if "code" in qp:
        code = qp["code"]

        with st.spinner("Connecting to Yahoo..."):
            try:
                token_data = exchange_code_for_tokens(code)

                # Store token data
                st.session_state.token_data = {
                    "access_token": token_data.get("access_token"),
                    "refresh_token": token_data.get("refresh_token"),
                    "consumer_key": CLIENT_ID,
                    "consumer_secret": CLIENT_SECRET,
                    "token_type": token_data.get("token_type"),
                    "expires_in": token_data.get("expires_in"),
                    "token_time": datetime.utcnow().timestamp(),
                    "guid": token_data.get("xoauth_yahoo_guid")
                }

                st.session_state.access_token = token_data.get("access_token")
                st.session_state.token_expiry = datetime.utcnow() + timedelta(
                    seconds=token_data.get("expires_in", 3600))

                st.success("✅ Successfully connected!")

                # Clear the code from URL
                st.query_params.clear()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")

        return

    # Check if we have a stored access token
    if "access_token" in st.session_state:
        access_token = st.session_state.access_token

        st.success("🔐 Connected to Yahoo Fantasy!")

        # Fetch user's games if not already loaded
        if "games_data" not in st.session_state:
            with st.spinner("Loading your fantasy seasons..."):
                try:
                    games_data = get_user_games(access_token)
                    st.session_state.games_data = games_data
                except Exception as e:
                    st.error(f"Error: {e}")
                    if st.button("Start Over"):
                        st.session_state.clear()
                        st.rerun()
                    return

        games_data = st.session_state.games_data
        football_games = extract_football_games(games_data)

        if not football_games:
            st.warning("No football leagues found for your account.")
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()
            return

        st.subheader("📋 Select Your League")

        # Display football seasons
        season_options = {f"{game['season']} NFL Season": game['game_key']
                          for game in football_games}

        selected_season = st.selectbox(
            "1. Choose a season:",
            options=list(season_options.keys())
        )

        if selected_season:
            game_key = season_options[selected_season]

            # Auto-load leagues for selected season
            if "current_game_key" not in st.session_state or st.session_state.current_game_key != game_key:
                with st.spinner("Loading leagues..."):
                    try:
                        leagues_data = get_user_football_leagues(access_token, game_key)
                        st.session_state.current_leagues = leagues_data
                        st.session_state.current_game_key = game_key
                    except Exception as e:
                        st.error(f"Error: {e}")

            # Display leagues if loaded
            if "current_leagues" in st.session_state:
                leagues_data = st.session_state.current_leagues

                try:
                    leagues = \
                    leagues_data.get("fantasy_content", {}).get("users", {}).get("0", {}).get("user", [])[1].get(
                        "games", {}).get("0", {}).get("game", [])[1].get("leagues", {})

                    league_list = []
                    for key in leagues:
                        if key == "count":
                            continue
                        league = leagues[key].get("league", [])[0]
                        league_list.append({
                            "league_key": league.get("league_key"),
                            "name": league.get("name"),
                            "num_teams": league.get("num_teams"),
                            "season": league.get("season"),
                        })

                    if league_list:
                        st.write("2. Choose your league:")

                        league_names = [f"{league['name']} ({league['num_teams']} teams)" for league in league_list]
                        selected_league_name = st.radio("", league_names, key="league_radio")

                        selected_league = league_list[league_names.index(selected_league_name)]

                        st.divider()

                        st.write("3. Enter your email:")
                        user_email = st.text_input("Your email address:", placeholder="your.email@example.com")

                        st.write("4. Submit your request:")

                        if st.button("📤 Submit League History Request", type="primary"):
                            if not user_email or "@" not in user_email:
                                st.error("Please enter a valid email address.")
                            else:
                                with st.spinner("Sending your request..."):
                                    success = send_email(
                                        user_email,
                                        selected_league,
                                        st.session_state.token_data
                                    )

                                    if success:
                                        st.success("✅ Request submitted successfully!")
                                        st.info("We'll email you your league history within 24-48 hours.")
                                        st.balloons()
                                    else:
                                        st.success("✅ Request received!")
                                        st.info("Please send the data shown above to joeyeleff@gmail.com")

                    else:
                        st.info("No leagues found for this season.")

                except Exception as e:
                    st.error(f"Error parsing leagues: {e}")

        st.divider()

        if st.button("Start Over"):
            st.session_state.clear()
            st.rerun()

    else:
        # Show the authorization button
        st.write("### Get Your Fantasy Football League History")

        st.write("📊 We'll create a custom website with your league's complete history including:")
        st.write("- All-time standings and championships")
        st.write("- Season-by-season records")
        st.write("- Head-to-head records")
        st.write("- Playoff brackets")
        st.write("- And much more!")

        st.divider()

        st.write("**How it works:**")
        st.write("1. Connect your Yahoo account")
        st.write("2. Select your league")
        st.write("3. We'll build your custom history site")
        st.write("4. Receive your site within 24-48 hours")

        st.warning(
            "⚠️ We only access your league data to build your history site. Your Yahoo credentials are never stored.")

        auth_url = build_authorize_url()

        st.link_button("🔐 Connect Yahoo Account", auth_url, type="primary")


if __name__ == "__main__":
    main()