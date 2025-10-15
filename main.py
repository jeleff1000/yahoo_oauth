import os
import streamlit as st
import urllib.parse
import requests
import base64
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Load your client credentials
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID") or st.secrets.get("YAHOO_CLIENT_ID", None)
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET") or st.secrets.get("YAHOO_CLIENT_SECRET", None)

# For deployment - NO TRAILING SLASH
REDIRECT_URI = os.environ.get("REDIRECT_URI", "http://localhost:8501")

# OAuth 2.0 endpoints
AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

# Paths
ROOT_DIR = Path(__file__).parent
OAUTH_DIR = ROOT_DIR / "oauth"
DATA_DIR = ROOT_DIR / "fantasy_football_data"
SCRIPTS_DIR = ROOT_DIR / "fantasy_football_data_scripts"
INITIAL_IMPORT_SCRIPT = SCRIPTS_DIR / "initial_import.py"


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


def save_oauth_token(token_data: dict, league_info: dict = None):
    """Save OAuth token in the format expected by the scripts"""
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    oauth_file = OAUTH_DIR / "Oauth.json"

    # Format 2 (nested structure) - compatible with oauth_utils.py
    oauth_data = {
        "token_data": {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "consumer_key": CLIENT_ID,
            "consumer_secret": CLIENT_SECRET,
            "token_type": token_data.get("token_type", "bearer"),
            "expires_in": token_data.get("expires_in", 3600),
            "token_time": datetime.utcnow().timestamp(),
            "guid": token_data.get("xoauth_yahoo_guid")
        },
        "timestamp": datetime.now().isoformat()
    }

    if league_info:
        oauth_data["league_info"] = league_info

    with open(oauth_file, 'w') as f:
        json.dump(oauth_data, f, indent=2)

    return oauth_file


def run_initial_import():
    """Run the initial_import.py script to fetch all league data"""
    if not INITIAL_IMPORT_SCRIPT.exists():
        st.error(f"❌ Initial import script not found at: {INITIAL_IMPORT_SCRIPT}")
        return False
    
    try:
        st.info("🚀 Starting initial data import... This may take several minutes.")

        # Create placeholders for progress
        log_placeholder = st.empty()

        # Run the script
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [sys.executable, str(INITIAL_IMPORT_SCRIPT)]

        with st.spinner("Importing league data..."):
            process = subprocess.Popen(
                cmd,
                cwd=str(SCRIPTS_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output to UI
            output_lines = []
            for line in process.stdout:
                output_lines.append(line.strip())
                # Show last 10 lines of output
                log_placeholder.code('\n'.join(output_lines[-10:]))

            process.wait()

            if process.returncode == 0:
                st.success("✅ Data import completed successfully!")
                st.balloons()
                return True
            else:
                st.error(f"❌ Import failed with exit code {process.returncode}")
                st.code('\n'.join(output_lines))
                return False

    except Exception as e:
        st.error(f"❌ Error running import: {e}")
        return False


def main():
    st.title("🏈 Yahoo Fantasy Football League History")
    
    # Check if credentials are loaded
    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("❌ Credentials not configured!")
        st.info("Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET environment variables or in Streamlit secrets.")
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
                
                # Store token data in session state
                st.session_state.token_data = {
                    "access_token": token_data.get("access_token"),
                    "refresh_token": token_data.get("refresh_token"),
                    "token_type": token_data.get("token_type"),
                    "expires_in": token_data.get("expires_in"),
                    "xoauth_yahoo_guid": token_data.get("xoauth_yahoo_guid")
                }
                
                st.session_state.access_token = token_data.get("access_token")
                st.session_state.token_expiry = datetime.utcnow() + timedelta(seconds=token_data.get("expires_in", 3600))
                
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
                    leagues = leagues_data.get("fantasy_content", {}).get("users", {}).get("0", {}).get("user", [])[1].get("games", {}).get("0", {}).get("game", [])[1].get("leagues", {})
                    
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
                        
                        st.write("3. Import your league data:")
                        st.info("This will fetch all historical data from your league and save it locally.")

                        if st.button("📥 Import League Data Now", type="primary"):
                            with st.spinner("Saving OAuth credentials..."):
                                # Save OAuth token with league info
                                oauth_file = save_oauth_token(
                                    st.session_state.token_data,
                                    selected_league
                                )
                                st.success(f"✅ OAuth credentials saved to: {oauth_file}")

                            # Run initial import
                            if run_initial_import():
                                st.success("🎉 All done! Your league data has been imported.")
                                st.info(f"Data saved to: {DATA_DIR}")

                                # Show what was created
                                if DATA_DIR.exists():
                                    st.write("**Files created:**")
                                    parquet_files = list(DATA_DIR.glob("*.parquet"))
                                    if parquet_files:
                                        for pf in parquet_files:
                                            size = pf.stat().st_size / 1024  # KB
                                            st.write(f"- {pf.name} ({size:.1f} KB)")

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
        st.write("### Import Your Fantasy Football League Data")

        st.write("📊 This tool will fetch and save your complete league history including:")
        st.write("- All-time schedules and matchups")
        st.write("- Player statistics")
        st.write("- Transaction history")
        st.write("- Draft data")
        st.write("- Playoff information")

        st.divider()
        
        st.write("**How it works:**")
        st.write("1. Connect your Yahoo account")
        st.write("2. Select your league")
        st.write("3. Data is automatically imported and saved locally")
        st.write("4. Use the data with your analysis scripts")

        st.warning("⚠️ We only access your league data to build local files. Your Yahoo credentials are stored securely in the `oauth/` folder.")

        auth_url = build_authorize_url()
        
        st.link_button("🔐 Connect Yahoo Account", auth_url, type="primary")


if __name__ == "__main__":
    main()
