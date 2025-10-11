import os
import streamlit as st
import urllib.parse
import requests
import base64
import json
import pandas as pd
from datetime import datetime, timedelta

# Load your client credentials
CLIENT_ID = os.environ.get("YAHOO_CLIENT_ID") or st.secrets.get("YAHOO_CLIENT_ID", None)
CLIENT_SECRET = os.environ.get("YAHOO_CLIENT_SECRET") or st.secrets.get("YAHOO_CLIENT_SECRET", None)

# For deployment, use your actual domain
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


def get_league_standings(access_token: str, league_key: str):
    """Get standings for a specific league"""
    return yahoo_api_call(access_token, f"league/{league_key}/standings?format=json")


def get_league_settings(access_token: str, league_key: str):
    """Get settings for a specific league"""
    return yahoo_api_call(access_token, f"league/{league_key}/settings?format=json")


def get_league_scoreboard(access_token: str, league_key: str):
    """Get scoreboard for a specific league"""
    return yahoo_api_call(access_token, f"league/{league_key}/scoreboard?format=json")


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


def main():
    st.title("🏈 Yahoo Fantasy Football League History Downloader")
    
    # Check if credentials are loaded
    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("❌ Credentials not configured!")
        st.write("Please add your Yahoo credentials to secrets.")
        return
    
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
            except Exception as e:
                st.error(f"❌ Error exchanging code for token: {e}")
        
        return
    
    # Check if we have a stored access token
    if "access_token" in st.session_state:
        access_token = st.session_state.access_token
        
        st.success("🔐 You are authenticated!")
        
        # Fetch user's games if not already loaded
        if "games_data" not in st.session_state:
            with st.spinner("Loading your fantasy games..."):
                try:
                    games_data = get_user_games(access_token)
                    st.session_state.games_data = games_data
                except Exception as e:
                    st.error(f"Error fetching games: {e}")
                    if st.button("Logout and Try Again"):
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
        
        st.subheader("📋 Select a Football Season")
        
        # Display football seasons
        season_options = {f"{game['season']} - {game['name']}": game['game_key'] 
                         for game in football_games}
        
        selected_season = st.selectbox(
            "Choose a season to view leagues:",
            options=list(season_options.keys())
        )
        
        if selected_season:
            game_key = season_options[selected_season]
            
            # Fetch leagues for selected season
            if st.button("Load Leagues"):
                with st.spinner(f"Loading leagues for {selected_season}..."):
                    try:
                        leagues_data = get_user_football_leagues(access_token, game_key)
                        st.session_state.current_leagues = leagues_data
                        st.session_state.current_game_key = game_key
                    except Exception as e:
                        st.error(f"Error fetching leagues: {e}")
            
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
                        })
                    
                    if league_list:
                        st.subheader("🏆 Your Leagues")
                        
                        for league in league_list:
                            with st.expander(f"{league['name']} ({league['num_teams']} teams)"):
                                league_key = league['league_key']
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if st.button("📊 View Standings", key=f"standings_{league_key}"):
                                        with st.spinner("Loading standings..."):
                                            standings = get_league_standings(access_token, league_key)
                                            st.session_state[f"standings_{league_key}"] = standings
                                
                                with col2:
                                    if st.button("⚙️ View Settings", key=f"settings_{league_key}"):
                                        with st.spinner("Loading settings..."):
                                            settings = get_league_settings(access_token, league_key)
                                            st.session_state[f"settings_{league_key}"] = settings
                                
                                with col3:
                                    if st.button("📥 Download All Data", key=f"download_{league_key}"):
                                        with st.spinner("Preparing download..."):
                                            try:
                                                standings = get_league_standings(access_token, league_key)
                                                settings = get_league_settings(access_token, league_key)
                                                
                                                all_data = {
                                                    "league_info": league,
                                                    "standings": standings,
                                                    "settings": settings,
                                                }
                                                
                                                json_str = json.dumps(all_data, indent=2)
                                                st.download_button(
                                                    label="💾 Download JSON",
                                                    data=json_str,
                                                    file_name=f"{league['name'].replace(' ', '_')}_{league_key}.json",
                                                    mime="application/json"
                                                )
                                            except Exception as e:
                                                st.error(f"Error: {e}")
                                
                                # Display stored data
                                if f"standings_{league_key}" in st.session_state:
                                    st.write("**Standings:**")
                                    st.json(st.session_state[f"standings_{league_key}"])
                                
                                if f"settings_{league_key}" in st.session_state:
                                    st.write("**Settings:**")
                                    st.json(st.session_state[f"settings_{league_key}"])
                    
                    else:
                        st.info("No leagues found for this season.")
                
                except Exception as e:
                    st.error(f"Error parsing leagues: {e}")
                    st.json(leagues_data)
        
        st.divider()
        
        if st.button("🔓 Logout"):
            st.session_state.clear()
            st.rerun()
    
    else:
        # Show the authorization button
        st.write("### Connect Your Yahoo Fantasy Account")
        st.write("This app will download your fantasy football league history including:")
        st.write("- League standings and results")
        st.write("- League settings and scoring")
        st.write("- Team rosters and matchups")
        st.write("- Historical data from all seasons")
        
        st.warning("⚠️ Your data is only accessed while you're using the app. Nothing is stored on our servers.")
        
        auth_url = build_authorize_url()
        
        st.link_button("🔐 Connect Yahoo Account", auth_url)
        
        st.divider()
        
        with st.expander("ℹ️ How does this work?"):
            st.markdown("""
            1. Click "Connect Yahoo Account"
            2. Log in to Yahoo and authorize the app
            3. Select which seasons you want to view
            4. Download your league data as JSON files
            
            **Privacy:** We use Yahoo's OAuth 2.0 for secure authentication. 
            Your credentials are never stored, and data is only accessed during your session.
            """)


if __name__ == "__main__":
    main()
