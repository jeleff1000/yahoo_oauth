import xml.etree.ElementTree as ET
from yahoo_oauth import OAuth2
import re
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.DEBUG)

oauth_file = r"C:\Users\joeye\OneDrive\Desktop\yahoo_oauth\oauth\Oauth.json"
league_id = "449.l.198278"

oauth = OAuth2(None, None, from_file=oauth_file)

url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_id}/teams"

response = oauth.session.get(url, timeout=30)

text = response.text
text = re.sub(r' xmlns="[^"]+"', "", text, count=1)
root = ET.fromstring(text)

print("=== TEAMS ===")
for team in root.findall(".//team"):
    team_name = team.findtext("name") or "N/A"

    manager = team.find(".//manager")
    if manager is not None:
        nickname = manager.findtext("nickname") or "MISSING"
        guid = manager.findtext("guid") or "MISSING"
        manager_id = manager.findtext("manager_id") or "MISSING"

        hidden = " <-- HIDDEN" if nickname == "--hidden--" else ""

        # Use ascii replacement for console compatibility
        team_name_safe = team_name.encode("ascii", "replace").decode()

        print(f"Team: {team_name_safe}")
        print(f"  nickname: {nickname}{hidden}")
        print(f"  guid: {guid}")
        print(f"  manager_id: {manager_id}")
        print()
