from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
import xml.etree.ElementTree as ET
import re
import csv
import nfl_data_py as nfl
import pandas as pd
import time
import datetime
import requests
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter
import multiprocessing as mp
from requests.exceptions import ConnectionError, Timeout

def clean_name(name):
    # Replace specific character sequences
    name = re.sub(r'[èéêëÈÉÊË]', 'e', name)

    pattern = r"[.\-']|(\bjr\b\.?)|(\bsr\b\.?)|(\bII\b)|(\bIII\b)"
    cleaned_name = re.sub(pattern, '', name, flags=re.IGNORECASE).strip()
    capitalized_name = ' '.join([word.capitalize() for word in cleaned_name.split()])
    return capitalized_name

# Define the manual mapping for player names (case insensitive)
manual_mapping = {
    "hollywood brown": "Marquise Brown",
    "will fuller v": "Will Fuller",
    "jeff wilson": "Jeffery Wilson",
    "willie snead iv": "Willie Snead",
    "charles johnson": "Charles D Johnson",
    "kenneth barber": "Peyton Barber",
    "rodney smith": "Rod Smith",
    "bisi johnson": "Olabisi Johnson",
    "chris herndon": "Christopher Herndon",
    "scotty miller": "Scott Miller",
    "trenton richardson": "Trent Richardson"
}

# Function to apply manual mapping
def apply_manual_mapping(name):
    return manual_mapping.get(name.lower(), name)

# Function to rename players based on conditions
def rename_players(filtered_data):
    filtered_data.loc[
        (filtered_data['name_full'] == 'Alex Smith') & (filtered_data['primary_position'] == 'TE'),
        'name_full'
    ] = 'Edwin Smith'
    filtered_data.loc[
        (filtered_data['name_full'] == 'David Johnson') & (filtered_data['primary_position'] == 'TE'),
        'name_full'
    ] = 'Dave Johnson'
    filtered_data.loc[
        (filtered_data['name_full'] == 'Ryan Griffin') & (filtered_data['primary_position'] == 'QB'),
        'name_full'
    ] = 'Ryan Walsh Griffin'
    filtered_data.loc[
        (filtered_data['name_full'] == 'Chris Thompson') & (filtered_data['primary_position'] == 'WR'),
        'name_full'
    ] = 'Christopher Thompson'
    # Rename team names based on player_id
    filtered_data.loc[
        filtered_data['player_id'] == '100019', 'name_full'
    ] = 'Giants'
    filtered_data.loc[
        filtered_data['player_id'] == '100020', 'name_full'
    ] = 'Jets'
    filtered_data.loc[
        filtered_data['player_id'] == '100014', 'name_full'
    ] = 'Rams'
    filtered_data.loc[
        filtered_data['player_id'] == '100024', 'name_full'
    ] = 'Chargers'
    return filtered_data