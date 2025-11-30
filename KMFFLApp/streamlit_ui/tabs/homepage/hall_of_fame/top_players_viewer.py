# This file is a compatibility shim. The canonical TopPlayersViewer implementation
# lives in ../top_players.py (the module `tabs.hall_of_fame.top_players`).
# We re-export here to avoid breaking imports that referenced `top_players_viewer`.
from .top_players import TopPlayersViewer  # re-export

__all__ = ["TopPlayersViewer"]

