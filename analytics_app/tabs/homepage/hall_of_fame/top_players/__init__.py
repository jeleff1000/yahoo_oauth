# Safely re-export TopPlayersViewer from the sibling top_players.py file without causing
# circular imports. We locate the top_players.py file in the parent directory and load it
# under a private module name, then expose TopPlayersViewer here.
from pathlib import Path
import importlib.util
import sys

_this_dir = Path(__file__).parent
_source = _this_dir.parent / "top_players.py"

if _source.exists():
    spec = importlib.util.spec_from_file_location(
        "tabs.hall_of_fame._top_players_impl", str(_source)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore
    TopPlayersViewer = getattr(module, "TopPlayersViewer")
else:
    # Fallback: define a placeholder to fail loudly at runtime
    TopPlayersViewer = None

__all__ = ["TopPlayersViewer"]
