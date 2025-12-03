"""
Playoff Bracket Module (Import Wrapper)

This file maintains backward compatibility by importing from the refactored
playoff_bracket submodule.

NEW STRUCTURE:
- playoff_bracket/
  - __init__.py (main entry point)
  - utils.py (settings, validation)
  - championship_bracket.py (champion detection)
  - consolation_bracket.py (sacko detection)
  - placement_games.py (placement game detection)

OLD USAGE (still supported):
    from playoff_bracket import simulate_playoff_brackets

NEW USAGE (recommended):
    from playoff_bracket import simulate_playoff_brackets
    # OR
    import playoff_bracket
    playoff_bracket.simulate_playoff_brackets(...)
"""

# Import from submodule
from .playoff_bracket import simulate_playoff_brackets

# Re-export for backward compatibility
__all__ = ['simulate_playoff_brackets']
