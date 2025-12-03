"""
Enhanced Narrative Engine for Player Recaps
Uses the recap_narratives.csv file for dynamic, procedurally-generated player stories
"""
import pandas as pd
import random
from pathlib import Path
from typing import Optional, Dict, Any, List

# Re-export spotlight helper functions so player_recap can import them here
try:
    from .helpers.contextual_helpers import (
        build_player_spotlight_lines,
        build_player_spotlight_paragraph,
    )
except Exception:
    # If import fails at module import time, define no-op fallbacks to avoid hard failures.
    def build_player_spotlight_lines(*args, **kwargs):
        return []

    def build_player_spotlight_paragraph(*args, **kwargs):
        return ""


class NarrativeEngine:
    def __init__(self):
        """Load narrative templates from CSV"""
        csv_path = Path(__file__).parent / "recap_narratives.csv"
        try:
            self.templates = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: Could not load recap_narratives.csv: {e}")
            self.templates = pd.DataFrame()

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition string against context data
        Example: "w=1&a=1&pw=1&ats=1" or "percentile>=0.9"
        """
        if pd.isna(condition) or not condition:
            return True

        parts = condition.split('&')
        for part in parts:
            part = part.strip()

            # Handle comparison operators
            if '>=' in part:
                key, val = part.split('>=')
                key, val = key.strip(), val.strip()
                context_val = context.get(key)
                if context_val is None:
                    return False
                try:
                    if float(context_val) < float(val):
                        return False
                except (ValueError, TypeError):
                    return False

            elif '<=' in part:
                key, val = part.split('<=')
                key, val = key.strip(), val.strip()
                context_val = context.get(key)
                if context_val is None:
                    return False
                try:
                    if float(context_val) > float(val):
                        return False
                except (ValueError, TypeError):
                    return False

            elif '>' in part:
                key, val = part.split('>')
                key, val = key.strip(), val.strip()
                context_val = context.get(key)
                if context_val is None:
                    return False
                try:
                    if float(context_val) <= float(val):
                        return False
                except (ValueError, TypeError):
                    return False

            elif '<' in part:
                key, val = part.split('<')
                key, val = key.strip(), val.strip()
                context_val = context.get(key)
                if context_val is None:
                    return False
                try:
                    if float(context_val) >= float(val):
                        return False
                except (ValueError, TypeError):
                    return False

            elif '=' in part:
                key, val = part.split('=')
                key, val = key.strip(), val.strip()
                context_val = context.get(key)
                if context_val is None:
                    return False
                # Try numeric comparison first
                try:
                    if float(context_val) != float(val):
                        return False
                except (ValueError, TypeError):
                    # Fall back to string comparison
                    if str(context_val) != str(val):
                        return False
            else:
                # Just check if key exists and is truthy
                if not context.get(part):
                    return False

        return True

    def _format_template(self, template: str, context: Dict[str, Any]) -> str:
        """Format template string with context values"""
        result = template
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                # Format based on type
                if isinstance(value, float):
                    if 'percentile' in key.lower():
                        # Convert to percentile display (e.g., 0.95 -> "95th")
                        pct = int(value * 100) if value <= 1 else int(value)
                        suffix = self._ordinal_suffix(pct)
                        formatted = f"{pct}{suffix}"
                    elif 'points' in key.lower() or 'margin' in key.lower() or 'spread' in key.lower():
                        formatted = f"{abs(value):.1f}"
                    else:
                        formatted = f"{value:.2f}"
                elif isinstance(value, int):
                    formatted = str(value)
                else:
                    formatted = str(value)
                result = result.replace(placeholder, formatted)

        return result

    def _ordinal_suffix(self, n: int) -> str:
        """Get ordinal suffix for a number (e.g., 1st, 2nd, 3rd)"""
        if 10 <= n % 100 <= 20:
            return 'th'
        else:
            return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')

    def get_narrative(self, category: str, context: Dict[str, Any], subcategory: Optional[str] = None) -> Optional[str]:
        """
        Get a narrative based on category and context

        Args:
            category: Main category (e.g., 'projection_outcome', 'player_performance')
            context: Dictionary with all available data for template formatting
            subcategory: Optional specific subcategory to filter by

        Returns:
            Formatted narrative string or None
        """
        if self.templates.empty:
            return None

        # Filter by category
        candidates = self.templates[self.templates['category'] == category].copy()

        if subcategory:
            candidates = candidates[candidates['subcategory'] == subcategory]

        if candidates.empty:
            return None

        # Filter by conditions that match
        matching = []
        for _, row in candidates.iterrows():
            if self._evaluate_condition(row['condition'], context):
                matching.append(row)

        if not matching:
            return None

        # Weight-based random selection
        weights = [row['weight'] for row in matching]
        selected = random.choices(matching, weights=weights, k=1)[0]

        # Format and return
        return self._format_template(selected['template'], context)

    def get_player_narrative(self, player_data: Dict[str, Any]) -> List[str]:
        """
        Generate comprehensive player narrative using multiple categories

        Args:
            player_data: Dictionary containing all player stats and flags

        Returns:
            List of narrative strings about the player
        """
        narratives = []

        # Add context to player_data for percentile display
        if 'percentile' in player_data:
            pct_val = player_data['percentile']
            if isinstance(pct_val, (int, float)):
                pct = int(pct_val * 100) if pct_val <= 1 else int(pct_val)
                player_data['percentile_display'] = f"{pct}{self._ordinal_suffix(pct)}"

        # 1. Player performance level
        perf_narrative = self.get_narrative('player_performance', player_data)
        if perf_narrative:
            narratives.append(perf_narrative)

        # 2. Historical context (career/season bests)
        history_narrative = self.get_narrative('player_history', player_data)
        if history_narrative:
            narratives.append(history_narrative)

        # 3. Manager-specific history
        manager_narrative = self.get_narrative('manager_history', player_data)
        if manager_narrative:
            narratives.append(manager_narrative)

        # 4. Context (consistency, breakout, regression)
        context_narrative = self.get_narrative('context', player_data)
        if context_narrative:
            narratives.append(context_narrative)

        return narratives

    def get_matchup_narrative(self, matchup_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate matchup outcome narrative

        Args:
            matchup_data: Dictionary with w, a, pw, ats flags and relevant stats

        Returns:
            Formatted narrative string
        """
        return self.get_narrative('projection_outcome', matchup_data)


# Singleton instance
_engine = None

def get_engine() -> NarrativeEngine:
    """Get or create the narrative engine singleton"""
    global _engine
    if _engine is None:
        _engine = NarrativeEngine()
    return _engine
