"""
Simulations tab components.

Provides simulation tools including:
- Predictive analytics (playoff odds, machine, etc.)
- What-if scenarios (schedule shuffles, strength of schedule, score tweaks)
"""
from .simulations_overview import SimulationDataViewer, display_simulations_overview

# Backward compatibility
display_simulations_viewer = display_simulations_overview

__all__ = [
    "SimulationDataViewer",
    "display_simulations_overview",
    "display_simulations_viewer",  # backward compatibility
]
