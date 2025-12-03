"""
Draft tab data access.

Optimized loaders for draft tab components with column selection.
"""
from .draft_data import load_draft_data
from .combined import load_optimized_draft_data

__all__ = [
    "load_draft_data",
    "load_optimized_draft_data",
]
