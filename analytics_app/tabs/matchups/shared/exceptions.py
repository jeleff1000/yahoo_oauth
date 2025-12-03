"""
Exception handling utilities for matchups tab.
Provides better error reporting and logging.
"""

import streamlit as st
import logging
from typing import Optional, Callable, Any
from functools import wraps

# Configure logger
logger = logging.getLogger(__name__)


class MatchupDataError(Exception):
    """Base exception for matchup data errors."""

    pass


class DataNotFoundError(MatchupDataError):
    """Raised when required data is not found."""

    pass


class FilterError(MatchupDataError):
    """Raised when filtering operations fail."""

    pass


class GraphRenderError(MatchupDataError):
    """Raised when graph rendering fails."""

    pass


class ComponentLoadError(MatchupDataError):
    """Raised when a UI component fails to load."""

    pass


def handle_graph_import_error(func: Callable) -> Callable:
    """
    Decorator to handle graph import errors gracefully.

    Args:
        func: Function that imports and displays graphs

    Returns:
        Wrapped function with error handling
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            logger.error(f"Graph import failed: {e}")
            st.warning(
                f"⚠️ Graph unavailable: Could not import required module - {str(e)}"
            )
            return None
        except ModuleNotFoundError as e:
            logger.error(f"Graph module not found: {e}")
            st.warning(f"⚠️ Graph unavailable: Module not found - {str(e)}")
            return None
        except AttributeError as e:
            logger.error(f"Graph attribute error: {e}")
            st.warning(f"⚠️ Graph unavailable: Function or class not found - {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading graph: {e}", exc_info=True)
            st.error(f"❌ Error loading graph: {str(e)}")
            return None

    return wrapper


def handle_data_operation(func: Callable) -> Callable:
    """
    Decorator to handle data operation errors gracefully.

    Args:
        func: Function that performs data operations

    Returns:
        Wrapped function with error handling
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            logger.error(f"Data column missing: {e}")
            st.error(f"❌ Data error: Required column {str(e)} not found in dataset")
            return None
        except ValueError as e:
            logger.error(f"Data value error: {e}")
            st.error(f"❌ Data error: {str(e)}")
            return None
        except TypeError as e:
            logger.error(f"Data type error: {e}")
            st.error(f"❌ Data type error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected data operation error: {e}", exc_info=True)
            st.error(f"❌ Unexpected error: {str(e)}")
            return None

    return wrapper


def safe_import_graph_module(
    module_path: str, function_name: str, fallback_message: Optional[str] = None
) -> Optional[Callable]:
    """
    Safely import a graph module and function.

    Args:
        module_path: Python module path (e.g., '.graphs.win_loss_graph')
        function_name: Function name to import
        fallback_message: Optional custom message if import fails

    Returns:
        Imported function or None if import fails
    """
    try:
        # Import the module
        from importlib import import_module

        module = import_module(module_path, package="tabs.matchups")

        # Get the function
        if hasattr(module, function_name):
            return getattr(module, function_name)
        else:
            logger.error(f"Function {function_name} not found in {module_path}")
            if fallback_message:
                st.warning(f"⚠️ {fallback_message}")
            else:
                st.warning(f"⚠️ Graph function '{function_name}' not available")
            return None

    except ImportError as e:
        logger.error(f"Failed to import {module_path}: {e}")
        if fallback_message:
            st.warning(f"⚠️ {fallback_message}")
        else:
            st.warning(f"⚠️ Graph module unavailable: {str(e)}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error importing {module_path}: {e}", exc_info=True)
        st.error(f"❌ Error loading graph: {str(e)}")
        return None


def validate_dataframe(df: Any, required_columns: list[str]) -> bool:
    """
    Validate that a DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, False otherwise (also shows error message)
    """
    try:
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            st.error("❌ Invalid data: Expected DataFrame")
            return False

        if df.empty:
            st.info("ℹ️ No data available")
            return False

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(
                f"❌ Data error: Missing required columns: {', '.join(missing_columns)}"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"DataFrame validation error: {e}", exc_info=True)
        st.error(f"❌ Validation error: {str(e)}")
        return False


def log_performance(operation_name: str) -> Callable:
    """
    Decorator to log performance of operations.

    Args:
        operation_name: Name of the operation being performed

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                if elapsed > 1.0:  # Log slow operations (> 1 second)
                    logger.warning(f"{operation_name} took {elapsed:.2f}s")
                else:
                    logger.debug(f"{operation_name} took {elapsed:.2f}s")

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"{operation_name} failed after {elapsed:.2f}s: {e}", exc_info=True
                )
                raise

        return wrapper

    return decorator
