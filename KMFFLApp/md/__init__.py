"""
Top-level shim package to expose streamlit_ui.md as `md`.

This lets external scripts use `import md` (the same API as the streamlit_ui/md package)
when the repository root is on PYTHONPATH or the project is installed in editable mode
(e.g. `pip install -e .`).

This file intentionally re-exports all public names from streamlit_ui.md.
"""
# Try the straightforward re-export first
try:
    from streamlit_ui.md import *  # noqa: F401,F403
    # Attempt to mirror __all__ if present (use runtime lookup to avoid static-analysis warnings)
    try:
        import importlib
        _mod = importlib.import_module("streamlit_ui.md")
        _streamlit_all = getattr(_mod, "__all__", None)
        if _streamlit_all is not None:
            __all__ = list(_streamlit_all)
    except Exception:
        # If the subpackage doesn't define __all__ or importlib fails, leave it alone
        pass
except Exception:
    # If direct import fails (for example, because the package path isn't on sys.path),
    # provide a helpful fallback that adds the repository root to sys.path and retries.
    import sys
    import os

    # Calculate repo root (two levels up if this file is inside the repo root's md/ directory)
    this_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(this_dir))

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Retry importing again. If this still fails, let the original ImportError propagate.
    from streamlit_ui.md import *  # noqa: F401,F403
    try:
        import importlib
        _mod = importlib.import_module("streamlit_ui.md")
        _streamlit_all = getattr(_mod, "__all__", None)
        if _streamlit_all is not None:
            __all__ = list(_streamlit_all)
    except Exception:
        pass
