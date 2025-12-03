#!/usr/bin/env python3
"""
Auto-add @st.fragment decorators to tab display functions.
This prevents reruns from jumping back to the first tab.
"""
import re
from pathlib import Path

# Directory containing tab files
TABS_DIR = Path(__file__).parent / "tabs"

# Function name patterns that should get @st.fragment
FUNCTION_PATTERNS = [
    r"^def display\(",
    r"^def display_.*\(",
    r"^def render\(",
    r"^def render_.*\(",
    r"^def _display_.*\(",  # Private display methods (subtabs)
    r"^def _render_.*\(",   # Private render methods (subtabs)
]

def should_add_fragment(line: str) -> bool:
    """Check if this function definition should get @st.fragment"""
    stripped = line.strip()
    for pattern in FUNCTION_PATTERNS:
        if re.match(pattern, stripped):
            return True
    return False

def has_fragment_decorator(lines: list, func_line_idx: int) -> bool:
    """Check if function already has @st.fragment decorator"""
    # Look at previous lines for decorator
    for i in range(max(0, func_line_idx - 5), func_line_idx):
        if "@st.fragment" in lines[i]:
            return True
    return False

def add_fragments_to_file(file_path: Path) -> tuple[bool, int]:
    """
    Add @st.fragment to display/render functions in a file.
    Returns: (was_modified, num_fragments_added)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  WARN: Error reading {file_path.name}: {e}")
        return False, 0

    modified = False
    fragments_added = 0
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a function that needs @st.fragment
        if should_add_fragment(line) and not has_fragment_decorator(lines, i):
            # Get indentation of the function
            indent = len(line) - len(line.lstrip())
            indent_str = line[:indent]

            # Check if streamlit is imported
            has_st_import = any("import streamlit" in l for l in lines[:i])

            if has_st_import:
                # Add @st.fragment decorator with same indentation
                new_lines.append(f"{indent_str}@st.fragment\n")
                fragments_added += 1
                modified = True

        new_lines.append(line)
        i += 1

    # Write back if modified
    if modified:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return True, fragments_added
        except Exception as e:
            print(f"  WARN: Error writing {file_path.name}: {e}")
            return False, 0

    return False, 0

def main():
    """Process all Python files in tabs directory"""
    print("Scanning for tab files...")
    print(f"Directory: {TABS_DIR}")
    print()

    if not TABS_DIR.exists():
        print(f"ERROR: Directory not found: {TABS_DIR}")
        return

    # Find all Python files recursively
    py_files = list(TABS_DIR.rglob("*.py"))
    print(f"Found {len(py_files)} Python files")
    print()

    total_modified = 0
    total_fragments = 0

    for py_file in sorted(py_files):
        # Skip __pycache__ and special files
        if "__pycache__" in str(py_file) or py_file.name.startswith("_"):
            continue

        modified, num_fragments = add_fragments_to_file(py_file)

        if modified:
            rel_path = py_file.relative_to(TABS_DIR)
            print(f"[OK] {rel_path}: Added {num_fragments} @st.fragment decorator(s)")
            total_modified += 1
            total_fragments += num_fragments

    print()
    print("=" * 60)
    print(f"Complete!")
    print(f"Modified {total_modified} files")
    print(f"Added {total_fragments} @st.fragment decorators")
    print("=" * 60)
    print()
    print("Tip: Restart your Streamlit app to see the changes")

if __name__ == "__main__":
    main()
