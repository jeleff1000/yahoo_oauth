"""Shared modern table styling for predictive viewers using AgGrid with theme-adaptive blue-green gradient"""
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode


def _get_responsive_width(col_name: str, is_first: bool = False, total_columns: int = 10) -> dict:
    """
    Get responsive column width configuration based on column type.
    Returns dict with 'width', 'minWidth', 'maxWidth' for AgGrid.
    """
    if is_first:
        return {'width': 100, 'minWidth': 80, 'maxWidth': 140}
    elif col_name in ['Year', 'Week', 'Actual', 'Record']:
        return {'width': 65, 'minWidth': 50, 'maxWidth': 85}
    elif 'Seed' in col_name or 'seed' in col_name:
        return {'width': 55, 'minWidth': 45, 'maxWidth': 75}
    elif '%' in col_name or col_name.startswith('p_') or col_name.startswith('P_'):
        return {'width': 55, 'minWidth': 45, 'maxWidth': 75}
    elif '-' in col_name and col_name[0].isdigit():  # Record columns like "10-4"
        return {'width': 50, 'minWidth': 40, 'maxWidth': 65}
    else:
        return {'width': 60, 'minWidth': 45, 'maxWidth': 80}


def render_modern_table(df, title="", color_columns=None, reverse_columns=None, format_specs=None, column_names=None, gradient_mode="row"):
    """
    Render a modern styled AgGrid table with blue-green gradient coloring and sorting.

    Optimized for both mobile and desktop with responsive column widths.

    Args:
        df: DataFrame to render
        title: Optional table title
        color_columns: List of columns to apply gradient coloring
        reverse_columns: List of columns where lower is better (reversed gradient)
        format_specs: Dict of column_name: format_string (e.g., {'col': '{:.2f}'})
        column_names: Dict mapping old column names to display names (e.g., {'win': 'Wins'})
        gradient_mode: "row" for row-based gradients (compare values in same row) or
                      "column" for column-based gradients (compare values in same column)
    """
    color_columns = color_columns or []
    reverse_columns = reverse_columns or []
    format_specs = format_specs or {}
    column_names = column_names or {}

    if title:
        st.markdown(f"<h4 style='margin-top: 1rem; margin-bottom: 0.5rem;'>{title}</h4>", unsafe_allow_html=True)

    # Prepare dataframe - reset index to make it a regular column
    display_df = df.reset_index()

    # Filter out columns where all values are 0.00
    for col in display_df.columns:
        if col not in ['Manager', 'Year'] and col in display_df.select_dtypes(include=['number']).columns:
            if display_df[col].notna().any() and (display_df[col].fillna(0) == 0).all():
                # Drop columns that are all zeros
                display_df = display_df.drop(columns=[col])
                if col in color_columns:
                    color_columns.remove(col)
                if col in reverse_columns:
                    reverse_columns.remove(col)
                if col in format_specs:
                    del format_specs[col]
                if col in column_names:
                    del column_names[col]

    # Calculate min/max for gradient columns BEFORE renaming - exclude Total row
    gradients = {}
    all_gradient_cols = list(set(color_columns + reverse_columns))
    for col in all_gradient_cols:
        if col in display_df.columns:
            # Exclude Total row from gradient calculation
            col_data = display_df[col].copy()

            # Check for Total row in first column(s)
            mask = pd.Series([True] * len(display_df))
            for c in display_df.columns[:3]:  # Check first few columns for 'Total'
                if display_df[c].dtype == 'object' or display_df[c].dtype.name == 'string':
                    mask &= display_df[c].astype(str).str.lower() != 'total'

            col_data = col_data[mask]
            numeric_vals = pd.to_numeric(col_data, errors='coerce').dropna()

            if len(numeric_vals) > 0:
                gradients[col] = {
                    'min': numeric_vals.min(),
                    'max': numeric_vals.max(),
                    'reverse': col in reverse_columns
                }

    # Rename columns for display AFTER calculating gradients
    renamed_gradient_cols = {}  # Map original -> renamed for gradient columns
    if column_names:
        display_df = display_df.rename(columns=column_names)
        # Track mapping of original to renamed for gradient columns
        for orig_col in all_gradient_cols:
            renamed_gradient_cols[orig_col] = column_names.get(orig_col, orig_col)
        # Also update color_columns and reverse_columns to use renamed names
        color_columns = [column_names.get(col, col) for col in color_columns]
        reverse_columns = [column_names.get(col, col) for col in reverse_columns]
    else:
        # No renaming, so original == renamed
        for orig_col in all_gradient_cols:
            renamed_gradient_cols[orig_col] = orig_col

    # Build GridOptions
    gb = GridOptionsBuilder.from_dataframe(display_df)

    # Configure each column
    total_cols = len(display_df.columns)
    for idx, col in enumerate(display_df.columns):
        col_config = {}

        # Get original column name (before rename)
        original_col = col
        if column_names:
            # Find original name if this is a renamed column
            for orig, display in column_names.items():
                if display == col:
                    original_col = orig
                    break

        # Get responsive width configuration
        is_first = (idx == 0)
        width_config = _get_responsive_width(col, is_first, total_cols)
        col_config.update(width_config)

        # Apply number formatting
        if original_col in format_specs:
            format_str = format_specs[original_col]
            if '.0f' in format_str:
                col_config['type'] = 'numericColumn'
                col_config['valueFormatter'] = JsCode(f"(params) => params.value !== null && params.value !== undefined ? params.value.toFixed(0) : '—'")
            elif '.1f' in format_str or '.2f' in format_str:
                decimals = 2 if '.2f' in format_str else 1
                col_config['type'] = 'numericColumn'
                col_config['valueFormatter'] = JsCode(f"(params) => params.value !== null && params.value !== undefined ? params.value.toFixed({decimals}) : '—'")

        # Enable sorting
        col_config['sortable'] = True
        col_config['filter'] = False

        # Apply gradient coloring via cellStyle
        if original_col in gradients:
            is_reverse = gradients[original_col]['reverse']
            col_min = gradients[original_col]['min']
            col_max = gradients[original_col]['max']

            if gradient_mode == "row":
                # Get all gradient column names (renamed versions)
                display_gradient_cols = [renamed_gradient_cols[orig] for orig in all_gradient_cols
                                         if renamed_gradient_cols[orig] in display_df.columns]

                # Create JsCode for row-based cell styling with blue-green gradient
                cell_style_js = f"""
                function(params) {{
                    if (params.value === null || params.value === undefined || isNaN(params.value)) {{
                        return {{}};
                    }}

                    var value = parseFloat(params.value);

                    // Calculate min/max across all gradient columns in this row
                    var rowValues = [];
                    var gradientCols = {display_gradient_cols};

                    for (var i = 0; i < gradientCols.length; i++) {{
                        var colName = gradientCols[i];
                        var cellValue = params.data[colName];
                        if (cellValue !== null && cellValue !== undefined && !isNaN(cellValue)) {{
                            rowValues.push(parseFloat(cellValue));
                        }}
                    }}

                    if (rowValues.length === 0) {{
                        return {{}};
                    }}

                    var min = Math.min(...rowValues);
                    var max = Math.max(...rowValues);

                    if (min === max) {{
                        return {{}};
                    }}

                    var normalized = (value - min) / (max - min);
                    {'normalized = 1 - normalized;' if is_reverse else ''}

                    // Blue-green gradient with better contrast
                    var r, g, b;
                    if (normalized < 0.33) {{
                        // Light blue to medium blue
                        var t = normalized * 3;
                        r = Math.floor(220 - (100 * t));  // 220 → 120
                        g = Math.floor(240 - (70 * t));   // 240 → 170
                        b = Math.floor(255 - (30 * t));   // 255 → 225
                    }} else if (normalized < 0.67) {{
                        // Medium blue to teal
                        var t = (normalized - 0.33) * 3;
                        r = Math.floor(120 - (70 * t));   // 120 → 50
                        g = Math.floor(170 - (30 * t));   // 170 → 140
                        b = Math.floor(225 - (45 * t));   // 225 → 180
                    }} else {{
                        // Teal to dark green
                        var t = (normalized - 0.67) * 3;
                        r = Math.floor(50 - (30 * t));    // 50 → 20
                        g = Math.floor(140 - (20 * t));   // 140 → 120
                        b = Math.floor(180 - (60 * t));   // 180 → 120
                    }}

                    // Calculate text color based on background brightness
                    var brightness = (r * 299 + g * 587 + b * 114) / 1000;
                    var textColor = brightness < 130 ? '#ffffff' : '#000000';

                    return {{
                        'backgroundColor': 'rgb(' + r + ',' + g + ',' + b + ')',
                        'color': textColor,
                        'fontWeight': '500'
                    }};
                }}
                """
            else:  # gradient_mode == "column"
                # Create JsCode for column-based cell styling
                cell_style_js = f"""
                function(params) {{
                    if (params.value === null || params.value === undefined || isNaN(params.value)) {{
                        return {{}};
                    }}

                    var value = parseFloat(params.value);
                    var min = {col_min};
                    var max = {col_max};

                    if (min === max) {{
                        return {{}};
                    }}

                    var normalized = (value - min) / (max - min);
                    {'normalized = 1 - normalized;' if is_reverse else ''}

                    // Blue-green gradient with better contrast
                    var r, g, b;
                    if (normalized < 0.33) {{
                        // Light blue to medium blue
                        var t = normalized * 3;
                        r = Math.floor(220 - (100 * t));  // 220 → 120
                        g = Math.floor(240 - (70 * t));   // 240 → 170
                        b = Math.floor(255 - (30 * t));   // 255 → 225
                    }} else if (normalized < 0.67) {{
                        // Medium blue to teal
                        var t = (normalized - 0.33) * 3;
                        r = Math.floor(120 - (70 * t));   // 120 → 50
                        g = Math.floor(170 - (30 * t));   // 170 → 140
                        b = Math.floor(225 - (45 * t));   // 225 → 180
                    }} else {{
                        // Teal to dark green
                        var t = (normalized - 0.67) * 3;
                        r = Math.floor(50 - (30 * t));    // 50 → 20
                        g = Math.floor(140 - (20 * t));   // 140 → 120
                        b = Math.floor(180 - (60 * t));   // 180 → 120
                    }}

                    // Calculate text color based on background brightness
                    var brightness = (r * 299 + g * 587 + b * 114) / 1000;
                    var textColor = brightness < 130 ? '#ffffff' : '#000000';

                    return {{
                        'backgroundColor': 'rgb(' + r + ',' + g + ',' + b + ')',
                        'color': textColor,
                        'fontWeight': '500'
                    }};
                }}
                """
            col_config['cellStyle'] = JsCode(cell_style_js)

        gb.configure_column(col, **col_config)

    # Configure default column settings
    gb.configure_default_column(
        resizable=True,
        filterable=False,
        sortable=True,
        editable=False
    )

    # Configure grid options - compact for better mobile experience
    gb.configure_grid_options(
        domLayout='normal',
        enableCellTextSelection=True,
        rowHeight=36,  # More compact rows
        headerHeight=38,  # Slightly smaller header
        suppressColumnVirtualisation=True,
        suppressHorizontalScroll=False  # Allow horizontal scroll on mobile
    )

    # Theme-adaptive CSS for AgGrid - optimized for mobile & desktop
    custom_css = {
        ".ag-header-cell": {
            "font-weight": "600",
            "font-size": "0.85em",
            "padding": "4px 6px",
            "white-space": "nowrap",
            "overflow": "hidden",
            "text-overflow": "ellipsis"
        },
        ".ag-cell": {
            "font-size": "0.85em",
            "padding": "4px 6px"
        },
        ".ag-root-wrapper": {
            "border-radius": "8px",
            "overflow": "hidden",
            "box-shadow": "0 2px 8px rgba(0,0,0,0.08)"
        },
        # Mobile optimizations
        "@media (max-width: 768px)": {
            ".ag-header-cell": {
                "font-size": "0.75em",
                "padding": "3px 4px"
            },
            ".ag-cell": {
                "font-size": "0.75em",
                "padding": "3px 4px"
            }
        }
    }

    grid_options = gb.build()

    # Calculate responsive height - more compact for mobile
    row_height = 36
    header_height = 40
    table_height = min(550, header_height + len(display_df) * row_height + 10)

    # Render AgGrid
    AgGrid(
        display_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        custom_css=custom_css,
        height=table_height,
        theme='streamlit',  # Uses Streamlit's theme (auto light/dark)
        fit_columns_on_grid_load=True  # Better for responsive layouts
    )
