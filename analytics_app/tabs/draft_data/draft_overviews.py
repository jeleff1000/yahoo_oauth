import streamlit as st
import pandas as pd


@st.fragment
def display_draft_overview(draft_data):
    st.header("Average Draft Prices")

    # Standardize column names to snake_case
    draft_data = draft_data.rename(
        columns={
            "Year": "year",
            "Team Manager": "manager",
            "Primary Position": "yahoo_position",
            "Cost": "cost",
            "Is Keeper Status": "is_keeper_status",
        }
    )

    draft_data["year"] = draft_data["year"].astype(str)
    allowed_yahoo_positions = ["QB", "RB", "WR", "TE", "DEF", "K"]
    position_order = pd.CategoricalDtype(allowed_yahoo_positions, ordered=True)

    if "yahoo_position" not in draft_data.columns:
        st.error("The 'yahoo_position' column is missing from draft_data.")
        return

    years = sorted(draft_data["year"].unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox(
            "Start Year", years, index=0, key="start_year_selectbox"
        )
    with col2:
        end_year = st.selectbox(
            "End Year", years, index=len(years) - 1, key="end_year_selectbox"
        )

    # Non-keeper filter (is_keeper_status != 1)
    non_keeper_data = draft_data[
        (draft_data["year"] >= start_year)
        & (draft_data["year"] <= end_year)
        & (draft_data["manager"].notnull())
        & (draft_data["manager"].str.strip() != "")
        & (draft_data["yahoo_position"].isin(allowed_yahoo_positions))
        & (draft_data["cost"] > 0)
        & (
            (draft_data["is_keeper_status"].isnull())
            | (draft_data["is_keeper_status"] != 1)
        )
    ].copy()
    non_keeper_data["yahoo_position"] = non_keeper_data["yahoo_position"].astype(
        position_order
    )
    non_keeper_data["rank_num"] = (
        non_keeper_data.groupby(["year", "yahoo_position"])["cost"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    agg_dict = {"cost": ["mean", "max", "min", "median", "count"]}

    # Add SPAR metrics if available
    if "spar" in non_keeper_data.columns:
        agg_dict["spar"] = ["mean", "median"]

    agg_df = (
        non_keeper_data.groupby(["yahoo_position", "rank_num"])
        .agg(agg_dict)
        .reset_index()
    )

    # Flatten column names
    if "spar" in non_keeper_data.columns:
        agg_df.columns = [
            "yahoo_position",
            "rank_num",
            "avg_cost",
            "max_cost",
            "min_cost",
            "median_cost",
            "times_drafted",
            "avg_spar",
            "median_spar",
        ]
        # Calculate SPAR efficiency
        agg_df["spar_per_dollar"] = (agg_df["avg_spar"] / agg_df["avg_cost"]).round(2)
    else:
        agg_df.columns = [
            "yahoo_position",
            "rank_num",
            "avg_cost",
            "max_cost",
            "min_cost",
            "median_cost",
            "times_drafted",
        ]
    agg_df["position_rank"] = agg_df["yahoo_position"].astype(str) + agg_df[
        "rank_num"
    ].astype(str)
    agg_df["yahoo_position"] = agg_df["yahoo_position"].astype(position_order)
    agg_df = agg_df.sort_values(["yahoo_position", "rank_num"])
    agg_df = agg_df[agg_df["times_drafted"] >= 1]

    # Display columns based on available metrics
    columns_to_display = [
        "position_rank",
        "yahoo_position",
        "avg_cost",
        "max_cost",
        "min_cost",
        "median_cost",
        "times_drafted",
    ]
    if "avg_spar" in agg_df.columns:
        columns_to_display.extend(["avg_spar", "median_spar", "spar_per_dollar"])

    st.subheader("Average Drafted Player Prices & Value")
    st.dataframe(
        agg_df[columns_to_display],
        hide_index=True,
        column_config=(
            {
                "avg_spar": st.column_config.NumberColumn("Avg SPAR", format="%.1f"),
                "median_spar": st.column_config.NumberColumn(
                    "Median SPAR", format="%.1f"
                ),
                "spar_per_dollar": st.column_config.NumberColumn(
                    "SPAR/$", format="%.2f", help="Average SPAR per dollar spent"
                ),
            }
            if "avg_spar" in agg_df.columns
            else None
        ),
    )

    # Keeper filter (is_keeper_status == 1)
    keeper_data = draft_data[
        (draft_data["year"] >= start_year)
        & (draft_data["year"] <= end_year)
        & (draft_data["manager"].notnull())
        & (draft_data["manager"].str.strip() != "")
        & (draft_data["yahoo_position"].isin(allowed_yahoo_positions))
        & (draft_data["cost"] > 0)
        & (draft_data["is_keeper_status"] == 1)
    ].copy()

    if not keeper_data.empty:
        keeper_data["yahoo_position"] = keeper_data["yahoo_position"].astype(
            position_order
        )
        keeper_data["rank_num"] = (
            keeper_data.groupby(["year", "yahoo_position"])["cost"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        # Build aggregation dict for keepers
        keeper_agg_dict = {"cost": ["mean", "max", "min", "median", "count"]}

        # Add SPAR metrics if available
        if "spar" in keeper_data.columns:
            keeper_agg_dict["spar"] = ["mean", "median"]

        keeper_agg = (
            keeper_data.groupby(["yahoo_position", "rank_num"])
            .agg(keeper_agg_dict)
            .reset_index()
        )

        # Flatten column names
        if "spar" in keeper_data.columns:
            keeper_agg.columns = [
                "yahoo_position",
                "rank_num",
                "avg_cost",
                "max_cost",
                "min_cost",
                "median_cost",
                "times_kept",
                "avg_spar",
                "median_spar",
            ]
            # Calculate SPAR efficiency
            keeper_agg["spar_per_dollar"] = (
                keeper_agg["avg_spar"] / keeper_agg["avg_cost"]
            ).round(2)
        else:
            keeper_agg.columns = [
                "yahoo_position",
                "rank_num",
                "avg_cost",
                "max_cost",
                "min_cost",
                "median_cost",
                "times_kept",
            ]

        keeper_agg["position_rank"] = keeper_agg["yahoo_position"].astype(
            str
        ) + keeper_agg["rank_num"].astype(str)
        keeper_agg["yahoo_position"] = keeper_agg["yahoo_position"].astype(
            position_order
        )
        keeper_agg = keeper_agg.sort_values(["yahoo_position", "rank_num"])

        # Display columns based on available metrics
        keeper_columns = [
            "position_rank",
            "yahoo_position",
            "avg_cost",
            "max_cost",
            "min_cost",
            "median_cost",
            "times_kept",
        ]
        if "avg_spar" in keeper_agg.columns:
            keeper_columns.extend(["avg_spar", "median_spar", "spar_per_dollar"])

        st.subheader("Average Keeper Prices & Value")
        st.dataframe(
            keeper_agg[keeper_columns],
            hide_index=True,
            column_config=(
                {
                    "avg_spar": st.column_config.NumberColumn(
                        "Avg SPAR", format="%.1f"
                    ),
                    "median_spar": st.column_config.NumberColumn(
                        "Median SPAR", format="%.1f"
                    ),
                    "spar_per_dollar": st.column_config.NumberColumn(
                        "SPAR/$", format="%.2f", help="Average SPAR per dollar spent"
                    ),
                }
                if "avg_spar" in keeper_agg.columns
                else None
            ),
        )
    else:
        st.info("No keeper data found for the selected year range.")
