"""
data_loader.py
----------------
Centralized data access layer for the Public Health Analytics Dashboard.

All raw CSV extracts are loaded and joined here so that page modules never
touch the filesystem directly. Streamlit's cache decorators are used to
avoid re-reading / re-joining data on every widget interaction.

Author : Analytics Engineering Team
Module : Public Health Surveillance Dashboard
"""

from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------- #
# Raw dimension / fact loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=3600)
def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Read every cleaned CSV extract into memory once and cache the result."""
    files = {
        "dim_dates": "dim_dates_cleaned.csv",
        "dim_disease": "dim_disease_cleaned.csv",
        "dim_program": "dim_program_cleaned.csv",
        "dim_source": "dim_source_cleaned.csv",
        "dim_state": "dim_state_cleaned.csv",
        "fact_outbreak": "fact_outbreak_cleaned.csv",
        "fact_surveillance": "fact_disease_surveillance_cleaned.csv",
        "fact_environmental": "fact_environmental_cleaned.csv",
        "fact_health_programs": "fact_health_programs_cleaned.csv",
        "fact_lab_healthcare": "fact_lab_healthcare_cleaned.csv",
    }
    tables = {}
    for key, filename in files.items():
        path = DATA_DIR / filename
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        tables[key] = df

    # Parse the date dimension once, up front.
    tables["dim_dates"]["full_date"] = pd.to_datetime(tables["dim_dates"]["full_date"])
    return tables


# --------------------------------------------------------------------------- #
# Star-schema joins -> flat analytical tables (one per subject area)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=3600)
def get_surveillance_master() -> pd.DataFrame:
    """Disease surveillance fact joined with date / state / disease / source dims."""
    t = load_raw_tables()
    df = (
        t["fact_surveillance"]
        .merge(t["dim_dates"], on="date_id", how="left")
        .merge(t["dim_state"], on="state_id", how="left")
        .merge(t["dim_disease"], on="disease_id", how="left")
        .merge(t["dim_source"], on="source_id", how="left")
    )
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_outbreak_master() -> pd.DataFrame:
    """Outbreak fact joined with date / state / disease / source dims."""
    t = load_raw_tables()
    df = (
        t["fact_outbreak"]
        .merge(t["dim_dates"], on="date_id", how="left")
        .merge(t["dim_state"], on="state_id", how="left")
        .merge(t["dim_disease"], on="disease_id", how="left")
        .merge(t["dim_source"], on="source_id", how="left")
    )
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_environmental_master() -> pd.DataFrame:
    t = load_raw_tables()
    df = t["fact_environmental"].merge(t["dim_dates"], on="date_id", how="left").merge(
        t["dim_state"], on="state_id", how="left"
    )
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_programs_master() -> pd.DataFrame:
    t = load_raw_tables()
    df = (
        t["fact_health_programs"]
        .merge(t["dim_dates"], on="date_id", how="left")
        .merge(t["dim_state"], on="state_id", how="left")
        .merge(t["dim_program"], on="program_id", how="left")
    )
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_lab_master() -> pd.DataFrame:
    t = load_raw_tables()
    df = t["fact_lab_healthcare"].merge(t["dim_dates"], on="date_id", how="left").merge(
        t["dim_state"], on="state_id", how="left"
    )
    return df


# --------------------------------------------------------------------------- #
# Filter option helpers (used to populate sidebar widgets)
# --------------------------------------------------------------------------- #
def get_filter_options() -> dict:
    t = load_raw_tables()
    dates = t["dim_dates"]
    return {
        "states": sorted(t["dim_state"]["state_name"].dropna().unique().tolist()),
        "regions": sorted(t["dim_state"]["region"].dropna().unique().tolist()),
        "years": sorted(dates["year"].dropna().unique().tolist()),
        "months": list(
            dates.sort_values("month_num")["month_name"].drop_duplicates().values
        ),
        "diseases": sorted(t["dim_disease"]["disease_name"].dropna().unique().tolist()),
        "disease_categories": sorted(
            t["dim_disease"]["disease_category"].dropna().unique().tolist()
        ),
        "sources": sorted(t["dim_source"]["source_name"].dropna().unique().tolist()),
    }


def apply_common_filters(
    df: pd.DataFrame,
    states: list | None = None,
    regions: list | None = None,
    years: list | None = None,
    months: list | None = None,
    diseases: list | None = None,
    disease_categories: list | None = None,
    sources: list | None = None,
) -> pd.DataFrame:
    """Apply the sidebar filter selections to any joined master table."""
    out = df.copy()
    if states:
        out = out[out["state_name"].isin(states)]
    if regions and "region" in out.columns:
        out = out[out["region"].isin(regions)]
    if years:
        out = out[out["year"].isin(years)]
    if months:
        out = out[out["month_name"].isin(months)]
    if diseases and "disease_name" in out.columns:
        out = out[out["disease_name"].isin(diseases)]
    if disease_categories and "disease_category" in out.columns:
        out = out[out["disease_category"].isin(disease_categories)]
    if sources and "source_name" in out.columns:
        out = out[out["source_name"].isin(sources)]
    return out
