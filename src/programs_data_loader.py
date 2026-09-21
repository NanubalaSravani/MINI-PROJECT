# ================================================================
# DATA LOADING
# ================================================================
# Every CSV lives in the data folder at the top of the project, so
# the path is built from this file's location and never depends on
# the folder the app was started from.

from pathlib import Path

import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


PROGRAMS_FILES = [
    "fact_health_programs.csv",
    "dim_program.csv",
    "dim_state.csv",
    "dim_date.csv"
]


# The dashboard uses friendly display names. The raw column names
# from the CSVs are mapped across here.

PROGRAMS_COLUMN_NAMES = {
    "state_name": "State",
    "year": "Year",
    "month_name": "Month",
    "month_num": "MonthNum",
    "program_name": "Program",
    "program_coverage_pct": "ProgramCoverage",
    "people_screened": "PeopleScreened",
    "beneficiaries_reached": "BeneficiariesReached",
    "maternal_health_beneficiaries": "MaternalHealthBeneficiaries",
    "child_immunization_count": "ChildImmunization",
    "high_risk_individuals": "HighRiskIndividuals",
    "chronic_disease_patients": "ChronicDiseasePatients",
    "health_vulnerability_index": "HealthVulnerabilityIndex",
    "socioeconomic_score": "SocioeconomicScore",
    "children_population": "Children",
    "adults_population": "Adults",
    "elderly_population": "Elderly"
}


PROGRAMS_NUMBER_COLUMNS = [
    "ProgramCoverage",
    "PeopleScreened",
    "BeneficiariesReached",
    "MaternalHealthBeneficiaries",
    "ChildImmunization",
    "HighRiskIndividuals",
    "ChronicDiseasePatients",
    "HealthVulnerabilityIndex",
    "SocioeconomicScore",
    "Children",
    "Adults",
    "Elderly"
]


def missing_files(file_names):
    """Return the names of any required CSV that is not in data."""

    return [
        name for name in file_names
        if not (DATA_DIR / name).exists()
    ]


@st.cache_data
def get_programs_master():
    """Load, clean and join the health programs star schema.

    Returns the joined frame, the number of rows dropped for a
    negative count, and the number of exact duplicate fact rows.
    """

    fact = pd.read_csv(DATA_DIR / "fact_health_programs.csv")
    program = pd.read_csv(DATA_DIR / "dim_program.csv")
    state = pd.read_csv(DATA_DIR / "dim_state.csv")
    date = pd.read_csv(DATA_DIR / "dim_date.csv")

    # The raw fact file contains exact duplicate rows.

    duplicate_rows = int(fact.duplicated().sum())

    fact = fact.drop_duplicates()

    # Each dimension must have only one row per id.
    # If an id repeats, the merge would create extra fact rows
    # and every total on the dashboard would come out too high.

    program = program.drop_duplicates(subset=["program_id"])
    state = state.drop_duplicates(subset=["state_id"])
    date = date.drop_duplicates(subset=["date_id"])

    # Join the star schema.

    data = (
        fact
        .merge(program, on="program_id", how="left")
        .merge(state, on="state_id", how="left")
        .merge(date, on="date_id", how="left")
    )

    # Rename to the display names used throughout the dashboard.

    data = data.rename(columns=PROGRAMS_COLUMN_NAMES)

    # Make sure the number columns are really numbers.
    # Any blank or text value becomes NaN instead of breaking a chart.

    for column in PROGRAMS_NUMBER_COLUMNS:

        if column in data.columns:

            data[column] = pd.to_numeric(data[column], errors="coerce")

    # A negative number of people is not possible, so those rows go.

    negative_rows = int((data["HighRiskIndividuals"] < 0).sum())

    data = data[data["HighRiskIndividuals"] >= 0]

    # Year as text so the line chart gives each year its own colour.

    data["Year"] = data["Year"].astype(str)

    data = data.reset_index(drop=True)

    return data, negative_rows, duplicate_rows
