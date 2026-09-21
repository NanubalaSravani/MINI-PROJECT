"""
kpis.py
-------
Pure functions that turn a filtered surveillance dataframe into the KPI
values shown at the top of each dashboard page. Kept separate from the
Streamlit rendering code so the calculations can be unit tested in isolation.
"""

import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    """Return the standard Executive Overview / Disease Surveillance KPI set."""
    if df.empty:
        return {
            "population_under_surveillance": 0,
            "total_reported_cases": 0,
            "active_cases": 0,
            "recovered_cases": 0,
            "deaths": 0,
            "case_fatality_rate": 0.0,
            "recovery_rate": 0.0,
            "public_health_risk_score": 0.0,
        }

    total_cases = df["total_reported_cases"].sum()
    active = df["active_cases"].sum()
    recovered = df["recovered_cases"].sum()
    deaths = df["deaths"].sum()

    # Recompute CFR / Recovery Rate from aggregated totals (more accurate than
    # averaging pre-computed row-level percentages across mixed populations).
    cfr = (deaths / total_cases * 100) if total_cases else 0.0
    recovery_rate = (recovered / total_cases * 100) if total_cases else 0.0

    # Population under surveillance should not double count the same
    # state/date/disease row multiple times if duplicated; take distinct sum
    # per state+date since it's a snapshot metric, not additive across disease rows.
    pop = (
        df.drop_duplicates(subset=["state_id", "date_id"])["population_under_surveillance"]
        .sum()
    )

    return {
        "population_under_surveillance": pop,
        "total_reported_cases": total_cases,
        "active_cases": active,
        "recovered_cases": recovered,
        "deaths": deaths,
        "case_fatality_rate": round(cfr, 2),
        "recovery_rate": round(recovery_rate, 2),
        "public_health_risk_score": round(df["public_health_risk_score"].mean(), 1),
    }


def compute_hosp_icu_rates(df: pd.DataFrame) -> dict:
    if df.empty or df["total_reported_cases"].sum() == 0:
        return {"hospitalization_rate": 0.0, "icu_rate": 0.0}
    total = df["total_reported_cases"].sum()
    hosp_rate = df["hospitalized_cases"].sum() / total * 100
    icu_rate = df["icu_admissions"].sum() / total * 100
    return {"hospitalization_rate": round(hosp_rate, 2), "icu_rate": round(icu_rate, 2)}


def format_number(value: float) -> str:
    """Human friendly K / M / B suffixing for KPI tiles."""
    value = float(value)
    for unit, threshold in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= threshold:
            return f"{value / threshold:,.2f}{unit}"
    return f"{value:,.0f}"
