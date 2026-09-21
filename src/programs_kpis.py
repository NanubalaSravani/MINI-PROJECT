# ================================================================
# KPI CALCULATION AND FORMATTING
# ================================================================
# A figure such as 10,347,642,180 does not fit inside a KPI card and
# was being cut off with three dots. Large counts are therefore shown
# in the short form used in reports (10.35B) and the exact figure is
# printed underneath in small text, so nothing is hidden from the
# reader.

import pandas as pd


def compact_number(value):
    """Return a short, report-style version of a large count."""

    if pd.isna(value):
        return "N/A"

    value = float(value)

    for size, suffix in (
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K")
    ):

        if abs(value) >= size:

            return f"{value / size:,.2f}{suffix}"

    return f"{value:,.0f}"


def exact_number(value):
    """Return the full figure with thousand separators."""

    if pd.isna(value):
        return "No data"

    return f"{float(value):,.0f}"


def kpi_count(column, label, total):
    """Draw one count KPI: short value, exact figure underneath."""

    column.metric(
        label,
        compact_number(total),
        delta=exact_number(total),
        delta_color="off",
        help=f"Exact value: {exact_number(total)}"
    )


def kpi_rate(column, label, value, suffix="", decimals=1):
    """Draw one percentage or index KPI."""

    if pd.isna(value):

        column.metric(label, "N/A")

        return

    column.metric(
        label,
        f"{value:,.{decimals}f}{suffix}"
    )
