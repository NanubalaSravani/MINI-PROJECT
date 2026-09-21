# ================================================================
# FILTERS
# Shared main-page filter panel.
# ================================================================
# Every dashboard calls the same panel, so the filters look and
# behave the same on all five pages and no page writes its own
# selectbox code.

import streamlit as st

from src.styling import filter_bar_header


def filter_options(data):
    """Read the choices for each filter box out of the data.

    Months come back in calendar order, taken from the date table
    rather than sorted alphabetically.
    """

    months = (
        data[["MonthNum", "Month"]]
        .drop_duplicates()
        .sort_values("MonthNum")["Month"]
        .tolist()
    )

    return {
        "State": sorted(data["State"].dropna().unique().tolist()),
        "Year": sorted(data["Year"].dropna().unique().tolist()),
        "Month": months,
        "Program": sorted(data["Program"].dropna().unique().tolist())
    }


def _clear_filters(labels):
    for label in labels:
        st.session_state[f"filter_{label}"] = "All"


def get_current_filters(data, sanitize=True):
    """Return current program filters without drawing widgets.

    ``sanitize`` is used before widgets are created. After a widget has been
    instantiated, Streamlit does not allow its session_state key to be
    assigned, so callers that are already inside the rendered filter bar use
    ``sanitize=False``.
    """
    options = filter_options(data)
    for label in options:
        st.session_state.setdefault(f"filter_{label}", "All")
        allowed = ["All"] + options[label]
        if sanitize and st.session_state[f"filter_{label}"] not in allowed:
            st.session_state[f"filter_{label}"] = "All"
    return {label: st.session_state[f"filter_{label}"] for label in options}


def _render_active_summary(selected):
    active = []
    for label, value in selected.items():
        if value != "All":
            active.append(
                f'<span class="active-filter-chip">{label}: {value}</span>'
            )
    if not active:
        active.append('<span class="active-filter-chip">All available data</span>')
    st.markdown(
        '<div class="active-filter-summary"><span class="summary-label">Active filters:</span>'
        + "".join(active) + "</div>",
        unsafe_allow_html=True,
    )


def render_filter_bar(data, note=None):
    """Draw the program filter bar below the KPI cards."""
    options = filter_options(data)
    with st.container(border=True):
        header_col, reset_col = st.columns([5, 1])
        with header_col:
            filter_bar_header(note or "Selections are applied to every visual on this page")
        with reset_col:
            st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
            st.button(
                "♻️ Reset",
                use_container_width=True,
                on_click=_clear_filters,
                args=(list(options.keys()),),
                key="programs_reset_filters",
            )

        cols = st.columns(len(options), gap="medium")
        for col, (label, choices) in zip(cols, options.items()):
            with col:
                st.selectbox(label, options=["All"] + choices, key=f"filter_{label}")

        # The widgets above are already instantiated. Read their values
        # without sanitizing/mutating session_state.
        selected = get_current_filters(data, sanitize=False)
        _render_active_summary(selected)

    return selected


def sidebar_filters(data, note=None):
    # Backward-compatible alias; the filters are intentionally rendered in
    # the main content area rather than the sidebar.
    return render_filter_bar(data, note=note)

def apply_filters(data, selected):
    """Return only the rows matching every chosen filter."""

    filtered = data.copy()

    for column, value in selected.items():

        if value != "All":

            filtered = filtered[filtered[column] == value]

    return filtered


def population_base(data):
    """One row per state and month.

    Children / Adults / Elderly are recorded once per state per
    month and then repeated on every programme row. Summing the raw
    rows would multiply the population by the number of programmes,
    so the age-group figures are always taken from this
    de-duplicated frame.

    Every other measure, including the immunisation count, does
    vary by programme and is summed from the filtered rows instead.
    """

    return data.drop_duplicates(subset=["State", "Year", "Month"])
