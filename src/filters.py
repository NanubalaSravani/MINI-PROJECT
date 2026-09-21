"""
filters.py
----------
Shared main-page filters for the public-health dashboards.

The filter widgets are rendered below the KPI cards on the Executive dashboard,
while the selected values live in session_state so the current filter context is
available before the widgets are drawn. This keeps KPI calculations and charts
synchronized with the visible filter bar after every Streamlit rerun.
"""

import html
import streamlit as st
from src.data_loader import get_filter_options
from src.styling import filter_bar_header


FILTER_KEYS = [
    "f_region", "f_state", "f_year", "f_month",
    "f_disease_cat", "f_disease", "f_source",
]


def _clear_filters():
    for key in FILTER_KEYS:
        st.session_state[key] = []


def _ensure_filter_state(sanitize=True):
    options = get_filter_options()
    for key in FILTER_KEYS:
        st.session_state.setdefault(key, [])
    # Remove stale selections only before widgets are instantiated. Streamlit
    # forbids changing a widget's session_state key after that widget exists.
    option_map = {
        "f_region": options["regions"],
        "f_state": options["states"],
        "f_year": options["years"],
        "f_month": options["months"],
        "f_disease_cat": options["disease_categories"],
        "f_disease": options["diseases"],
        "f_source": options["sources"],
    }
    if sanitize:
        for key, allowed in option_map.items():
            cleaned = [v for v in st.session_state[key] if v in allowed]
            if cleaned != st.session_state[key]:
                st.session_state[key] = cleaned
    return options


def get_current_filters() -> dict:
    """Return the current filter state without rendering the widgets."""
    _ensure_filter_state(sanitize=True)
    return {
        "regions": list(st.session_state["f_region"]),
        "states": list(st.session_state["f_state"]),
        "years": list(st.session_state["f_year"]),
        "months": list(st.session_state["f_month"]),
        "disease_categories": list(st.session_state["f_disease_cat"]),
        "diseases": list(st.session_state["f_disease"]),
        "sources": list(st.session_state["f_source"]),
    }


def _render_active_summary(filters: dict):
    labels = [
        ("Region", filters["regions"]),
        ("State", filters["states"]),
        ("Year", filters["years"]),
        ("Month", filters["months"]),
        ("Disease Type", filters["disease_categories"]),
        ("Disease", filters["diseases"]),
        ("Primary Source", filters["sources"]),
    ]
    active = []
    for label, values in labels:
        if values:
            value_text = ", ".join(str(v) for v in values)
            active.append(
                f'<span class="active-filter-chip">{html.escape(label)}: {html.escape(value_text)}</span>'
            )
    if not active:
        active.append('<span class="active-filter-chip">All available data</span>')
    st.markdown(
        '<div class="active-filter-summary"><span class="summary-label">Active filters:</span>'
        + "".join(active) + "</div>",
        unsafe_allow_html=True,
    )


def render_filter_bar() -> dict:
    """Render the shared filter bar and return its current selections."""
    options = _ensure_filter_state()
    with st.container(border=True):
        header_col, reset_col = st.columns([5, 1])
        with header_col:
            filter_bar_header("Selections are applied to every visual on this page")
        with reset_col:
            st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
            st.button(
                "♻️ Reset",
                use_container_width=True,
                on_click=_clear_filters,
                key="reset_exec_filters",
            )

        row1 = st.columns(4, gap="medium")
        with row1[0]:
            st.multiselect("Region", options["regions"], key="f_region")
        with row1[1]:
            st.multiselect("State", options["states"], key="f_state")
        with row1[2]:
            st.multiselect("Year", options["years"], key="f_year")
        with row1[3]:
            st.multiselect("Month", options["months"], key="f_month")

        row2 = st.columns(3, gap="medium")
        with row2[0]:
            st.multiselect("Disease Type", options["disease_categories"], key="f_disease_cat")
        with row2[1]:
            st.multiselect("Disease", options["diseases"], key="f_disease")
        with row2[2]:
            st.multiselect("Primary Source", options["sources"], key="f_source")

        # IMPORTANT: do not call get_current_filters() here. That helper
        # sanitizes session_state, and these widget keys have already been
        # instantiated above. Mutating them at this point triggers:
        # StreamlitAPIException: session_state.<key> cannot be modified
        # after the widget with key <key> is instantiated.
        current = {
            "regions": list(st.session_state["f_region"]),
            "states": list(st.session_state["f_state"]),
            "years": list(st.session_state["f_year"]),
            "months": list(st.session_state["f_month"]),
            "disease_categories": list(st.session_state["f_disease_cat"]),
            "diseases": list(st.session_state["f_disease"]),
            "sources": list(st.session_state["f_source"]),
        }
        _render_active_summary(current)

    return current


# Backward-compatible name used by older page code.
def render_sidebar_filters() -> dict:
    return render_filter_bar()
