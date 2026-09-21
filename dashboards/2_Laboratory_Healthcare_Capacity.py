"""
2_Laboratory_Healthcare_Capacity.py
--------------------------------------
Laboratory & Healthcare Capacity dashboard — testing throughput, vaccination
progress, hospital/ICU capacity, lab performance, and bed occupancy.

Uses the shared data loader (src.data_loader.get_lab_master) and shared
styling (src.styling) so this page matches the rest of the suite, while
keeping its own purpose-built KPI-delta cards, gauge, and filters.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import get_lab_master
from src.report_generator import add_sidebar_report_button
from src.styling import (
    inject_css, page_header, kpi_card_delta, snapshot_row, section_title,
    filter_bar_header, insight_banner, RISK_HEATMAP_SCALE, BAR_SEQUENCE,
)

# Note: st.set_page_config() is intentionally NOT called here — app.py
# already calls it once, centrally, before st.navigation(). Calling it again
# from a page script throws StreamlitAPIException on real navigation (this
# was a live bug here previously; AppTest's switch_page() doesn't catch it
# since it resets state differently than an actual browser session).

# DASHBOARD COLOR PALETTE
# (kept local since a couple of names below — GREEN, AMBER, SLATE_BLUE — are
# used verbatim through the chart code; values match src/styling.py exactly
# so this page stays visually identical to the rest of the suite)

PRIMARY_NAVY = "#17324D"
TEAL = "#0F6B78"
# Plain single-series bars now use the shared executive blue/gray/teal
# sequence (BAR_SEQUENCE[0] = "#1F77B4") instead of the page's own navy, so
# every bar chart in the suite draws from the same clean palette.
SLATE_BLUE = BAR_SEQUENCE[0]
GREEN = "#16855B"
AMBER = "#C98A00"

WHITE = "#FFFFFF"
TEXT = "#000000"
SECONDARY_TEXT = "#000000"
BORDER = "rgba(100, 116, 139, 0.22)"

inject_css()

# Supplemental CSS just for native st.metric() KPI cards and the gauge/
# expander containers used on this page — the shared inject_css() doesn't
# style st.metric() at all (other pages use the custom kpi_card() component
# instead), so this is additive, not a conflicting override of the shared
# navy sidebar / page background / header styling.
st.markdown(f"""
<style>
div[data-testid="stMetric"] {{
    background-color: {WHITE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    box-shadow: 0 2px 8px rgba(18, 53, 91, 0.06) !important;
    min-height: 105px !important;
    box-sizing: border-box !important;
}}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] span {{
    color: {TEXT} !important;
    opacity: 1 !important;
    visibility: visible !important;
    -webkit-text-fill-color: {TEXT} !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    white-space: normal !important;
    word-break: break-word !important;
    line-height: 1.2 !important;
}}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] span {{
    color: {PRIMARY_NAVY} !important;
    opacity: 1 !important;
    visibility: visible !important;
    -webkit-text-fill-color: {PRIMARY_NAVY} !important;
    font-size: 25px !important;
    font-weight: 700 !important;
}}

div[data-testid="stMetric"] p {{
    opacity: 1 !important;
    color: {TEXT} !important;
}}

div[data-testid="stExpander"] {{
    background-color: {WHITE} !important;
    border: 1px solid {BORDER} !important;
    border-left: 4px solid {TEAL} !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(18, 53, 91, 0.05) !important;
    margin-top: 20px !important;
    margin-bottom: 20px !important;
    overflow: hidden !important;
}}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary * {{
    font-weight: 650 !important;
    color: {PRIMARY_NAVY} !important;
    font-size: 1.25rem !important;
}}

div[data-testid="stExpander"] summary:hover,
div[data-testid="stExpander"] summary:hover * {{
    color: {TEAL} !important;
}}
</style>
""", unsafe_allow_html=True)


# 1. LOAD DATA
# Reuses the shared loader (already joins fact_lab_healthcare with
# dim_dates + dim_state) instead of duplicating CSV-reading logic.

@st.cache_data
def load_data():
    df = get_lab_master()
    # Merges can upcast year to float (e.g. 2021.0); keep it a clean int
    df["year"] = df["year"].astype("Int64")
    return df


df = load_data()

# Subtitle: quick context on coverage of the underlying dataset
n_states = df["state_name"].nunique()
date_min = df["year_month"].min()
date_max = df["year_month"].max()

page_header(
    "Laboratory & Healthcare Capacity",
    f"Covering {n_states} states · {date_min} to {date_max}",
)


# 2. FILTER STATE

# Filters are rendered below the KPI cards. Their current values are kept in
# session_state so the data and KPI calculations can use the active context
# before the widgets are drawn on each rerun.
st.session_state.setdefault("state_filter", "All")
st.session_state.setdefault("year_filter", "All")
st.session_state.setdefault("month_filter", "All")

def reset_filters():
    st.session_state["state_filter"] = "All"
    st.session_state["year_filter"] = "All"
    st.session_state["month_filter"] = "All"

state_options = ["All"] + sorted(df["state_name"].dropna().unique().tolist())
year_options = ["All"] + sorted(df["year"].dropna().unique().tolist())
month_options = ["All"] + df[["month_num", "month_name"]].drop_duplicates().sort_values("month_num")["month_name"].tolist()

selected_state = st.session_state["state_filter"] if st.session_state["state_filter"] in state_options else "All"
selected_year = st.session_state["year_filter"] if st.session_state["year_filter"] in year_options else "All"
selected_month = st.session_state["month_filter"] if st.session_state["month_filter"] in month_options else "All"
st.session_state["state_filter"] = selected_state
st.session_state["year_filter"] = selected_year
st.session_state["month_filter"] = selected_month


# 4. APPLY FILTERS

filtered_df = df.copy()

if selected_state != "All":
    filtered_df = filtered_df[
        filtered_df["state_name"] == selected_state
    ]

if selected_year != "All":
    filtered_df = filtered_df[
        filtered_df["year"] == selected_year
    ]

if selected_month != "All":
    filtered_df = filtered_df[
        filtered_df["month_name"] == selected_month
    ]

# Guard against empty dataset
if filtered_df.empty:
    st.warning("⚠️ **No data available for the selected filter combination.** Please adjust or reset your State, Year, or Month selections.")
    st.stop()

add_sidebar_report_button(
    "Laboratory & Healthcare Capacity",
    filtered_df,
    {"State": selected_state, "Year": selected_year, "Month": selected_month},
)

# 5. KPI CALCULATIONS & DELTAS

total_tests = filtered_df["total_tests"].sum()

positive_tests = filtered_df["positive_tests"].sum()

if total_tests > 0:
    positivity_rate = (
        positive_tests / total_tests
    ) * 100
else:
    positivity_rate = 0

vaccination_coverage = (
    filtered_df["vaccination_coverage_pct"].mean()
)

booster_coverage = (
    filtered_df["booster_coverage_pct"].mean()
)

hospital_beds = (
    filtered_df["hospital_beds"].sum()
)

doctors = (
    filtered_df["doctors"].sum()
)

icu_utilization = (
    filtered_df["icu_utilization_pct"].mean()
)

reporting_compliance = (
    filtered_df["reporting_compliance_pct"].mean()
)

# Determine ICU Utilization Status & Colors
# >85: ALERT (red text), 50-85: MODERATE, <50: NORMAL
def icu_status_color(value):
    """Single source of truth for ICU status thresholds/colors —
    used by both the KPI card and the ICU gauge chart."""
    if value > 85:
        return "ALERT", "#D9381E", "rgba(217, 56, 30, 0.12)"        # Bold Red
    elif value >= 50:
        return "MODERATE", "#C98A00", "rgba(201, 138, 0, 0.12)"     # Amber / Orange
    else:
        return "NORMAL", "#16855B", "rgba(22, 133, 91, 0.12)"       # Green

icu_status, icu_color, icu_bg_color = icu_status_color(icu_utilization)

# Key point — single headline callout, matches the pattern used on the
# other dashboards (shown directly under the page header).
_lab_scope_text = selected_state if selected_state != "All" else "across all states"
insight_banner(
    f"Testing positivity rate is **{positivity_rate:.2f}%**; vaccination coverage stands at "
    f"**{vaccination_coverage:.2f}%**; ICU utilization is **{icu_status}** at **{icu_utilization:.2f}%** "
    f"({_lab_scope_text})."
)

# Config for Month-over-Month (MoM) KPI deltas:
# key -> (column, aggregation, kind)
#   kind="pct_of_prev"  -> % change relative to previous month's value (sums)
#   kind="pp_diff"      -> plain point difference (percentages/rates, means)
#   kind="ratio_pp"     -> point difference of a computed ratio (positivity rate)
_DELTA_METRICS = {
    "total_tests":          ("total_tests", "sum", "pct_of_prev"),
    "positive_tests":       ("positive_tests", "sum", "pct_of_prev"),
    "vaccination_coverage": ("vaccination_coverage_pct", "mean", "pp_diff"),
    "booster_coverage":     ("booster_coverage_pct", "mean", "pp_diff"),
    "hospital_beds":        ("hospital_beds", "sum", "pct_of_prev"),
    "doctors":               ("doctors", "sum", "pct_of_prev"),
    "icu_utilization":       ("icu_utilization_pct", "mean", "pp_diff"),
    "reporting_compliance":  ("reporting_compliance_pct", "mean", "pp_diff"),
}

def calculate_kpi_deltas(current_filtered, full_dataset, state_sel):
    """Compute Month-over-Month deltas for each KPI, comparing the latest
    month in the current filter selection to the prior month (within the
    same state scope, ignoring the year/month filters so there's always
    something to compare against)."""
    ref_df = full_dataset if state_sel == "All" else full_dataset[full_dataset["state_name"] == state_sel]

    unique_ym = sorted(current_filtered["year_month"].unique())
    if not unique_ym:
        return {}

    latest_ym = unique_ym[-1]
    all_ym = sorted(ref_df["year_month"].unique())
    idx = all_ym.index(latest_ym) if latest_ym in all_ym else -1
    prev_ym = all_ym[idx - 1] if idx > 0 else None

    cur_df = current_filtered[current_filtered["year_month"] == latest_ym]
    prev_df = ref_df[ref_df["year_month"] == prev_ym] if prev_ym else pd.DataFrame()

    deltas = {}
    for key, (col, agg, kind) in _DELTA_METRICS.items():
        cur_val = getattr(cur_df[col], agg)()
        prev_val = getattr(prev_df[col], agg)() if not prev_df.empty else 0

        if kind == "pct_of_prev":
            deltas[key] = f"{((cur_val - prev_val) / prev_val) * 100:+.1f}% MoM" if prev_val > 0 else None
        else:  # pp_diff
            deltas[key] = f"{(cur_val - prev_val):+.2f}% MoM" if not prev_df.empty else None

    # Positivity Rate is a derived ratio, computed separately
    tot_cur = cur_df["total_tests"].sum()
    tot_prev = prev_df["total_tests"].sum() if not prev_df.empty else 0
    pos_cur = cur_df["positive_tests"].sum()
    pos_prev = prev_df["positive_tests"].sum() if not prev_df.empty else 0
    pr_cur = (pos_cur / tot_cur * 100) if tot_cur > 0 else 0
    pr_prev = (pos_prev / tot_prev * 100) if tot_prev > 0 else 0
    deltas["positivity_rate"] = f"{(pr_cur - pr_prev):+.2f}% MoM" if tot_prev > 0 else None

    return deltas

deltas = calculate_kpi_deltas(filtered_df, df, selected_state)


# 6. KPI DISPLAY

row1 = st.columns(5)

with row1[0]:
    kpi_card_delta("Total Tests", f"{total_tests:,.0f}", delta=deltas.get("total_tests"), bg="#D6EAF8")

with row1[1]:
    kpi_card_delta("Positive Tests", f"{positive_tests:,.0f}", delta=deltas.get("positive_tests"), bg="#EBF5FB")

with row1[2]:
    kpi_card_delta("Positivity Rate", f"{positivity_rate:.2f}%", delta=deltas.get("positivity_rate"),
                    invert=True, bg="#F5B7B1")

with row1[3]:
    kpi_card_delta("Vaccination Coverage", f"{vaccination_coverage:.2f}%", delta=deltas.get("vaccination_coverage"), bg="#D5F5E3")

with row1[4]:
    kpi_card_delta("Booster Coverage", f"{booster_coverage:.2f}%", delta=deltas.get("booster_coverage"), bg="#E8F8F5")


row2 = st.columns(4)

with row2[0]:
    kpi_card_delta("Hospital Beds", f"{hospital_beds:,.0f}", delta=deltas.get("hospital_beds"), bg="#FDEBD0")

with row2[1]:
    kpi_card_delta("Doctors", f"{doctors:,.0f}", delta=deltas.get("doctors"), bg="#EBF5FB")

with row2[2]:
    kpi_card_delta(
        "ICU Utilization", f"{icu_utilization:.2f}%", delta=deltas.get("icu_utilization"),
        invert=True, bg="#FADBD8", badge=icu_status, badge_color=icu_color,
    )

with row2[3]:
    kpi_card_delta("Reporting Compliance", f"{reporting_compliance:.2f}%", delta=deltas.get("reporting_compliance"),
                    bg="linear-gradient(135deg, #D5F5E3 0%, #FCF3CF 50%, #FADBD8 100%)")

# Main-page filters — placed directly under the KPI cards.
with st.container(border=True):
    header_col, reset_col = st.columns([5, 1])
    with header_col:
        filter_bar_header("Selections are applied to every visual on this page")
    with reset_col:
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        st.button("♻️ Reset", on_click=reset_filters, use_container_width=True, key="lab_reset_filters")

    fc1, fc2, fc3 = st.columns(3, gap="medium")
    with fc1:
        st.selectbox("State", state_options, key="state_filter")
    with fc2:
        st.selectbox("Year", year_options, key="year_filter")
    with fc3:
        st.selectbox("Month", month_options, key="month_filter")

    active_lab = []
    for label, value in [("State", selected_state), ("Year", selected_year), ("Month", selected_month)]:
        if value != "All":
            active_lab.append(f'<span class="active-filter-chip">{label}: {value}</span>')
    if not active_lab:
        active_lab.append('<span class="active-filter-chip">All available data</span>')
    st.markdown(
        '<div class="active-filter-summary"><span class="summary-label">Active filters:</span>' + "".join(active_lab) + '</div>',
        unsafe_allow_html=True,
    )

# TESTING TREND + VACCINATION PROGRESS
# SIDE-BY-SIDE LAYOUT

col1, col2 = st.columns(2)

# ==================================================
# VISUAL 1 - TESTING TREND
# ==================================================

with col1:

    section_title("Testing Trend", "Monthly total tests vs positive tests")

    monthly_testing = (
        filtered_df
        .groupby("year_month", as_index=False)
        .agg(
            total_tests=("total_tests", "sum"),
            positive_tests=("positive_tests", "sum")
        )
        .sort_values("year_month")
    )

    fig_testing = go.Figure()

    fig_testing.add_trace(
        go.Scatter(
            x=monthly_testing["year_month"],
            y=monthly_testing["total_tests"],
            mode="lines+markers",
            name="Total Tests",
            line=dict(
                color=PRIMARY_NAVY,
                width=3,
                shape="spline"
            ),
            marker=dict(size=5, color=PRIMARY_NAVY, line=dict(width=1, color="white")),
            fill="tozeroy",
            fillcolor="rgba(23, 50, 77, 0.10)",
            hovertemplate=(
                "Month: %{x}<br>"
                "Total Tests: %{y:,.0f}"
                "<extra></extra>"
            )
        )
    )

    fig_testing.add_trace(
        go.Scatter(
            x=monthly_testing["year_month"],
            y=monthly_testing["positive_tests"],
            mode="lines+markers",
            name="Positive Tests",
            line=dict(
                color=TEAL,
                width=3,
                shape="spline"
            ),
            marker=dict(size=5, color=TEAL, line=dict(width=1, color="white")),
            fill="tozeroy",
            fillcolor="rgba(15, 107, 120, 0.16)",
            hovertemplate=(
                "Month: %{x}<br>"
                "Positive Tests: %{y:,.0f}"
                "<extra></extra>"
            )
        )
    )

    fig_testing.update_layout(
        xaxis_title=dict(
            text="Month",
            font=dict(
                color=TEXT,
                size=13
            )
        ),
        yaxis_title=dict(
            text="Tests",
            font=dict(
                color=TEXT,
                size=13
            )
        ),
        hovermode="x unified",
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(
            family="Arial",
            color=TEXT,
            size=11
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(
                color=TEXT,
                size=10
            )
        ),
        margin=dict(
            l=50,
            r=20,
            t=55,
            b=60
        )
    )

    fig_testing.update_xaxes(
        showgrid=False,
        linecolor=BORDER,
        tickangle=-40,
        nticks=12,
        tickfont=dict(
            color=TEXT,
            size=9
        )
    )

    fig_testing.update_yaxes(
        showgrid=True,
        gridcolor="rgba(100, 116, 139, 0.12)",
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(
            color=TEXT,
            size=9
        )
    )

    st.plotly_chart(
        fig_testing,
        use_container_width=True
    )


# ==================================================
# VISUAL 2 - VACCINATION PROGRESS
# ==================================================

with col2:

    section_title("Vaccination Progress", "Monthly vaccination and booster coverage")

    monthly_vaccination = (
        filtered_df
        .groupby("year_month", as_index=False)
        .agg(
            vaccination_coverage=(
                "vaccination_coverage_pct",
                "mean"
            ),
            booster_coverage=(
                "booster_coverage_pct",
                "mean"
            )
        )
        .sort_values("year_month")
    )

    fig_vaccination = go.Figure()

    fig_vaccination.add_trace(
        go.Scatter(
            x=monthly_vaccination["year_month"],
            y=monthly_vaccination["vaccination_coverage"],
            mode="lines",
            name="Vaccination Coverage",
            line=dict(
                color=TEAL,
                width=3
            ),
            fill="tozeroy",
            fillcolor="rgba(15, 107, 120, 0.18)",
            hovertemplate=(
                "Month: %{x}<br>"
                "Vaccination Coverage: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )

    fig_vaccination.add_trace(
        go.Scatter(
            x=monthly_vaccination["year_month"],
            y=monthly_vaccination["booster_coverage"],
            mode="lines",
            name="Booster Coverage",
            line=dict(
                color=GREEN,
                width=3
            ),
            fill="tozeroy",
            fillcolor="rgba(22, 133, 91, 0.18)",
            hovertemplate=(
                "Month: %{x}<br>"
                "Booster Coverage: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )

    fig_vaccination.update_layout(
        xaxis_title=dict(
            text="Month",
            font=dict(
                color=TEXT,
                size=13
            )
        ),
        yaxis_title=dict(
            text="Coverage (%)",
            font=dict(
                color=TEXT,
                size=13
            )
        ),
        hovermode="x unified",
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(
            family="Arial",
            color=TEXT,
            size=11
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(
                color=TEXT,
                size=10
            )
        ),
        margin=dict(
            l=50,
            r=20,
            t=55,
            b=45
        )
    )

    fig_vaccination.update_xaxes(
        showgrid=False,
        linecolor=BORDER,
        tickfont=dict(
            color=TEXT,
            size=9
        )
    )

    fig_vaccination.update_yaxes(
        showgrid=True,
        gridcolor="rgba(100, 116, 139, 0.12)",
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(
            color=TEXT,
            size=9
        ),
        range=[0, 100]
    )

    st.plotly_chart(
        fig_vaccination,
        use_container_width=True
    )

# HOSPITAL CAPACITY + ICU UTILIZATION +
# LABORATORY PERFORMANCE
# SIDE-BY-SIDE LAYOUT

col3, col4, col5 = st.columns(3)


# ==================================================
# VISUAL 3 - HOSPITAL CAPACITY
# ==================================================

with col3:

    section_title("Hospital Capacity", "Beds, doctors, PHCs, and CHCs")

    hospital_capacity = pd.DataFrame({
        "Category": [
            "Hospital Beds",
            "Doctors",
            "PHCs",
            "CHCs"
        ],
        "Value": [
            filtered_df["hospital_beds"].sum(),
            filtered_df["doctors"].sum(),
            filtered_df["phc_count"].sum(),
            filtered_df["chc_count"].sum()
        ]
    })

    fig_capacity = go.Figure()

    fig_capacity.add_trace(
        go.Bar(
            x=hospital_capacity["Category"],
            y=hospital_capacity["Value"],
            text=hospital_capacity["Value"],
            texttemplate="%{text:,.0f}",
            textposition="outside",
            name="Capacity",
            marker_color=SLATE_BLUE,
            hovertemplate=(
                "%{x}<br>"
                "Value: %{y:,.0f}"
                "<extra></extra>"
            )
        )
    )

    fig_capacity.update_layout(
        height=390,
        xaxis_title="",
        yaxis_title="Count",
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(
            family="Arial",
            color=TEXT,
            size=10
        ),
        showlegend=False,
        margin=dict(
            l=45,
            r=15,
            t=30,
            b=70
        )
    )

    fig_capacity.update_xaxes(
        tickfont=dict(
            color=TEXT,
            size=9
        ),
        showgrid=False
    )

    fig_capacity.update_yaxes(
        tickfont=dict(
            color=TEXT,
            size=9
        ),
        showgrid=True,
        gridcolor="rgba(100, 116, 139, 0.12)"
    )

    st.plotly_chart(
        fig_capacity,
        use_container_width=True
    )


# ==================================================
# VISUAL 4 - ICU UTILIZATION
# ==================================================

with col4:

    section_title("ICU Utilization", "Average ICU utilization rate")

    icu_value = filtered_df[
        "icu_utilization_pct"
    ].mean()

    gauge_status, gauge_color, _ = icu_status_color(icu_value)

    fig_icu = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=icu_value,

            number=dict(
                suffix="%",
                valueformat=".2f",
                font=dict(
                    color=PRIMARY_NAVY,
                    size=36,
                    family="Arial"
                )
            ),

            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickwidth=1,
                    tickcolor="rgba(100, 116, 139, 0.3)",
                    tickfont=dict(
                        color=TEXT,
                        size=10
                    )
                ),

                bar=dict(
                    color=gauge_color,
                    thickness=0.28
                ),

                bgcolor=WHITE,

                bordercolor=BORDER,
                borderwidth=1,

                steps=[
                    {'range': [0, 50], 'color': "rgba(22, 133, 91, 0.10)"},
                    {'range': [50, 85], 'color': "rgba(201, 138, 0, 0.10)"},
                    {'range': [85, 100], 'color': "rgba(217, 56, 30, 0.14)"}
                ],

                threshold=dict(
                    line=dict(color="#D9381E", width=3),
                    thickness=0.75,
                    value=85
                )
            ),

            title=dict(
                text=f"<b style='color:{gauge_color}; font-size:22px;'>{gauge_status}</b><br><span style='color:#000000; font-size:13px;'>ICU Utilization Status</span>",
                font=dict(
                    family="Arial"
                )
            )
        )
    )

    fig_icu.update_layout(
        height=390,
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(
            family="Arial",
            color=TEXT
        ),
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    st.plotly_chart(
        fig_icu,
        use_container_width=True
    )


# ==================================================
# VISUAL 5 - LABORATORY PERFORMANCE
# ==================================================

with col5:

    section_title("Laboratory Performance", "Reporting rate and turnaround time")

    reporting_rate = filtered_df[
        "reporting_rate_pct"
    ].mean()

    turnaround_time = filtered_df[
        "turnaround_time_days"
    ].mean()

    laboratory_performance = pd.DataFrame({
        "Metric": [
            "Reporting Rate",
            "Turnaround Time"
        ],
        "Value": [
            reporting_rate,
            turnaround_time
        ]
    })

    fig_lab = go.Figure()

    fig_lab.add_trace(
        go.Bar(
            x=laboratory_performance["Metric"],
            y=laboratory_performance["Value"],
            text=laboratory_performance["Value"],
            texttemplate="%{text:.2f}",
            textposition="outside",
            name="Laboratory Performance",
            marker_color=BAR_SEQUENCE[3],
            hovertemplate=(
                "%{x}<br>"
                "Value: %{y:.2f}"
                "<extra></extra>"
            )
        )
    )

    fig_lab.update_layout(
        height=390,
        xaxis_title="",
        yaxis_title="Value",
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(
            family="Arial",
            color=TEXT,
            size=10
        ),
        showlegend=False,
        margin=dict(
            l=45,
            r=15,
            t=30,
            b=70
        )
    )

    fig_lab.update_xaxes(
        tickfont=dict(
            color=TEXT,
            size=9
        ),
        showgrid=False
    )

    fig_lab.update_yaxes(
        tickfont=dict(
            color=TEXT,
            size=9
        ),
        showgrid=True,
        gridcolor="rgba(100, 116, 139, 0.12)"
    )

    st.plotly_chart(
        fig_lab,
        use_container_width=True
    )
# VISUAL 6 - BED OCCUPANCY HEAT MAP

section_title("Bed Occupancy by State", "Average bed occupancy by state and month — green = safe, yellow = warning, red = critical")

# Calculate average bed occupancy for each State and Month
bed_occupancy = (
    filtered_df
    .groupby(
        ["state_name", "year_month"],
        as_index=False
    )
    .agg(
        bed_occupancy=("bed_occupancy_pct", "mean")
    )
)

# Create State × Month matrix
heatmap_data = bed_occupancy.pivot(
    index="state_name",
    columns="year_month",
    values="bed_occupancy"
)

# Sort states alphabetically
heatmap_data = heatmap_data.sort_index()

# Sort months chronologically
heatmap_data = heatmap_data.reindex(
    sorted(heatmap_data.columns),
    axis=1
)

fig_bed = go.Figure(
    data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,

    # Shared Green -> Yellow -> Red risk heatmap scale (src/chart_colors.py)
    # so this reads exactly the same as every other risk heatmap in the suite.
    colorscale=RISK_HEATMAP_SCALE,

        colorbar=dict(
            title=dict(text="Occupancy (%)", font=dict(color=TEXT, size=12)),
            tickfont=dict(color=TEXT, size=11),
        ),

        hovertemplate=(
            "State: %{y}<br>"
            "Month: %{x}<br>"
            "Bed Occupancy: %{z:.2f}%"
            "<extra></extra>"
        ),

        xgap=1,
        ygap=1
    )
)

fig_bed.update_layout(
    height=650,

    plot_bgcolor=WHITE,
    paper_bgcolor=WHITE,

    font=dict(
        family="Arial",
        color=TEXT,
        size=11
    ),

    xaxis=dict(
        title="Month",
        title_font=dict(
            color=TEXT,
            size=14
        ),
        tickfont=dict(
            color=TEXT,
            size=10
        ),
        showgrid=False,
        side="bottom"
    ),

    yaxis=dict(
        title="State",
        title_font=dict(
            color=TEXT,
            size=14
        ),
        tickfont=dict(
            color=TEXT,
            size=10
        ),
        showgrid=False,
        autorange="reversed"
    ),

    margin=dict(
        l=120,
        r=80,
        t=40,
        b=70
    )
)

st.plotly_chart(
    fig_bed,
    use_container_width=True
)


# 7. FILTERED DATA DETAILS TABLE

with st.expander(" View & Explore Filtered Data Table", expanded=False):
    st.markdown("<h4 style='color:#17324D; margin-top:5px; margin-bottom:10px;'>Detailed Operations Breakdown</h4>", unsafe_allow_html=True)
    display_cols = [
        "state_name", "year_month", "total_tests", "positive_tests",
        "vaccination_coverage_pct", "booster_coverage_pct",
        "hospital_beds", "doctors", "icu_utilization_pct", "reporting_compliance_pct"
    ]
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    rename_map = {
        "state_name": "State",
        "year_month": "Month",
        "total_tests": "Total Tests",
        "positive_tests": "Positive Tests",
        "vaccination_coverage_pct": "Vaccination (%)",
        "booster_coverage_pct": "Booster (%)",
        "hospital_beds": "Hospital Beds",
        "doctors": "Doctors",
        "icu_utilization_pct": "ICU Util (%)",
        "reporting_compliance_pct": "Compliance (%)"
    }
    st.dataframe(
        filtered_df[available_cols].rename(columns=rename_map),
        use_container_width=True,
        hide_index=True
    )


# 8. FOOTER

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
_by_state = filtered_df.groupby("state_name", as_index=False).agg(
    positivity_rate=("positivity_rate", "mean"),
    vaccination_coverage_pct=("vaccination_coverage_pct", "mean"),
    icu_utilization_pct=("icu_utilization_pct", "mean"),
)
_top_positivity = _by_state.sort_values("positivity_rate", ascending=False).iloc[0]
_low_vax = _by_state.sort_values("vaccination_coverage_pct", ascending=True).iloc[0]
_top_icu = _by_state.sort_values("icu_utilization_pct", ascending=False).iloc[0]
snapshot_row([
    ("Highest positivity rate", f'{_top_positivity["state_name"]} • {_top_positivity["positivity_rate"]:.2f}%'),
    ("Lowest vaccination coverage", f'{_low_vax["state_name"]} • {_low_vax["vaccination_coverage_pct"]:.2f}%'),
    ("Highest ICU utilization", f'{_top_icu["state_name"]} • {_top_icu["icu_utilization_pct"]:.2f}%'),
])

st.markdown(
    f"""
    <div style="
        margin-top: 30px;
        padding-top: 15px;
        border-top: 1px solid {BORDER};
        text-align: center;
        font-size: 12px;
        color: {SECONDARY_TEXT};
    ">
        Source: Laboratory & Healthcare Capacity dataset · Data range: {date_min} to {date_max} ·
    </div>
    """,
    unsafe_allow_html=True
)