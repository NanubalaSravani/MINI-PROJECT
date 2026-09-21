"""
1_Geographic_Environmental_Intelligence.py
-------------------------------------------
Geographic & Environmental Intelligence page.

The page is intentionally kept as Streamlit orchestration. Reusable calculations,
risk logic, and Plotly builders live in src/geographic.py, while the visual system
lives in src/styling.py.
"""

import streamlit as st

from src.data_loader import get_environmental_master, get_surveillance_master
from src.geographic import (
    PLOTLY_CONFIG,
    build_environmental_trend,
    build_gauge,
    build_hotspot_chart,
    build_radar,
    build_state_map,
    build_urban_rural,
    build_water_scatter,
    calculate_kpis,
    calculate_pressure,
    common_layout,
    decision_snapshot,
    filter_data,
    get_date_bounds,
    get_date_window,
    risk_band,
    risk_color,
)
from src.styling import inject_geographic_css, page_header, insight_banner, filter_bar_header
from src.report_generator import add_sidebar_report_button

inject_geographic_css()

# Data

env = get_environmental_master()
ds = get_surveillance_master()
MIN_DATE, MAX_DATE = get_date_bounds(env)


def reset_filters():
    """Reset all dashboard filters to the complete available dataset."""
    st.session_state["geo_period"] = "All available data"
    st.session_state["phs_custom_date_range"] = (MIN_DATE, MAX_DATE)
    st.session_state["geo_region"] = "All Regions"
    st.session_state["geo_state_focus"] = "All States"
    st.session_state["geo_disease_focus"] = "All Diseases"


# Initialize filter state before KPI calculations. The widgets themselves are
# rendered below the KPI cards, but session_state keeps the selected context
# available to the calculations before the widgets are drawn.
st.session_state.setdefault("geo_period", "All available data")
st.session_state.setdefault("phs_custom_date_range", (MIN_DATE, MAX_DATE))
st.session_state.setdefault("geo_region", "All Regions")
st.session_state.setdefault("geo_state_focus", "All States")
st.session_state.setdefault("geo_disease_focus", "All Diseases")

period = st.session_state["geo_period"]
region = st.session_state["geo_region"]
state_focus = st.session_state["geo_state_focus"]
disease_focus = st.session_state["geo_disease_focus"]

region_options = ["All Regions"] + sorted(env["region"].dropna().unique().tolist())
if region not in region_options:
    region = "All Regions"
    st.session_state["geo_region"] = region
state_pool = env if region == "All Regions" else env[env["region"] == region]
state_options = ["All States"] + sorted(state_pool["state_name"].dropna().unique().tolist())
if state_focus not in state_options:
    state_focus = "All States"
    st.session_state["geo_state_focus"] = state_focus
disease_options = ["All Diseases"] + sorted(ds["disease_name"].dropna().unique().tolist())
if disease_focus not in disease_options:
    disease_focus = "All Diseases"
    st.session_state["geo_disease_focus"] = disease_focus

if period == "Custom range":
    custom = st.session_state.get("phs_custom_date_range", (MIN_DATE, MAX_DATE))
    if isinstance(custom, (tuple, list)) and len(custom) == 2:
        start_date, end_date = custom
    else:
        start_date = end_date = custom
else:
    start_date, end_date = get_date_window(period, MIN_DATE, MAX_DATE)


# Header

page_header(
    "Geographic & Environmental Intelligence",
    "A decision-support view that connects geographic risk, environmental stressors "
    "and disease burden to identify where public-health attention is most needed.",
)

# Filtered data

env_f, ds_f = filter_data(
    env, ds, start_date, end_date, region, state_focus, disease_focus
)

if env_f.empty:
    st.error(
        "No environmental records match the selected filters. "
        "Broaden the analysis period or geography."
    )
    st.stop()

# The report generator's chart builders expect the shorter column names
# used elsewhere in the app (e.g. "geographic_risk" rather than
# "geographic_risk_score"), so pass a renamed copy for report purposes only
# — the page's own charts keep using env_f/ds_f untouched.
_report_df = env_f.rename(columns={
    "geographic_risk_score": "geographic_risk",
    "environmental_risk_score": "environmental_risk",
    "water_quality_index": "water_quality",
    "sanitation_coverage_pct": "sanitation",
    "healthcare_accessibility_score": "healthcare_access",
})
add_sidebar_report_button(
    "Geographic & Environmental Intelligence",
    _report_df,
    {
        "Analysis Period": period if period != "Custom range"
            else f"{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}",
        "Region": region,
        "State": state_focus,
        "Disease Focus": disease_focus,
    },
)


# KPI row

kpis = calculate_kpis(env_f)

insight_banner(
    f"Environmental risk is averaging **{kpis['environmental_risk']:.1f}** ({risk_band(kpis['environmental_risk'])}); "
    f"AQI is standing at **{kpis['aqi']:.0f}**; water quality index is **{kpis['water_quality']:.1f}/100** "
    f"({region if region != 'All Regions' else 'across all regions'})."
)

kpi_items = [
    ("Geographic Risk", f'{kpis["geographic_risk"]:.1f}', "risk score",
     "linear-gradient(135deg, #D5F5E3 0%, #FCF3CF 50%, #FADBD8 100%)"),
    ("Environmental Risk", f'{kpis["environmental_risk"]:.1f}', risk_band(kpis["environmental_risk"]), "#F5B7B1"),
    ("AQI", f'{kpis["aqi"]:.0f}', "average index", "#FDEBD0"),
    ("Rainfall", f'{kpis["rainfall"]:.1f} mm', "average", "#D6EAF8"),
    ("Temperature", f'{kpis["temperature"]:.1f} °C', "average", "#FADBD8"),
    ("Water Quality", f'{kpis["water_quality"]:.1f}', "index / 100", "#EBF5FB"),
    ("Sanitation", f'{kpis["sanitation"]:.1f}%', "coverage", "#D5F5E3"),
    ("Healthcare Access", f'{kpis["healthcare_access"]:.1f}', "score / 100", "#E8F8F5"),
]
cards = ['<div class="kpi-grid">']
for label, value, meta, bg in kpi_items:
    cards.append(
        f'<div class="kpi" style="background:{bg};"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-meta">● {meta}</div></div>'
    )
cards.append("</div>")
st.markdown("".join(cards), unsafe_allow_html=True)

# Main-page filter bar — intentionally placed directly under the KPI cards.
with st.container(border=True):
    header_col, reset_col = st.columns([5, 1])
    with header_col:
        filter_bar_header("Selections are applied to every visual on this page")
    with reset_col:
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        st.button("♻️ Reset", use_container_width=True, key="geo_reset_filters", on_click=reset_filters)

    row1 = st.columns([1.3, 1, 1, 1], gap="medium")
    with row1[0]:
        st.selectbox(
            "Analysis period",
            ["All available data", "Latest month", "Past 30 days", "Past 90 days", "Past 12 months", "Custom range"],
            key="geo_period",
            help=(f"Your dataset ends on {MAX_DATE.strftime('%d %b %Y')}. Relative periods are calculated from the latest available data."),
        )
    with row1[1]:
        st.selectbox("Region", region_options, key="geo_region")
    with row1[2]:
        st.selectbox("State focus", state_options, key="geo_state_focus")
    with row1[3]:
        st.selectbox("Disease for case analysis", disease_options, key="geo_disease_focus")

    if period == "Custom range":
        st.date_input(
            "Custom date range",
            value=(MIN_DATE, MAX_DATE),
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            key="phs_custom_date_range",
        )

    active_geo = []
    for label, value, default in [
        ("Period", period, "All available data"),
        ("Region", region, "All Regions"),
        ("State", state_focus, "All States"),
        ("Disease", disease_focus, "All Diseases"),
    ]:
        if value != default:
            active_geo.append(f'<span class="active-filter-chip">{label}: {value}</span>')
    if period == "Custom range":
        active_geo.append('<span class="active-filter-chip">Custom date range selected</span>')
    if not active_geo:
        active_geo.append('<span class="active-filter-chip">All available data</span>')
    st.markdown(
        '<div class="active-filter-summary"><span class="summary-label">Active filters:</span>' + "".join(active_geo) + '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f"<div class='filter-bar-caption' style='margin:10px 4px 0;'>Data window: <b>{start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}</b> · Environmental risk = environmental fact table · Disease burden = disease surveillance fact table.</div>",
    unsafe_allow_html=True,
)


# Geographic risk landscape

state_map, fig_geo = build_state_map(env_f, ds_f)

st.markdown(
    '<div class="section-title">Geographic Risk Landscape</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">Interactive state-risk pulse map • bubble size = geographic risk • '
    'color = risk intensity • hover for full state intelligence</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(fig_geo, use_container_width=True, config=PLOTLY_CONFIG)

# Hotspots + environmental fingerprint + gauge

c1, c2, c3 = st.columns([1.35, 1.05, .85])

with c1:
    st.markdown('<div class="section-title">Geographic Hotspots</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Top states ranked by geographic risk</div>', unsafe_allow_html=True)
    st.plotly_chart(build_hotspot_chart(state_map), use_container_width=True, config=PLOTLY_CONFIG)

with c2:
    st.markdown('<div class="section-title">Environmental Risk Fingerprint</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Which environmental pressures are driving attention?</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_radar(calculate_pressure(env_f)),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

with c3:
    st.markdown('<div class="section-title">Risk Gauge</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Composite environmental risk</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_gauge(kpis["environmental_risk"]),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

# Urban/rural disease burden + environmental trend

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="section-title">Urban vs Rural Disease Burden</div>', unsafe_allow_html=True)
    caption = (
        "Reported cases by settlement type"
        if disease_focus == "All Diseases"
        else f"Reported cases by settlement type • {disease_focus}"
    )
    st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)
    if ds_f.empty:
        st.info("No disease surveillance records match the selected filters.")
    else:
        st.plotly_chart(build_urban_rural(ds_f), use_container_width=True, config=PLOTLY_CONFIG)

with c2:
    st.markdown('<div class="section-title">Environmental Indicators Trend</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Use the selector to switch the environmental signal</div>', unsafe_allow_html=True)
    trend_metric = st.selectbox(
        "Indicator",
        ["AQI", "Rainfall", "Temperature", "Water Quality", "Environmental Risk"],
        key="trend_indicator",
        label_visibility="collapsed",
    )
    st.plotly_chart(
        build_environmental_trend(env_f, trend_metric),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


# Water quality vs zoonotic incidence

st.markdown('<div class="section-title">Water Quality vs Zoonotic Incidence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Bubble size = case rate • color = environmental risk • each bubble is a state</div>',
    unsafe_allow_html=True,
)

fig_sc, scatter, corr = build_water_scatter(env_f)
if fig_sc is not None:
    st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_CONFIG)

    strength = "weak" if abs(corr) < .3 else "moderate" if abs(corr) < .6 else "strong"
    direction = "positive" if corr >= 0 else "negative"
    st.markdown(
        f'<div class="insight"><b>Analyst signal:</b> water quality and zoonotic incidence show a '
        f'<b>{strength} {direction}</b> association in the selected data (r = {corr:.2f}). '
        f'This is an analytical association, not proof of causation.</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Not enough state-level records to calculate the water-quality association.")

# Decision snapshot

st.markdown('<div class="section-title">Decision Snapshot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Three signals a public-health decision maker should notice first</div>',
    unsafe_allow_html=True,
)

top_state, aqi_state, water_state = decision_snapshot(state_map)
d1, d2, d3 = st.columns(3)

with d1:
    st.markdown(
        f'<div class="small-card"><div class="small-card-title">Highest geographic risk</div>'
        f'<div class="small-card-value">{top_state["state_name"]} • {top_state["geographic_risk"]:.1f}</div></div>',
        unsafe_allow_html=True,
    )

with d2:
    st.markdown(
        f'<div class="small-card"><div class="small-card-title">Highest AQI</div>'
        f'<div class="small-card-value">{aqi_state["state_name"]} • {aqi_state["aqi"]:.0f}</div></div>',
        unsafe_allow_html=True,
    )

with d3:
    st.markdown(
        f'<div class="small-card"><div class="small-card-title">Lowest water quality</div>'
        f'<div class="small-card-value">{water_state["state_name"]} • {water_state["water_quality"]:.1f}</div></div>',
        unsafe_allow_html=True,
    )

