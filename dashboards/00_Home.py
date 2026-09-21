"""
00_Home.py
------------
Landing page — shown before Executive Public Health Overview in the nav.
Just the branded hero graphic, sized to the page width; no additional
header banner on top of it since the image already carries its own
branding/hero content.
"""

from pathlib import Path

import streamlit as st

from src.styling import inject_css

# Note: st.set_page_config() is intentionally NOT called here — app.py
# already calls it once, centrally, before st.navigation(). See the
# comment in dashboards/2_Laboratory_Healthcare_Capacity.py for why.

inject_css()

with st.sidebar:
    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.caption(
        "HealthSentinel is a public-health analytics suite for India, "
        "bringing together disease surveillance, environmental risk, lab & "
        "hospital capacity, outbreak monitoring, and health-program "
        "performance in one place."
    )
    st.markdown(
        "**Coverage:** 32 states · Jan 2022 – Dec 2024  \n"
        "**Use case:** Public-health decision support"
    )
    st.markdown("---")
    st.caption("Infosys Public Health Analytics · Internal Use")

_hero_path = Path(__file__).resolve().parent.parent / "assets" / "home_hero.jpg"
try:
    # Newer Streamlit versions (renamed the parameter; use_column_width
    # is deprecated there but still works, just with a warning banner).
    st.image(str(_hero_path), use_container_width=True)
except TypeError:
    # Older Streamlit versions (this project's pinned requirements.txt,
    # 1.38.0) don't have use_container_width on st.image() at all.
    st.image(str(_hero_path), use_column_width=True)

# ==========================================
# Explore the Dashboards — one card per page in the sidebar nav, so the
# landing page tells people what's behind each tab before they click it.
# ==========================================
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Explore the Dashboards</div>'
    '<div class="section-caption">What you\'ll find behind each tab in the sidebar</div>',
    unsafe_allow_html=True,
)

DASHBOARDS = [
    {
        "icon": "📊",
        "title": "Executive Public Health Overview",
        "desc": "National disease burden, outcomes, and state performance summary — the "
                "top-level view for a quick read on how the country is doing.",
        "color": "#0F6B78",
    },
    {
        "icon": "🌍",
        "title": "Geographic & Environmental Intelligence",
        "desc": "Connects geographic risk, environmental stressors (air quality, water "
                "quality, climate) and disease burden to show where attention is most needed.",
        "color": "#16855B",
    },
    {
        "icon": "🧪",
        "title": "Laboratory & Healthcare Capacity",
        "desc": "Testing volumes, positivity rates, vaccination coverage, and hospital / "
                "ICU capacity across states, month by month.",
        "color": "#17324D",
    },
    {
        "icon": "🚨",
        "title": "Outbreak Monitoring & Forecasting",
        "desc": "Live alert levels and containment performance, plus ARIMA-based case "
                "forecasting and a priority containment matrix for active outbreaks.",
        "color": "#C43D3D",
    },
    {
        "icon": "🤝",
        "title": "Health Programs & Population Vulnerability",
        "desc": "Tracks public health program coverage and performance, and flags "
                "vulnerable populations that need attention.",
        "color": "#C98A00",
    },
]

cols = st.columns(3, gap="medium")
for i, dash in enumerate(DASHBOARDS):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color:{dash['color']}; margin-bottom:20px;">
                <div style="font-size:1.5rem; margin-bottom:6px;">{dash['icon']}</div>
                <div style="font-size:0.95rem; font-weight:700; color:#000000; margin-bottom:6px;">
                    {dash['title']}
                </div>
                <div style="font-size:0.8rem; color:#000000; line-height:1.45;">
                    {dash['desc']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
