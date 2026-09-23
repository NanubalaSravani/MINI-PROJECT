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
from src.data_loader import get_outbreak_master

# Note: st.set_page_config() is intentionally NOT called here — app.py
# already calls it once, centrally, before st.navigation(). See the
# comment in dashboards/2_Laboratory_Healthcare_Capacity.py for why.

inject_css()

# ==========================================
# OPERATIONAL EMERGENCY ALERT STATUS TICKER
# ==========================================
try:
    _outb_df = get_outbreak_master()
    if not _outb_df.empty and "alert_level" in _outb_df.columns:
        _high_alerts = _outb_df[_outb_df["alert_level"] == "High"]
        _high_count = len(_high_alerts)
        _top_states = _high_alerts["state_name"].value_counts().head(3).index.tolist() if "state_name" in _high_alerts.columns else []
        _states_str = ", ".join(_top_states) if _top_states else "Multiple States"

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, rgba(196,61,61,0.08) 0%, rgba(201,138,0,0.06) 100%);
                border: 1px solid rgba(196,61,61,0.3);
                border-left: 5px solid #C43D3D;
                border-radius: 8px;
                padding: 12px 18px;
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            ">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 1.4rem;">🚨</span>
                    <div>
                        <div style="font-size: 0.92rem; font-weight: 700; color: #9E1F1F; text-transform: uppercase; letter-spacing: 0.5px;">
                            Active Operational Surveillance Advisory
                        </div>
                        <div style="font-size: 0.85rem; color: #17324D; margin-top: 2px;">
                            <strong>{_high_count:,} High-Alert Outbreak Incidents</strong> active nationwide · Priority Containment Focus: <strong>{_states_str}</strong>
                        </div>
                    </div>
                </div>
                <div style="font-size: 0.8rem; background: #FFFFFF; border: 1px solid rgba(196,61,61,0.25); border-radius: 20px; padding: 4px 12px; font-weight: 600; color: #C43D3D;">
                    ● Threat Level: Elevated
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
except Exception:
    pass

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
    st.image(str(_hero_path), use_container_width=True)
except TypeError:
    st.image(str(_hero_path), use_container_width=True)

# ==========================================
# Explore the Dashboards — complete directory
# ==========================================
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Explore the Dashboards</div>'
    '<div class="section-caption">What you\'ll find behind each tab in the navigation menu</div>',
    unsafe_allow_html=True,
)

DASHBOARDS = [
    {
        "icon": "📊",
        "title": "Executive Public Health Overview",
        "desc": "National disease burden, outcomes, and state performance summary — the top-level view for leadership.",
        "color": "#0F6B78",
    },
    {
        "icon": "🌍",
        "title": "Geographic & Environmental Intelligence",
        "desc": "Connects spatial risks and environmental stressors (AQI, sanitation, rainfall) against epidemiological trends.",
        "color": "#16855B",
    },
    {
        "icon": "🧪",
        "title": "Laboratory & Healthcare Capacity",
        "desc": "Testing volumes, positivity rates, vaccination coverage, and critical care / ICU bed occupancy across states.",
        "color": "#17324D",
    },
    {
        "icon": "🚨",
        "title": "Outbreak Monitoring & Forecasting",
        "desc": "Live alert levels, ARIMA/Holt-Winters multi-model case forecasting, rolling Z-score surge detection, and ML risk drivers.",
        "color": "#C43D3D",
    },
    {
        "icon": "🤝",
        "title": "Health Programs & Population Vulnerability",
        "desc": "Tracks public health scheme reach, maternal-child immunization, and demographic vulnerability indices.",
        "color": "#C98A00",
    },
    {
        "icon": "📁",
        "title": "Upload & Custom Analysis",
        "desc": "Authenticated ingestion portal with strict schema validation, SQLite review audit log, and automated PDF report export.",
        "color": "#5A6A7A",
    },
    {
        "icon": "🤖",
        "title": "Ask Sentinel (AI Copilot)",
        "desc": "Grounded natural language epidemiological Q&A, optional Gemini LLM reasoning, and one-click Executive SITREP briefing.",
        "color": "#0F6B78",
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
