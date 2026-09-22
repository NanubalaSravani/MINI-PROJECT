"""
app.py
------
Single entry point for the Public Health Analytics multipage app.

Wires the Home page + 5-dashboard suite into one left-hand navigation menu
(via st.navigation) so the whole suite can be presented from one running
app. Order in the sidebar = order below. Home loads by default (just the
hero graphic); Executive Public Health Overview, Geographic & Environmental
Intelligence, Laboratory & Healthcare Capacity, and Health Programs &
Population Vulnerability are fully built out. Outbreak Monitoring &
Forecasting has a real Decision Snapshot but is otherwise still in
progress.

Run with:  streamlit run app.py
"""

import streamlit as st
import plotly.io as pio
from pathlib import Path

# Force every Plotly chart in the app (axis titles, tick labels, legends,
# colorbars, hover text) to render in solid black by default, rather than
# Plotly's default light-grey font, so chart text stays clearly readable
# against the light card backgrounds used throughout the dashboard.
# NOTE: layout.font.color alone does NOT cascade down to axis tick labels
# or colorbar tick labels in Plotly — those need their own explicit
# color, otherwise they fall back to Plotly's built-in grey regardless of
# the top-level font color. Every text surface is set explicitly below so
# no chart is left with grey text, whether or not it also sets its own
# per-chart overrides.
pio.templates["healthsentinel_dark_text"] = pio.templates["plotly_white"]
pio.templates["healthsentinel_dark_text"].layout.font.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.title.font.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.xaxis.title.font.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.xaxis.tickfont.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.yaxis.title.font.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.yaxis.tickfont.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.legend.font.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.legend.title.font.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.coloraxis.colorbar.tickfont.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.coloraxis.colorbar.title.font.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.hoverlabel.font.color = "#000000"
pio.templates["healthsentinel_dark_text"].layout.annotationdefaults.font.color = "#000000"
pio.templates.default = "healthsentinel_dark_text"

# Belt-and-suspenders: some individual charts set their own tickfont/title
# font dicts (color + size only), which take precedence over the shared
# template above for that one chart. Rather than hunt down and edit every
# such override across 7 dashboard files, wrap st.plotly_chart itself so
# every chart's x/y axis titles and tick labels are forced to bold, solid
# black text right before rendering — this runs after each page builds its
# figure, so it always wins regardless of what that page already set.
# (Uses a heavier font family rather than Plotly's newer font.weight
# property, since family strings are always safe to set and don't risk
# an "invalid property" error on older Plotly versions.)
_original_plotly_chart = st.plotly_chart


def _bold_black_axes_plotly_chart(figure_or_data=None, *args, **kwargs):
    try:
        # Pop figure_or_data from kwargs if it was passed as keyword to avoid
        # passing it twice (positionally + as keyword) when forwarding below.
        if figure_or_data is None:
            figure_or_data = kwargs.pop("figure_or_data", None)
        fig = figure_or_data
        if fig is not None and hasattr(fig, "update_xaxes"):
            _bold_axis_font = dict(color="#000000", family="Arial Black, Arial, sans-serif")
            fig.update_xaxes(tickfont={**_bold_axis_font, "size": 12}, title_font={**_bold_axis_font, "size": 13})
            fig.update_yaxes(tickfont={**_bold_axis_font, "size": 12}, title_font={**_bold_axis_font, "size": 13})
    except Exception:
        pass  # never let axis styling break a page render
    if figure_or_data is not None:
        return _original_plotly_chart(figure_or_data, *args, **kwargs)
    return _original_plotly_chart(*args, **kwargs)


st.plotly_chart = _bold_black_axes_plotly_chart

st.set_page_config(
    page_title="HealthSentinel",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

_assets_dir = Path(__file__).resolve().parent / "assets"
st.logo(
    str(_assets_dir / "logo_full.png"),
    icon_image=str(_assets_dir / "logo_icon.png"),
    size='large',
    
)

home = st.Page(
    "dashboards/00_Home.py",
    title="Home",
    default=True,
)
executive_overview = st.Page(
    "dashboards/0_Executive_Public_Health_Overview.py",
    title="Executive Public Health Overview",
)
geographic_environmental = st.Page(
    "dashboards/1_Geographic_Environmental_Intelligence.py",
    title="Geographic & Environmental Intelligence",
)
laboratory_healthcare = st.Page(
    "dashboards/2_Laboratory_Healthcare_Capacity.py",
    title="Laboratory & Healthcare Capacity",
)
outbreak_monitoring = st.Page(
    "dashboards/3_Outbreak_Monitoring_Forecasting.py",
    title="Outbreak Monitoring & Forecasting",
)
health_programs_vulnerability = st.Page(
    "dashboards/4_Health_Programs_Population_Vulnerability.py",
    title="Health Programs & Population Vulnerability",
)
upload_custom_analysis = st.Page(
    "dashboards/5_Upload_Custom_Analysis.py",
    title="Upload & Custom Analysis",
)
ask_sentinel_ai = st.Page(
    "dashboards/6_Ask_Sentinel_AI.py",
    title="Ask Sentinel (AI Copilot)",
)

# Flat list -> plain left-nav list, no section header, matching the existing
# look. Add further st.Page(...) entries here for any future dashboard.
pg = st.navigation(
    [
        home,
        executive_overview,
        geographic_environmental,
        laboratory_healthcare,
        outbreak_monitoring,
        health_programs_vulnerability,
        upload_custom_analysis,
        ask_sentinel_ai,
    ]
)

pg.run()

