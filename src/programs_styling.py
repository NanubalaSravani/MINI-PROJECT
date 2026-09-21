# ================================================================
# STYLING helpers specific to the Health Programs & Population
# Vulnerability page.
#
# Deliberately does NOT include the original apply_theme() from this
# dashboard's own submission — the shared src/styling.py already
# injects sidebar/background/KPI-card CSS app-wide (inject_css()),
# and calling a second, competing global stylesheet here would risk
# fighting it. Only the chart palette + style_chart() helper (which
# just themes individual Plotly figures, not the page chrome) are
# kept, values matched to the same brand palette used everywhere
# else in the app.
# ================================================================

from src.chart_colors import HEATMAP_SCALE, HOTSPOT_SCALE, PIE_SEQUENCE, BAR_SEQUENCE, DARK_CATEGORY_COLOURS

NAVY = "#17324D"
TEAL = "#0F6B78"
BACKGROUND = "#F5F7FA"
CARD = "#FFFFFF"
CHARCOAL = "#000000"
SLATE = "#47586B"
GREEN = "#16855B"
AMBER = "#C98A00"
RED = "#C43D3D"

# Soft pastel sequence for pies/lines/scatter with many categories (states,
# years) — matches the pastel palette used for every other pie in the suite.
CATEGORY_COLOURS = PIE_SEQUENCE

# Deep, high-contrast sequence for the two charts on this page that carry
# many state/year labels and need stronger colors (and readable dark legend
# text) than the pastel CATEGORY_COLOURS: the Program Coverage Over Time
# line chart and the Socioeconomic Score vs Health Vulnerability scatter.
DARK_CATEGORY_COLOURS = DARK_CATEGORY_COLOURS

# Clean executive blue/gray/teal sequence for plain categorical bar charts.
BAR_SEQUENCE = BAR_SEQUENCE

# Risk scale — Green (low/safe) -> Yellow (medium/warning) -> Red
# (high/critical). Matches RISK_HEATMAP_SCALE used elsewhere in the app so
# every risk-coded chart/heatmap in the suite reads the same way.
RISK_SCALE = HEATMAP_SCALE

# Vulnerability is a "hotspot" style continuous metric (higher = more
# concerning), so — same as every other hotspot chart in the suite — high
# values read warm and low values read cool, instead of the old flat blue
# gradient that didn't visually flag which states needed attention.
VULNERABILITY_SCALE = HOTSPOT_SCALE


def style_chart(fig):
    """Applied to every chart on this page so they share one look."""
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=CHARCOAL,
        title_font_color=NAVY,
        margin=dict(t=60, b=60),
    )
    fig.update_xaxes(gridcolor="#E3E8EF", zerolinecolor="#E3E8EF")
    fig.update_yaxes(gridcolor="#E3E8EF", zerolinecolor="#E3E8EF")
    return fig
