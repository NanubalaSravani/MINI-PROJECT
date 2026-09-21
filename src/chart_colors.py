"""
chart_colors.py
----------------
Single source of truth for every chart color in the HealthSentinel suite.

Before this module existed, almost every dashboard file (and src/geographic.py)
defined its own local copy of "the brand colors" — several different reds,
three different risk gradients (a pastel green/yellow/red, a teal/amber/red,
and a plain "Reds" scale), and ad-hoc bar colors. That meant the same concept
(e.g. "this state is high-risk") could show up red on one page and dark-teal
on another, which breaks the whole point of color-coding a dashboard.

Every chart-building module should import its colors from here instead of
hardcoding hex values, so a color always carries the same meaning wherever
it appears:

    HEATMAP_SCALE   -> risk-style heatmaps / matrices (Green -> Yellow -> Red)
    HOTSPOT_SCALE   -> choropleths, geo maps, and any "hotspot" style chart
                       where high values should read as warm and low values
                       as cool (Blue/Green -> Yellow -> Orange/Red)
    PIE_SEQUENCE    -> pie / donut charts on light backgrounds (soft pastels)
    PIE_SEVERITY    -> pie/donut charts that specifically encode High/Medium/
                       Low severity — pastel siblings of HEATMAP_SCALE so a
                       severity pie and a severity heatmap agree with each other
    BAR_SEQUENCE    -> plain categorical bar/column charts (clean blues/grays/teal)

Status colors (SUCCESS / WARNING / DANGER) are kept separate on purpose:
outcomes like "Deaths" or "Emergency Alerts" are more readable in their
universally-understood red/amber/green than forced into the blue/gray/teal
bar palette, so they're exposed here too but are not part of BAR_SEQUENCE.
"""

# --------------------------------------------------------------------------- #
# 1) HEATMAPS — natural Green -> Yellow -> Red risk gradient.
#    Kept on the lighter/pastel side so black text stays readable on every
#    cell without needing per-cell text-color logic.
# --------------------------------------------------------------------------- #
HEATMAP_SCALE = [
    [0.0, "#B7E4C7"],   # low / safe -> green
    [0.5, "#FFE699"],   # medium / warning -> yellow
    [1.0, "#F4A199"],   # high / critical -> red
]

# --------------------------------------------------------------------------- #
# 2) HOTSPOTS — choropleths, geo scatter, and "top-N" ranking charts where a
#    single value needs to visually separate low (cool) from high (warm).
#    Same direction and meaning as HEATMAP_SCALE (red = high, green/blue =
#    low) just with a fuller, more map-friendly gradient.
# --------------------------------------------------------------------------- #
HOTSPOT_SCALE = [
    [0.00, "#2E86AB"],   # low -> cool blue
    [0.30, "#63BE7B"],   # low-medium -> green
    [0.55, "#FFE699"],   # medium -> yellow
    [0.78, "#F6A560"],   # medium-high -> orange
    [1.00, "#E14B4B"],   # high -> warm red
]

# --------------------------------------------------------------------------- #
# 3) PIE / DONUT charts — soft pastel tones that read cleanly against the
#    dashboard's light background.
# --------------------------------------------------------------------------- #
PIE_SEQUENCE = [
    "#B7E4C7",  # light green
    "#FFD8B1",  # peach
    "#D6C9F0",  # lavender
    "#A9DEF9",  # sky blue
    "#FFF3B0",  # soft yellow
    "#F6C6D9",  # soft pink
]

# Pastel siblings of HEATMAP_SCALE, for pies that specifically encode
# High / Moderate / Low severity so the pie agrees with any heatmap on the
# same page instead of using the more saturated status colors.
PIE_SEVERITY = {
    "High": "#F4A199",
    "Moderate": "#FFE699",
    "Low": "#B7E4C7",
}

# --------------------------------------------------------------------------- #
# 3b) DARK CATEGORY sequence — for charts carrying many category labels
#    (e.g. states, years) where the soft PIE_SEQUENCE pastels are too light
#    to tell apart or to read comfortably. Deep, high-contrast tones so both
#    the markers/lines and their legend labels stay legible.
# --------------------------------------------------------------------------- #
DARK_CATEGORY_COLOURS = [
    "#1B3A5C",  # deep navy
    "#7A1F1F",  # deep red
    "#1F5C3C",  # deep green
    "#5C3D1F",  # deep brown
    "#4A1F5C",  # deep purple
    "#1F5C5C",  # deep teal
    "#5C4A1F",  # deep olive
    "#3D1F5C",  # deep violet
    "#5C1F3D",  # deep maroon
    "#1F3D5C",  # deep steel blue
    "#2E5C1F",  # deep forest
    "#5C2E1F",  # deep rust
    "#1F5C4A",  # deep emerald
    "#3D5C1F",  # deep moss
    "#5C1F1F",  # deep brick
    "#2A2A2A",  # near-black gray
]

# --------------------------------------------------------------------------- #
# 4) BAR / COLUMN charts — clean, executive shades of blue / gray / teal.
# --------------------------------------------------------------------------- #
BAR_SEQUENCE = [
    "#1F77B4",  # blue
    "#AEC7E8",  # light blue
    "#7F7F7F",  # gray
    "#0F6B78",  # teal
    "#4C8FA8",  # steel blue
    "#C7C7C7",  # light gray
]

# --------------------------------------------------------------------------- #
# Status semantics — reused for outcome-style series (Deaths / Recovered /
# Emergency Alerts / etc.) where the universal red/amber/green reads faster
# than a neutral bar color. Intentionally NOT part of BAR_SEQUENCE.
# --------------------------------------------------------------------------- #
STATUS_SUCCESS = "#16855B"
STATUS_WARNING = "#C98A00"
STATUS_DANGER = "#C43D3D"
