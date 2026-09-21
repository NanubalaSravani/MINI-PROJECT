"""
pdf_report.py
-------------
Builds a downloadable PDF analysis report from an uploaded dataset.

Kept deliberately dependency-light: charts are rendered with matplotlib
(already a project dependency) straight to in-memory PNGs, and the
document itself is assembled with reportlab. No headless-browser /
kaleido step is required, so this works the same in any deployment
environment.
"""

from __future__ import annotations

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable,
)

NAVY = "#17324D"
TEAL = "#0F6B78"
TEXT = "#000000"
MUTED = "#000000"


def _fig_to_image(fig, width_cm=16.5):
    """Render a matplotlib figure to a reportlab Image, then close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=width_cm * cm, height=(width_cm * cm) * (fig.get_figheight() / fig.get_figwidth()))
    return img


def _style_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(axis="y", color="#E4EAF0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def build_pdf_report(
    df: pd.DataFrame,
    file_name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_col: str | None,
    insights: list[str],
) -> bytes:
    """Return the finished PDF report as raw bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor(NAVY), fontSize=20)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor(NAVY), spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], textColor=colors.HexColor(TEXT), fontSize=9.5, leading=14)
    muted = ParagraphStyle("MutedX", parent=styles["BodyText"], textColor=colors.HexColor(MUTED), fontSize=8.5)

    story = []
    story.append(Paragraph("Dataset Analysis Report", title_style))
    story.append(Paragraph(
        f"Source file: <b>{file_name}</b> &nbsp;|&nbsp; Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",
        muted,
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#DCE3EA"), thickness=1))
    story.append(Spacer(1, 12))

    # --- Overview ------------------------------------------------------- #
    story.append(Paragraph("Overview", h2))
    n_rows, n_cols = df.shape
    missing_pct = round(df.isna().mean().mean() * 100, 2)
    overview_data = [
        ["Rows", f"{n_rows:,}", "Columns", f"{n_cols:,}"],
        ["Numeric fields", f"{len(numeric_cols)}", "Categorical fields", f"{len(categorical_cols)}"],
        ["Date field detected", datetime_col or "None", "Missing data", f"{missing_pct}%"],
    ]
    t = Table(overview_data, colWidths=[4.2 * cm, 4.2 * cm, 4.2 * cm, 4.2 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor(MUTED)),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor(MUTED)),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor(TEXT)),
        ("TEXTCOLOR", (3, 0), (3, -1), colors.HexColor(TEXT)),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # --- Key insights ----------------------------------------------------#
    if insights:
        story.append(Paragraph("Key Insights", h2))
        for line in insights:
            story.append(Paragraph(f"&bull;&nbsp; {line}", body))
        story.append(Spacer(1, 6))

    # --- Numeric summary table ------------------------------------------#
    if numeric_cols:
        story.append(Paragraph("Numeric Summary Statistics", h2))
        desc = df[numeric_cols].describe().T[["count", "mean", "std", "min", "max"]].round(2)
        header = ["Column", "Count", "Mean", "Std Dev", "Min", "Max"]
        rows = [header] + [[idx] + list(r) for idx, r in zip(desc.index, desc.values)]
        rows = [[str(v) for v in row] for row in rows]
        tbl = Table(rows, colWidths=[4.2 * cm] + [2.46 * cm] * 5, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(TEXT)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))

    # --- Distribution charts --------------------------------------------#
    if numeric_cols:
        story.append(Paragraph("Distributions", h2))
        story.append(Paragraph(
            "Shaded histograms of the leading numeric fields, used to sanity-check spread, "
            "skew, and outliers before trusting any comparison drawn from them.", muted,
        ))
        story.append(Spacer(1, 6))
        cols_to_plot = numeric_cols[:4]
        n = len(cols_to_plot)
        ncols = 2
        nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(9, 2.6 * nrows))
        # ncols is fixed at 2 above, so plt.subplots() always returns an
        # array of axes here (even for a single chart, since the grid is
        # still 1x2 with one cell deleted below) — always flatten it.
        axes = axes.flatten()
        for i, col in enumerate(cols_to_plot):
            ax = axes[i]
            vals = df[col].dropna()
            ax.hist(vals, bins=24, color=TEAL, alpha=0.75, edgecolor="white", linewidth=0.4)
            ax.set_title(col, fontsize=9.5, color=TEXT, fontweight="bold", loc="left")
            _style_axes(ax)
        for j in range(n, len(axes)):
            fig.delaxes(axes[j])
        fig.tight_layout()
        story.append(_fig_to_image(fig))
        story.append(Spacer(1, 10))

    # --- Correlation heatmap ---------------------------------------------#
    if len(numeric_cols) >= 2:
        story.append(Paragraph("Correlation Between Numeric Fields", h2))
        story.append(Paragraph(
            "Highlights which fields move together (near +1), move oppositely (near -1), "
            "or show no meaningful relationship (near 0).", muted,
        ))
        story.append(Spacer(1, 6))
        corr = df[numeric_cols].corr().round(2)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8, color=TEXT)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=8, color=TEXT)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                v = corr.values[i, j]
                txt_color = "white" if abs(v) > 0.55 else TEXT
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors=TEXT)
        fig.tight_layout()
        story.append(_fig_to_image(fig, width_cm=13))
        story.append(Spacer(1, 10))

    # --- Category comparison (significance-style view) -------------------#
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        vc = df[cat_col].value_counts()
        top_cats = vc[(vc >= 2)].head(8).index.tolist()
        if len(top_cats) >= 2:
            story.append(Paragraph(f'"{num_col}" by "{cat_col}"', h2))
            story.append(Paragraph(
                "Bars show the mean; the black whisker shows one standard deviation, "
                "so a category whose whisker does not overlap another's bar is a meaningful, "
                "not just cosmetic, difference.", muted,
            ))
            story.append(Spacer(1, 6))
            grp = df[df[cat_col].isin(top_cats)].groupby(cat_col)[num_col].agg(["mean", "std"]).reindex(top_cats)
            fig, ax = plt.subplots(figsize=(8.5, 3.4))
            xpos = range(len(grp))
            ax.bar(xpos, grp["mean"], yerr=grp["std"].fillna(0), color=TEAL, alpha=0.85,
                   capsize=3, ecolor="#000000", error_kw={"linewidth": 1})
            ax.set_xticks(list(xpos))
            ax.set_xticklabels([str(c)[:14] for c in grp.index], rotation=30, ha="right", fontsize=8, color=TEXT)
            ax.set_ylabel(num_col, fontsize=9, color=TEXT)
            _style_axes(ax)
            fig.tight_layout()
            story.append(_fig_to_image(fig, width_cm=15))
            story.append(Spacer(1, 10))

    # --- Trend over time ---------------------------------------------------#
    if datetime_col and numeric_cols:
        story.append(Paragraph(f'Trend of "{numeric_cols[0]}" Over Time', h2))
        story.append(Spacer(1, 6))
        ts = df[[datetime_col, numeric_cols[0]]].dropna().sort_values(datetime_col)
        ts = ts.groupby(pd.Grouper(key=datetime_col, freq="MS"))[numeric_cols[0]].mean().dropna()
        if len(ts) >= 2:
            fig, ax = plt.subplots(figsize=(8.5, 3.2))
            ax.plot(ts.index, ts.values, color=TEAL, linewidth=2.2, marker="o", markersize=3.5)
            ax.fill_between(ts.index, ts.values, color=TEAL, alpha=0.15)
            ax.set_ylabel(numeric_cols[0], fontsize=9, color=TEXT)
            _style_axes(ax)
            fig.autofmt_xdate(rotation=30)
            fig.tight_layout()
            story.append(_fig_to_image(fig, width_cm=15))
            story.append(Spacer(1, 10))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#DCE3EA"), thickness=1))
    story.append(Paragraph(
        "Generated automatically by HealthSentinel — Upload & Custom Analysis. "
        "Figures are computed directly from the uploaded file at generation time.",
        muted,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
