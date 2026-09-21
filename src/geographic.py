"""
geographic.py
-------------
Reusable calculations and Plotly builders for the Geographic & Environmental
Intelligence dashboard.

The dashboard page intentionally contains mostly Streamlit orchestration,
while this module contains data preparation, risk helpers, and chart creation.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.chart_colors import HOTSPOT_SCALE


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
}


def get_date_bounds(env: pd.DataFrame):
    min_date = env["full_date"].min().date()
    max_date = env["full_date"].max().date()
    return min_date, max_date


def get_date_window(period, min_date, max_date):
    if period == "All available data":
        return min_date, max_date
    if period == "Latest month":
        return max_date, max_date
    if period == "Past 30 days":
        return max(min_date, max_date - pd.Timedelta(days=29)), max_date
    if period == "Past 90 days":
        return max(min_date, max_date - pd.Timedelta(days=89)), max_date
    if period == "Past 12 months":
        start = max(min_date, (pd.Timestamp(max_date) - pd.DateOffset(months=11)).date())
        return start, max_date
    return min_date, max_date


def filter_data(env, ds, start_date, end_date, region, state_focus, disease_focus):
    env_f = env[
        (env["full_date"].dt.date >= start_date)
        & (env["full_date"].dt.date <= end_date)
    ].copy()
    ds_f = ds[
        (ds["full_date"].dt.date >= start_date)
        & (ds["full_date"].dt.date <= end_date)
    ].copy()

    if region != "All Regions":
        env_f = env_f[env_f["region"] == region]
        ds_f = ds_f[ds_f["region"] == region]

    if state_focus != "All States":
        env_f = env_f[env_f["state_name"] == state_focus]
        ds_f = ds_f[ds_f["state_name"] == state_focus]

    if disease_focus != "All Diseases":
        ds_f = ds_f[ds_f["disease_name"] == disease_focus]

    return env_f, ds_f


def fmt(value, suffix=""):
    if pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K{suffix}"
    return f"{value:,.1f}{suffix}"


def risk_band(value):
    if value < 25:
        return "LOW"
    if value < 50:
        return "MODERATE"
    if value < 65:
        return "HIGH"
    return "VERY HIGH"


def risk_color(value):
    if value < 25:
        return "#16855B"
    if value < 50:
        return "#C98A00"
    if value < 65:
        return "#C98A00"
    return "#C43D3D"


def common_layout(fig, height=390):
    fig.update_layout(
        template="healthsentinel_dark_text",
        height=height,
        margin=dict(l=14, r=14, t=38, b=14),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#000000", size=10),
        title="",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=11),
    )
    fig.update_xaxes(gridcolor="#E8EDF2", zeroline=False)
    fig.update_yaxes(gridcolor="#E8EDF2", zeroline=False)
    return fig


def calculate_kpis(env_f):
    return {
        "geographic_risk": env_f["geographic_risk_score"].mean(),
        "environmental_risk": env_f["environmental_risk_score"].mean(),
        "aqi": env_f["aqi"].mean(),
        "rainfall": env_f["rainfall_mm"].mean(),
        "temperature": env_f["temperature_c"].mean(),
        "water_quality": env_f["water_quality_index"].mean(),
        "sanitation": env_f["sanitation_coverage_pct"].mean(),
        "healthcare_access": env_f["healthcare_accessibility_score"].mean(),
    }


def build_state_map(env_f, ds_f):
    disease_cases = (
        ds_f.groupby("state_name", as_index=False)
        .agg(disease_reported_cases=("urban_cases", "sum"))
    )
    disease_rural = (
        ds_f.groupby("state_name", as_index=False)["rural_cases"]
        .sum()
        .rename(columns={"rural_cases": "rural_reported_cases"})
    )
    disease_cases = disease_cases.merge(disease_rural, on="state_name", how="outer")
    disease_cases["disease_reported_cases"] = (
        disease_cases["disease_reported_cases"].fillna(0)
        + disease_cases["rural_reported_cases"].fillna(0)
    )
    disease_cases = disease_cases[["state_name", "disease_reported_cases"]]

    state_map = (
        env_f.groupby(
            ["state_name", "region", "latitude", "longitude"], as_index=False
        )
        .agg(
            geographic_risk=("geographic_risk_score", "mean"),
            environmental_risk=("environmental_risk_score", "mean"),
            aqi=("aqi", "mean"),
            water_quality=("water_quality_index", "mean"),
            case_rate=("case_rate_per_100k", "mean"),
            hotspots=("hotspot_flag", "sum"),
        )
    )

    state_map = state_map.merge(disease_cases, on="state_name", how="left")
    state_map["disease_reported_cases"] = state_map["disease_reported_cases"].fillna(0)
    state_map["risk_band"] = state_map["geographic_risk"].apply(risk_band)

    label_df = state_map.nlargest(5, "geographic_risk").copy()
    state_map["display_label"] = ""
    state_map.loc[label_df.index, "display_label"] = label_df["state_name"]

    fig = px.scatter_geo(
        state_map,
        lat="latitude",
        lon="longitude",
        size="geographic_risk",
        color="geographic_risk",
        text="display_label",
        hover_name="state_name",
        hover_data={
            "region": True,
            "geographic_risk": ":.1f",
            "environmental_risk": ":.1f",
            "aqi": ":.0f",
            "water_quality": ":.1f",
            "case_rate": ":.1f",
            "hotspots": True,
            "latitude": False,
            "longitude": False,
            "display_label": False,
        },
        color_continuous_scale=HOTSPOT_SCALE,
        labels={
            "geographic_risk": "Geographic Risk",
            "environmental_risk": "Environmental Risk",
            "case_rate": "Case Rate / 100K",
            "water_quality": "Water Quality",
            "hotspots": "Hotspot Observations",
        },
        projection="mercator",
    )
    fig.update_traces(
        marker=dict(opacity=.86, line=dict(width=1.2, color="#FFFFFF")),
        textfont=dict(size=9, color="#000000"),
    )
    fig.update_geos(
        scope="asia",
        center=dict(lat=22.5, lon=79.5),
        projection_scale=3.9,
        showland=True,
        landcolor="#E9EEF2",
        showocean=True,
        oceancolor="#F5F7FA",
        showcountries=True,
        countrycolor="#B8C4CF",
        showcoastlines=True,
        coastlinecolor="#D9E1E8",
        showlakes=True,
        lakecolor="#F5F7FA",
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Risk", thickness=12, len=.62, bgcolor="rgba(0,0,0,0)"
        )
    )
    return state_map, common_layout(fig, 485)


def build_hotspot_chart(state_map):
    top = state_map.nlargest(10, "geographic_risk").sort_values("geographic_risk")
    fig = px.bar(
        top,
        x="geographic_risk",
        y="state_name",
        orientation="h",
        text="geographic_risk",
        color="geographic_risk",
        color_continuous_scale=HOTSPOT_SCALE,
        hover_data=["environmental_risk", "aqi", "hotspots"],
        labels={"geographic_risk": "Geographic Risk", "state_name": ""},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False)
    return common_layout(fig, 385)


def calculate_pressure(env_f):
    return {
        "AQI pressure": np.clip((env_f["aqi"].mean() / 300) * 100, 0, 100),
        "Mosquito pressure": np.clip(
            (env_f["mosquito_breeding_index"].mean() / 150) * 100, 0, 100
        ),
        "Water stress": np.clip(100 - env_f["water_quality_index"].mean(), 0, 100),
        "Sanitation gap": np.clip(100 - env_f["sanitation_coverage_pct"].mean(), 0, 100),
        "Access gap": np.clip(
            100 - env_f["healthcare_accessibility_score"].mean(), 0, 100
        ),
    }


def build_radar(pressure):
    keys = list(pressure.keys())
    values = list(pressure.values())
    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=keys + [keys[0]],
            fill="toself",
            line=dict(color="#0F6B78", width=2),
            fillcolor="rgba(15,107,120,.14)",
            name="Pressure",
        )
    )
    fig.update_layout(
        template="healthsentinel_dark_text",
        height=385,
        margin=dict(l=35, r=35, t=25, b=25),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], showticklabels=False, gridcolor="#E8EDF2"),
            angularaxis=dict(gridcolor="#E8EDF2", tickfont=dict(size=9)),
        ),
        showlegend=False,
    )
    return fig


def build_gauge(environmental_risk):
    color = risk_color(environmental_risk)
    band = risk_band(environmental_risk)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=environmental_risk,
            number={"font": {"size": 35, "color": "#000000"}},
            title={
                "text": f"<b>{band}</b><br><span style='font-size:10px;color:#000000'>Environmental Risk</span>",
                "font": {"size": 14, "color": color},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)"},
                "bar": {"color": color, "thickness": .24},
                "bgcolor": "rgba(255,255,255,.035)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(72,224,192,.08)"},
                    {"range": [25, 50], "color": "rgba(245,199,93,.08)"},
                    {"range": [50, 65], "color": "rgba(255,158,85,.08)"},
                    {"range": [65, 100], "color": "rgba(255,107,120,.08)"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "thickness": .72, "value": environmental_risk},
            },
        )
    )
    fig.update_layout(
        template="healthsentinel_dark_text",
        height=385,
        margin=dict(l=12, r=12, t=70, b=10),
        title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_urban_rural(ds_f):
    ur = (
        ds_f.groupby("year_month", as_index=False)
        .agg(Urban=("urban_cases", "sum"), Rural=("rural_cases", "sum"))
        .melt(id_vars="year_month", var_name="Settlement", value_name="Cases")
    )
    fig = px.bar(
        ur,
        x="year_month",
        y="Cases",
        color="Settlement",
        barmode="stack",
        color_discrete_map={"Urban": "#0F6B78", "Rural": "#5d8fd3"},
        labels={"year_month": "", "Cases": "Reported Cases", "Settlement": ""},
    )
    common_layout(fig, 390)
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


def build_environmental_trend(env_f, metric):
    trend_map = {
        "AQI": ("aqi", "AQI"),
        "Rainfall": ("rainfall_mm", "Rainfall (mm)"),
        "Temperature": ("temperature_c", "Temperature (°C)"),
        "Water Quality": ("water_quality_index", "Water Quality"),
        "Environmental Risk": ("environmental_risk_score", "Environmental Risk"),
    }
    col, ylab = trend_map[metric]
    trend = (
        env_f.groupby("full_date", as_index=False)[col]
        .mean()
        .sort_values("full_date")
    )
    fig = px.line(
        trend,
        x="full_date",
        y=col,
        markers=True,
        labels={"full_date": "", col: ylab},
    )
    fig.update_traces(line_width=2.4, marker_size=5, line_color="#0F6B78")
    common_layout(fig, 390)
    fig.update_yaxes(title=ylab)
    return fig


def build_water_scatter(env_f):
    scatter = (
        env_f.groupby(["state_name", "region"], as_index=False)
        .agg(
            water_quality=("water_quality_index", "mean"),
            zoonotic_incidence=("zoonotic_disease_incidence", "mean"),
            case_rate=("case_rate_per_100k", "mean"),
            environmental_risk=("environmental_risk_score", "mean"),
            sanitation=("sanitation_coverage_pct", "mean"),
        )
    )

    if len(scatter) < 2:
        return None, scatter, None

    corr = scatter["water_quality"].corr(scatter["zoonotic_incidence"])
    fig = px.scatter(
        scatter,
        x="water_quality",
        y="zoonotic_incidence",
        size="case_rate",
        color="environmental_risk",
        hover_name="state_name",
        hover_data={
            "region": True,
            "water_quality": ":.1f",
            "zoonotic_incidence": ":.1f",
            "case_rate": ":.1f",
            "environmental_risk": ":.1f",
            "sanitation": ":.1f",
        },
        color_continuous_scale=[[0, "#0F6B78"], [.5, "#C98A00"], [1, "#C43D3D"]],
        labels={
            "water_quality": "Water Quality Index",
            "zoonotic_incidence": "Zoonotic Incidence",
            "case_rate": "Case Rate / 100K",
            "environmental_risk": "Environmental Risk",
        },
    )
    common_layout(fig, 430)
    fig.update_xaxes(title="Water Quality Index")
    fig.update_yaxes(title="Zoonotic Incidence")
    fig.add_vline(x=scatter["water_quality"].mean(), line_dash="dot", line_color="#B8C4CF")
    fig.add_hline(y=scatter["zoonotic_incidence"].mean(), line_dash="dot", line_color="#B8C4CF")
    return fig, scatter, corr


def decision_snapshot(state_map):
    return (
        state_map.nlargest(1, "geographic_risk").iloc[0],
        state_map.nlargest(1, "aqi").iloc[0],
        state_map.nsmallest(1, "water_quality").iloc[0],
    )
