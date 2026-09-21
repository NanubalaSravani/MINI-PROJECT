"""
report_generator.py
-------------------
Implements floating report generation icon + selection dialog on every dashboard.
Generates PDF reports containing user-selected charts and visuals, styled with brand guidelines.
"""

import io
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable,
)

# Colors matching src/styling.py
PRIMARY = "#17324D"
ACCENT = "#0F6B78"
TEXT = "#000000"
MUTED = "#000000"

# Matplotlib styling helper
def _style_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(axis="y", color="#E4EAF0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

def _fig_to_image(fig, width_cm=16.5):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=(width_cm * cm) * (fig.get_figheight() / fig.get_figwidth()))

# List of charts by dashboard page
AVAILABLE_CHARTS = {
    "Executive Public Health Overview": {
        "monthly_trend": "Monthly Disease Trend (Line)",
        "disease_dist": "Disease Distribution (Pie)",
        "outcome_comp": "Monthly Outcome Comparison (Bar)",
        "hosp_icu_rate": "Hospitalization & ICU Admission Rate (Bar)",
        "recovery_mortality": "Recovery vs Mortality Rate by State (Scatter)",
        "disease_trend": "Disease-wise Case Trend (Line)",
        "top_burden": "Top Diseases by Burden (Bar)",
        "reports_source": "Reports by Source (Pie)",
        "testing_positivity": "Testing Volume vs Positivity Rate (Scatter)"
    },
    "Geographic & Environmental Intelligence": {
        "geo_hotspots": "Geographic Hotspots (Bar)",
        "env_fingerprint": "Environmental Risk Fingerprint (Radar/Bar)",
        "risk_gauge": "Environmental Risk Gauge (Bar)",
        "urban_rural": "Urban vs Rural Disease Burden (Bar)",
        "env_trends": "Environmental Indicators Trend (Line)",
        "water_zoonotic": "Water Quality vs Zoonotic Incidence (Scatter)"
    },
    "Laboratory & Healthcare Capacity": {
        "testing_trend": "Testing Trend (Bar/Line)",
        "vaccination_progress": "Vaccination Progress (Line)",
        "hospital_capacity": "Hospital Capacity (Bar)",
        "icu_utilization": "Average ICU Utilization Rate (Bar/Line)",
        "lab_performance": "Laboratory Performance (Bar/Line)",
        "bed_occupancy": "Bed Occupancy by State (Heatmap/Bar)"
    },
    "Outbreak Monitoring & Forecasting": {
        "outbreak_surveillance": "Outbreak Surveillance Insights (Line)",
        "top_hotspots": "Top Hotspots & Vector Threats (Bar)",
        "arima_model": "ARIMA Predictive Modeling (Line)",
        "priority_matrix": "Priority Containment Matrix (Scatter/Bar)"
    },
    "Health Programs & Population Vulnerability": {
        "program_coverage": "Program Coverage Over Time (Line)",
        "reach_by_state": "Screened vs Beneficiaries Reached (Bar)",
        "pop_dist": "Population Distribution by State (Bar)",
        "high_risk_state": "High-Risk Population by State (Bar)",
        "vuln_index": "Health Vulnerability Index (Bar)",
        "socio_vs_vuln": "Socioeconomic Score vs Health Vulnerability (Scatter)"
    },
    "Upload & Custom Analysis": {
        "overview_summary": "Dataset Statistics Overview",
        "numeric_summary": "Numeric Summary Statistics (Table)",
        "dist_charts": "Leading Numeric Distributions (Histograms)",
        "corr_heatmap": "Correlation Heatmap",
        "cat_comparison": "Category Comparison (Bar)",
        "trend_time": "Trend Over Time (Line)"
    }
}

# Matplotlib Chart Builders
def draw_chart(chart_id, df, extra_data=None):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    
    # ----------------------------------------------------
    # EXECUTIVE PUBLIC HEALTH OVERVIEW CHARTS
    # ----------------------------------------------------
    if chart_id == "monthly_trend":
        trend = df.groupby(["year", "month_num", "month_name"], as_index=False)["total_reported_cases"].sum().sort_values(["year", "month_num"])
        trend["period"] = trend["month_name"].str[:3] + " " + trend["year"].astype(str)
        ax.plot(trend["period"], trend["total_reported_cases"], marker="o", color=ACCENT, linewidth=2.5)
        ax.fill_between(trend["period"], trend["total_reported_cases"], color=ACCENT, alpha=0.15)
        ax.set_title("Monthly Disease Trend", fontsize=11, fontweight="bold", color=PRIMARY)
        ax.set_ylabel("Total Reported Cases")
        _style_axes(ax)
        fig.autofmt_xdate(rotation=30)
        
    elif chart_id == "disease_dist":
        dist = df.groupby("disease_category")["total_reported_cases"].sum()
        ax.pie(dist, labels=dist.index, autopct="%1.1f%%", colors=["#17324D", "#0F6B78", "#3B8E93", "#7BB4B0", "#B7D9CE"], startangle=90)
        ax.set_title("Disease Distribution", fontsize=11, fontweight="bold", color=PRIMARY)
        
    elif chart_id == "outcome_comp":
        outcome = df.groupby(["year", "month_num", "month_name"], as_index=False).agg(
            active_cases=("active_cases", "sum"),
            recovered_cases=("recovered_cases", "sum"),
            deaths=("deaths", "sum")
        ).sort_values(["year", "month_num"])
        outcome["period"] = outcome["month_name"].str[:3] + " " + outcome["year"].astype(str)
        x = range(len(outcome))
        width = 0.25
        ax.bar([i - width for i in x], outcome["active_cases"], width, label="Active Cases", color="#C98A00")
        ax.bar(x, outcome["recovered_cases"], width, label="Recovered Cases", color="#16855B")
        ax.bar([i + width for i in x], outcome["deaths"], width, label="Deaths", color="#C43D3D")
        ax.set_xticks(x)
        ax.set_xticklabels(outcome["period"], rotation=30)
        ax.legend()
        ax.set_title("Monthly Outcome Comparison", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        
    elif chart_id == "hosp_icu_rate":
        hosp_by_state = df.groupby("state_name", as_index=False).agg(
            hospitalized_cases=("hospitalized_cases", "sum"),
            icu_admissions=("icu_admissions", "sum"),
            total_reported_cases=("total_reported_cases", "sum")
        )
        hosp_by_state["Hosp Rate"] = (hosp_by_state["hospitalized_cases"] / hosp_by_state["total_reported_cases"] * 100).fillna(0)
        hosp_by_state["ICU Rate"] = (hosp_by_state["icu_admissions"] / hosp_by_state["total_reported_cases"] * 100).fillna(0)
        hosp_by_state = hosp_by_state.sort_values("Hosp Rate", ascending=False).head(10)
        x = range(len(hosp_by_state))
        width = 0.35
        ax.bar([i - width/2 for i in x], hosp_by_state["Hosp Rate"], width, label="Hospitalization Rate", color=PRIMARY)
        ax.bar([i + width/2 for i in x], hosp_by_state["ICU Rate"], width, label="ICU Admission Rate", color="#C43D3D")
        ax.set_xticks(x)
        ax.set_xticklabels(hosp_by_state["state_name"], rotation=45, ha="right")
        ax.set_ylabel("Rate (%)")
        ax.legend()
        ax.set_title("Hospitalization & ICU Admission Rate", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)

    elif chart_id == "recovery_mortality":
        scatter_df = df.groupby("state_name", as_index=False).agg(
            total_reported_cases=("total_reported_cases", "sum"),
            deaths=("deaths", "sum"),
            recovered_cases=("recovered_cases", "sum")
        )
        scatter_df["Recovery Rate"] = ((scatter_df["recovered_cases"] / scatter_df["total_reported_cases"]) * 100).clip(upper=100).fillna(0)
        scatter_df["Mortality Rate"] = ((scatter_df["deaths"] / scatter_df["total_reported_cases"]) * 100).fillna(0)
        sc = ax.scatter(scatter_df["Recovery Rate"], scatter_df["Mortality Rate"], s=scatter_df["total_reported_cases"]/50 + 20, c=scatter_df["Mortality Rate"], cmap="Reds", alpha=0.75, edgecolors="grey")
        ax.set_xlabel("Recovery Rate (%)")
        ax.set_ylabel("Mortality Rate (%)")
        ax.set_title("Recovery vs Mortality Rate by State", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        fig.colorbar(sc, ax=ax, label="Mortality Rate (%)")
        
    elif chart_id == "disease_trend":
        trend = df.groupby(["year", "month_num", "month_name", "disease_name"], as_index=False)["total_reported_cases"].sum().sort_values(["year", "month_num"])
        trend["period"] = trend["month_name"].str[:3] + " " + trend["year"].astype(str)
        for disease in trend["disease_name"].unique()[:5]:
            sub = trend[trend["disease_name"] == disease]
            ax.plot(sub["period"], sub["total_reported_cases"], marker="o", label=disease)
        ax.legend()
        ax.set_title("Disease-wise Case Trend (Top 5 Diseases)", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        fig.autofmt_xdate(rotation=30)
        
    elif chart_id == "top_burden":
        burden = df.groupby("disease_name")["total_reported_cases"].sum().sort_values(ascending=False).head(10)
        ax.barh(burden.index[::-1], burden.values[::-1], color=ACCENT)
        ax.set_title("Top Diseases by Burden", fontsize=11, fontweight="bold", color=PRIMARY)
        ax.set_xlabel("Total Reported Cases")
        _style_axes(ax)
        
    elif chart_id == "reports_source":
        source = df.groupby("source_name")["total_reported_cases"].sum()
        ax.pie(source, labels=source.index, autopct="%1.1f%%", colors=["#17324D", "#0F6B78", "#3B8E93", "#7BB4B0"])
        ax.set_title("Reports by Source", fontsize=11, fontweight="bold", color=PRIMARY)
        
    elif chart_id == "testing_positivity":
        state_df = df.groupby("state_name", as_index=False).agg(
            total_reported_cases=("total_reported_cases", "sum"),
            recovered_cases=("recovered_cases", "sum")
        )
        state_df["rate"] = (state_df["recovered_cases"] / state_df["total_reported_cases"] * 100).fillna(0)
        ax.scatter(state_df["total_reported_cases"], state_df["rate"], color=ACCENT, alpha=0.8, s=80)
        ax.set_xlabel("Total Reported Cases")
        ax.set_ylabel("Rate Proxy (%)")
        ax.set_title("Volume vs Outcome Rate by State", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)

    # ----------------------------------------------------
    # GEOGRAPHIC & ENVIRONMENTAL INTELLIGENCE CHARTS
    # ----------------------------------------------------
    elif chart_id == "geo_hotspots":
        if "geographic_risk" in df.columns:
            hotspots = df.groupby("state_name")["geographic_risk"].mean().sort_values(ascending=False).head(10)
        elif "environmental_risk" in df.columns:
            hotspots = df.groupby("state_name")["environmental_risk"].mean().sort_values(ascending=False).head(10)
        else:
            hotspots = df.groupby("state_name").size().sort_values(ascending=False).head(10)
        ax.bar(hotspots.index, hotspots.values, color=PRIMARY)
        ax.set_title("Geographic Hotspots (Risk by State)", fontsize=11, fontweight="bold", color=PRIMARY)
        ax.set_ylabel("Average Risk Score")
        _style_axes(ax)
        fig.autofmt_xdate(rotation=30)
        
    elif chart_id == "env_fingerprint":
        metrics = ["aqi", "water_quality", "sanitation", "healthcare_access"]
        available_metrics = [m for m in metrics if m in df.columns]
        if available_metrics:
            means = df[available_metrics].mean()
            ax.bar(means.index, means.values, color=ACCENT)
            ax.set_title("Environmental Pressures Summary", fontsize=11, fontweight="bold", color=PRIMARY)
            _style_axes(ax)
            
    elif chart_id == "risk_gauge":
        risk_val = df["environmental_risk"].mean() if "environmental_risk" in df.columns else 5.0
        ax.bar(["Composite Risk"], [risk_val], color="#C43D3D", width=0.4)
        ax.set_ylim(0, 10)
        ax.set_title(f"Average Environmental Risk Score: {risk_val:.1f}/10", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        
    elif chart_id == "urban_rural":
        urban_cols = [c for c in ["urban_cases", "rural_cases"] if c in df.columns]
        if urban_cols:
            sums = df[urban_cols].sum()
            ax.bar(sums.index, sums.values, color=[PRIMARY, ACCENT])
            ax.set_title("Urban vs Rural Case Count", fontsize=11, fontweight="bold", color=PRIMARY)
            _style_axes(ax)
            
    elif chart_id == "env_trends":
        indicator = extra_data if extra_data else "AQI"
        col = indicator.lower().replace(" ", "_")
        if col in df.columns:
            trend = df.groupby(df.index.to_series().dt.strftime("%Y-%m") if hasattr(df.index, "dt") else "month_name")[col].mean()
            ax.plot(trend.index, trend.values, marker="o", color=ACCENT, linewidth=2.5)
            ax.set_title(f"Environmental Trend: {indicator}", fontsize=11, fontweight="bold", color=PRIMARY)
            _style_axes(ax)
            fig.autofmt_xdate(rotation=30)

    elif chart_id == "water_zoonotic":
        if "water_quality" in df.columns and "environmental_risk" in df.columns:
            sc = ax.scatter(df["water_quality"], df.get("cases_rate", df.index), s=50, c=df["environmental_risk"], cmap="YlOrRd")
            ax.set_xlabel("Water Quality Index")
            ax.set_ylabel("Case Rate / Index")
            ax.set_title("Water Quality vs Zoonotic Indicator", fontsize=11, fontweight="bold", color=PRIMARY)
            fig.colorbar(sc, ax=ax, label="Environmental Risk")
            _style_axes(ax)

    # ----------------------------------------------------
    # LABORATORY & HEALTHCARE CAPACITY CHARTS
    # ----------------------------------------------------
    elif chart_id == "testing_trend":
        trend = df.groupby(["year", "month_name"], as_index=False).agg(
            total_tests=("total_tests", "sum"),
            positive_tests=("positive_tests", "sum")
        )
        x = range(len(trend))
        ax.bar(x, trend["total_tests"], label="Total Tests", color="#CBD5E1")
        ax.plot(x, trend["positive_tests"], marker="o", label="Positive Tests", color="#C43D3D", linewidth=2.5)
        ax.set_xticks(x)
        ax.set_xticklabels(trend["month_name"], rotation=30)
        ax.legend()
        ax.set_title("Testing Volume & Positivity Trend", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        
    elif chart_id == "vaccination_progress":
        trend = df.groupby(["year", "month_name"], as_index=False).agg(
            vax=("vaccination_coverage_pct", "mean"),
            booster=("booster_coverage_pct", "mean")
        )
        ax.plot(trend["month_name"], trend["vax"], marker="o", label="Vaccine Coverage (%)", color=ACCENT)
        ax.plot(trend["month_name"], trend["booster"], marker="s", label="Booster Coverage (%)", color=PRIMARY)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.set_title("Vaccination Progress Over Time", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        
    elif chart_id == "hospital_capacity":
        cap = df.groupby("state_name")[["hospital_beds", "doctors"]].mean().head(10)
        x = range(len(cap))
        ax.bar(x, cap["hospital_beds"], label="Beds", color=PRIMARY)
        ax.set_xticks(x)
        ax.set_xticklabels(cap.index, rotation=30)
        ax.legend()
        ax.set_title("Average Bed Capacity by State (Top 10)", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        
    elif chart_id == "icu_utilization":
        icu = df.groupby("state_name")["icu_utilization_pct"].mean().sort_values(ascending=False).head(10)
        ax.bar(icu.index, icu.values, color="#C43D3D")
        ax.set_ylabel("ICU Utilization (%)")
        ax.set_title("Top 10 States by ICU Utilization", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        fig.autofmt_xdate(rotation=30)
        
    elif chart_id == "lab_performance":
        lab = df.groupby("state_name")["turnaround_time_days"].mean().sort_values().head(10)
        ax.bar(lab.index, lab.values, color=ACCENT)
        ax.set_ylabel("Average Turnaround Time (Days)")
        ax.set_title("Top States by Quickest Lab Turnaround Time", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        fig.autofmt_xdate(rotation=30)

    # ----------------------------------------------------
    # OUTBREAK MONITORING & FORECASTING CHARTS
    # ----------------------------------------------------
    elif chart_id == "outbreak_surveillance":
        cases = df.groupby("disease_name")["total_reported_cases"].sum().sort_values(ascending=False).head(5)
        ax.bar(cases.index, cases.values, color=ACCENT)
        ax.set_title("Outbreak Cases by Top Diseases", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        fig.autofmt_xdate(rotation=30)
        
    elif chart_id == "top_hotspots":
        hotspots = df.groupby("state_name")["total_reported_cases"].sum().sort_values(ascending=False).head(10)
        ax.bar(hotspots.index, hotspots.values, color=PRIMARY)
        ax.set_title("Top Outbreak Hotspots by Case Count", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        fig.autofmt_xdate(rotation=30)
        
    elif chart_id == "arima_model":
        ts = df.groupby(["year", "month_num"], as_index=False)["total_reported_cases"].sum().sort_values(["year", "month_num"])
        ax.plot(range(len(ts)), ts["total_reported_cases"], label="Observed", color=PRIMARY, marker="o")
        proj = ts["total_reported_cases"].rolling(window=2).mean().shift(-1)
        ax.plot(range(len(ts)), proj, label="Projection", color=ACCENT, linestyle="--", marker="x")
        ax.legend()
        ax.set_title("Outbreak Case Projections", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        
    elif chart_id == "priority_matrix":
        ax.scatter(df["public_health_risk_score"], df["total_reported_cases"], color="#C43D3D", alpha=0.7, s=50)
        ax.set_xlabel("Public Health Risk Score")
        ax.set_ylabel("Reported Cases")
        ax.set_title("Priority Containment Scatter", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)

    # ----------------------------------------------------
    # HEALTH PROGRAMS & POPULATION VULNERABILITY CHARTS
    # ----------------------------------------------------
    elif chart_id == "program_coverage":
        trend = df.groupby(["Year", "Month"], as_index=False)["ProgramCoverage"].mean().sort_values(["Year", "Month"])
        for year in trend["Year"].unique():
            sub = trend[trend["Year"] == year]
            ax.plot(sub["Month"], sub["ProgramCoverage"], marker="o", label=f"Year {year}")
        ax.legend()
        ax.set_ylabel("Coverage (%)")
        ax.set_title("Program Coverage Over Time", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        
    elif chart_id == "reach_by_state":
        reach = df.groupby("State")[["PeopleScreened", "BeneficiariesReached"]].sum().head(10)
        x = range(len(reach))
        width = 0.35
        ax.bar([i - width/2 for i in x], reach["PeopleScreened"], width, label="Screened", color=ACCENT)
        ax.bar([i + width/2 for i in x], reach["BeneficiariesReached"], width, label="Reached", color=PRIMARY)
        ax.set_xticks(x)
        ax.set_xticklabels(reach.index, rotation=30)
        ax.legend()
        ax.set_title("Program Screening vs Beneficiaries Reached", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        
    elif chart_id == "pop_dist":
        pop = df.groupby("State")[["Children", "Adults", "Elderly"]].mean().head(10)
        x = range(len(pop))
        ax.bar(x, pop["Children"], label="Children", color="#CBD5E1")
        ax.bar(x, pop["Adults"], bottom=pop["Children"], label="Adults", color=ACCENT)
        ax.bar(x, pop["Elderly"], bottom=pop["Children"]+pop["Adults"], label="Elderly", color=PRIMARY)
        ax.set_xticks(x)
        ax.set_xticklabels(pop.index, rotation=30)
        ax.legend()
        ax.set_title("Average Population Distribution", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        
    elif chart_id == "high_risk_state":
        hr = df.groupby("State")["HighRiskIndividuals"].sum().sort_values(ascending=False).head(10)
        ax.bar(hr.index, hr.values, color="#C43D3D")
        ax.set_ylabel("Total High-Risk Individuals")
        ax.set_title("High-Risk Population by State (Top 10)", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        fig.autofmt_xdate(rotation=30)
        
    elif chart_id == "vuln_index":
        vuln = df.groupby("State")["HealthVulnerabilityIndex"].mean().sort_values(ascending=False).head(10)
        ax.bar(vuln.index, vuln.values, color=ACCENT)
        ax.set_ylabel("Vulnerability Index")
        ax.set_title("Top 10 States by Health Vulnerability Index", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)
        fig.autofmt_xdate(rotation=30)
        
    elif chart_id == "socio_vs_vuln":
        ax.scatter(df["SocioeconomicScore"], df["HealthVulnerabilityIndex"], color=PRIMARY, alpha=0.7, s=60)
        ax.set_xlabel("Socioeconomic Score")
        ax.set_ylabel("Health Vulnerability Index")
        ax.set_title("Socioeconomic Score vs Health Vulnerability", fontsize=11, fontweight="bold", color=PRIMARY)
        _style_axes(ax)

    # ----------------------------------------------------
    # UPLOAD & CUSTOM ANALYSIS CHARTS
    # ----------------------------------------------------
    elif chart_id == "dist_charts":
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cols_to_plot = numeric_cols[:4]
        n = len(cols_to_plot)
        if n > 0:
            fig, axes = plt.subplots((n+1)//2, min(n, 2), figsize=(9, 2.6 * ((n+1)//2)))
            axes = axes.flatten() if n > 1 else [axes]
            for i, col in enumerate(cols_to_plot):
                ax_sub = axes[i]
                vals = df[col].dropna()
                ax_sub.hist(vals, bins=24, color=ACCENT, alpha=0.75, edgecolor="white", linewidth=0.4)
                ax_sub.set_title(col, fontsize=9.5, color=PRIMARY, fontweight="bold", loc="left")
                _style_axes(ax_sub)
            for j in range(n, len(axes)):
                fig.delaxes(axes[j])
            fig.tight_layout()

    elif chart_id == "corr_heatmap":
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().round(2)
            fig, ax_sub = plt.subplots(figsize=(7.5, 6))
            im = ax_sub.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax_sub.set_xticks(range(len(corr.columns)))
            ax_sub.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
            ax_sub.set_yticks(range(len(corr.index)))
            ax_sub.set_yticklabels(corr.index, fontsize=8)
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    v = corr.values[i, j]
                    txt_color = "white" if abs(v) > 0.55 else TEXT
                    ax_sub.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color)
            fig.colorbar(im, ax=ax_sub, fraction=0.046, pad=0.04)
            fig.tight_layout()

    elif chart_id == "cat_comparison":
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if cat_cols and numeric_cols:
            cat_col = cat_cols[0]
            num_col = numeric_cols[0]
            vc = df[cat_col].value_counts()
            top_cats = vc[(vc >= 2)].head(8).index.tolist()
            if len(top_cats) >= 2:
                grp = df[df[cat_col].isin(top_cats)].groupby(cat_col)[num_col].agg(["mean", "std"]).reindex(top_cats)
                xpos = range(len(grp))
                ax.bar(xpos, grp["mean"], yerr=grp["std"].fillna(0), color=ACCENT, alpha=0.85, capsize=3)
                ax.set_xticks(list(xpos))
                ax.set_xticklabels([str(c)[:14] for c in grp.index], rotation=30, ha="right", fontsize=8)
                ax.set_ylabel(num_col)
                ax.set_title(f"{num_col} by {cat_col}", fontsize=11, fontweight="bold", color=PRIMARY)
                _style_axes(ax)

    elif chart_id == "trend_time":
        dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not dt_cols:
            # check string cols that look like date
            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    dt_cols.append(col)
                    break
                except:
                    pass
        if dt_cols and numeric_cols:
            datetime_col = dt_cols[0]
            ts = df[[datetime_col, numeric_cols[0]]].dropna().sort_values(datetime_col)
            ts = ts.groupby(pd.Grouper(key=datetime_col, freq="MS"))[numeric_cols[0]].mean().dropna()
            if len(ts) >= 2:
                ax.plot(ts.index, ts.values, color=ACCENT, linewidth=2.2, marker="o")
                ax.fill_between(ts.index, ts.values, color=ACCENT, alpha=0.15)
                ax.set_ylabel(numeric_cols[0])
                ax.set_title(f"Trend of {numeric_cols[0]} Over Time", fontsize=11, fontweight="bold", color=PRIMARY)
                _style_axes(ax)
                fig.autofmt_xdate(rotation=30)

    return fig

# Custom report PDF constructor
def build_custom_pdf(page_name, df, selected_charts, active_filters):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor(PRIMARY), fontSize=18, alignment=0)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor(PRIMARY), spaceBefore=12, spaceAfter=6, fontSize=12)
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], textColor=colors.HexColor(TEXT), fontSize=9, leading=13)
    muted = ParagraphStyle("MutedX", parent=styles["BodyText"], textColor=colors.HexColor(MUTED), fontSize=8)

    story = []
    
    # Header
    story.append(Paragraph(f"Custom Report: {page_name}", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;|&nbsp; Target Dashboard: {page_name}",
        muted,
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#DCE3EA"), thickness=1))
    story.append(Spacer(1, 10))

    # Active Filters block
    if active_filters:
        story.append(Paragraph("Applied Dashboard Filters", h2))
        filter_rows = []
        keys = list(active_filters.keys())
        for i in range(0, len(keys), 2):
            k1 = keys[i]
            v1 = str(active_filters[k1])
            k2 = keys[i+1] if i+1 < len(keys) else ""
            v2 = str(active_filters[k2]) if i+1 < len(keys) else ""
            filter_rows.append([Paragraph(f"<b>{k1}:</b> {v1}", body), Paragraph(f"<b>{k2}:</b> {v2}" if k2 else "", body)])
            
        if filter_rows:
            t = Table(filter_rows, colWidths=[8.2 * cm, 8.2 * cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
        story.append(Spacer(1, 12))

    # Selected visuals
    for idx, chart_id in enumerate(selected_charts):
        chart_title = AVAILABLE_CHARTS[page_name].get(chart_id, "Visual Insight")
        story.append(Paragraph(f"{idx+1}. {chart_title}", h2))
        story.append(Spacer(1, 4))
        
        if chart_id == "overview_summary":
            n_rows, n_cols = df.shape
            overview_data = [
                ["Rows", f"{n_rows:,}", "Columns", f"{n_cols:,}"],
            ]
            t = Table(overview_data, colWidths=[8.2 * cm, 8.2 * cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(t)
        elif chart_id == "numeric_summary":
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                desc = df[numeric_cols].describe().T[["count", "mean", "std", "min", "max"]].round(2)
                header = ["Column", "Count", "Mean", "Std Dev", "Min", "Max"]
                rows = [header] + [[idx] + list(r) for idx, r in zip(desc.index, desc.values)]
                rows = [[str(v) for v in row] for row in rows]
                tbl = Table(rows, colWidths=[4.2 * cm] + [2.46 * cm] * 5, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(TEXT)),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ]))
                story.append(tbl)
        else:
            try:
                fig = draw_chart(chart_id, df)
                img = _fig_to_image(fig)
                story.append(img)
            except Exception as e:
                story.append(Paragraph(f"<i>Could not render chart: {str(e)}</i>", body))
            
        story.append(Spacer(1, 12))
        if (idx + 1) % 2 == 0 and idx + 1 < len(selected_charts):
            story.append(PageBreak())

    # Footer
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#DCE3EA"), thickness=1))
    story.append(Paragraph(
        "Report automatically compiled by HealthSentinel. All figures represent the active filters selected at generation time.",
        muted,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# Streamlit interactive modal
@st.dialog("Generate Custom Report")
def show_report_generator_dialog(page_name, df, active_filters):
    st.write("### Choose visuals to include in your report:")
    charts = AVAILABLE_CHARTS.get(page_name, {})
    
    selected_charts = []
    for chart_id, chart_label in charts.items():
        if st.checkbox(chart_label, value=True, key=f"rpt_chk_{chart_id}"):
            selected_charts.append(chart_id)
            
    st.markdown("---")
    if not selected_charts:
        st.warning("Please select at least one visual to compile.")
        return

    # NOTE: st.download_button triggers its own script rerun when clicked.
    # If it (and the bytes it serves) only exist inside `if st.button(...)`,
    # that rerun makes the compile-button condition False again, the whole
    # block (including the freshly generated bytes) disappears, and the
    # in-flight download gets served stale/empty content. Persisting the
    # generated PDF in session_state and rendering the download button
    # unconditionally (outside the compile button's if-block) avoids that.
    pdf_state_key = f"rpt_pdf_bytes_{page_name}"
    pdf_name_key = f"rpt_pdf_name_{page_name}"

    if st.button("📄 Compile PDF", type="primary", use_container_width=True):
        with st.spinner("Generating custom report..."):
            st.session_state[pdf_state_key] = build_custom_pdf(page_name, df, selected_charts, active_filters)
            st.session_state[pdf_name_key] = f"HealthSentinel_{page_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.success("Report successfully generated!")

    if st.session_state.get(pdf_state_key):
        st.download_button(
            label="⬇️ Download PDF Report",
            data=st.session_state[pdf_state_key],
            file_name=st.session_state.get(pdf_name_key, "HealthSentinel_Report.pdf"),
            mime="application/pdf",
            use_container_width=True,
            key=f"rpt_download_btn_{page_name}",
        )

# Inject button floating style
def render_floating_button_css():
    st.markdown(
        """
        <style>
        div.floating-report-container {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 999999;
        }
        div.floating-report-container button {
            background-color: #0F6B78 !important;
            color: white !important;
            border-radius: 50% !important;
            width: 56px !important;
            height: 56px !important;
            box-shadow: 0 4px 12px rgba(23, 50, 77, 0.3) !important;
            border: 2px solid white !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            font-size: 22px !important;
            transition: transform 0.2s ease, background-color 0.2s ease !important;
        }
        div.floating-report-container button:hover {
            transform: scale(1.1);
            background-color: #17324D !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def _clean_filter_dict(raw: dict) -> dict:
    """Normalize a page's assorted filter values into a flat label -> text
    mapping for display in the PDF's "Applied Dashboard Filters" block.
    Empty selections, empty lists, and "All"-style defaults are left as-is;
    multi-select lists are joined into a readable comma-separated string."""
    clean = {}
    for label, value in (raw or {}).items():
        if value is None or value == "" or value == []:
            clean[label] = "All"
        elif isinstance(value, (list, tuple, set)):
            clean[label] = ", ".join(str(v) for v in value) if value else "All"
        else:
            clean[label] = str(value)
    return clean


# Render sidebar entry point (left-nav "Generate Report" button)
def add_sidebar_report_button(page_name, df, active_filters):
    """Places a 'Generate Report' button in the left-hand navigation
    sidebar, directly under the page list, for the given dashboard. Clicking
    it opens the same report-selection dialog used elsewhere in the app;
    the report content (charts + the "Applied Dashboard Filters" summary)
    reflects whatever filters are currently active on the page, since the
    caller passes in the already-filtered dataframe and the current filter
    selections.
    """
    clean_filters = _clean_filter_dict(active_filters)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📄 Reports")
        st.caption(f"Export a PDF report for **{page_name}**, scoped to your current filters.")
        if st.button(
            "Generate Report",
            key=f"sidebar_report_btn_{page_name.replace(' ', '_')}",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.open_report_dialog = page_name
            st.rerun()

    if st.session_state.get("open_report_dialog") == page_name:
        show_report_generator_dialog(page_name, df, clean_filters)


# Render floating action button
def add_report_button(page_name, df, active_filters):
    render_floating_button_css()
    
    with st.container():
        st.markdown('<div class="floating-report-container">', unsafe_allow_html=True)
        if st.button("📝", key=f"floating_report_btn_{page_name.replace(' ', '_')}"):
            st.session_state.open_report_dialog = page_name
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    if st.session_state.get("open_report_dialog") == page_name:
        show_report_generator_dialog(page_name, df, active_filters)
