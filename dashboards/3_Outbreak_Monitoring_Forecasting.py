"""
3_Outbreak_Monitoring_Forecasting.py
---------------------------------------
Outbreak Monitoring & Forecasting dashboard — alert levels, containment
performance, ARIMA case forecasting, and priority containment scoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from src.styling import (
    inject_css, page_header, kpi_card, kpi_card_delta, snapshot_row, section_title,
    filter_bar_header, insight_banner, HOTSPOT_SCALE, PIE_SEVERITY,
)
from src.report_generator import add_sidebar_report_button

# ==========================================
# 1. THEME AND LAYOUT CONFIGURATION
# ==========================================
inject_css()

PRIMARY_NAVY = "#17324D"
TEAL = "#0F6B78"
SLATE_BLUE = PRIMARY_NAVY
GREEN = "#16855B"
AMBER = "#C98A00"
RED = "#C43D3D"

BACKGROUND = "#F5F7FA"
WHITE = "#FFFFFF"
TEXT = "#000000"
SECONDARY_TEXT = "#000000"
BORDER = "rgba(100, 116, 139, 0.22)"

# Check if running within the multipage dashboard suite
try:
    from src.data_loader import get_outbreak_master
    RUNNING_IN_SUITE = True
except ImportError:
    RUNNING_IN_SUITE = False

# Inject Reference CSS Styles
st.markdown(f"""
<style>
    /* Main Page Background */
    .stApp {{
        background-color: {BACKGROUND};
    }}

    /* Header buttons & Clean look */
    header[data-testid="stHeader"] {{
        background-color: {BACKGROUND} !important;
        border-bottom: none !important;
    }}
    header[data-testid="stHeader"] button {{
        color: {PRIMARY_NAVY} !important;
        background-color: transparent !important;
        border: none !important;
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1500px;
    }}

    /* Custom Titles */
    .main-title {{
        font-size: 32px;
        font-weight: 700;
        color: {PRIMARY_NAVY};
        margin-bottom: 5px;
    }}

    .main-subtitle {{
        font-size: 14px;
        color: {SECONDARY_TEXT};
        margin-bottom: 20px;
    }}

    /* Headers */
    h2, h3 {{
        color: {PRIMARY_NAVY} !important;
        font-weight: 650 !important;
    }}

    /* Sidebar Overrides */
    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY_NAVY};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: #E8EDF1 !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: #FFFFFF !important;
        font-size: 1.15rem;
    }}
    section[data-testid="stSidebar"] label {{
        color: #E8EDF1 !important;
        font-weight: 500;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: rgba(255,255,255,0.08);
        border-color: rgba(255,255,255,0.25);
    }}
    section[data-testid="stSidebar"] button {{
        background-color: {TEAL} !important;
        color: white !important;
        border: none !important;
    }}


    /* Card Panels for Plotly */
    .plotly-card {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(23, 50, 77, 0.06);
        margin-bottom: 20px;
    }}
    
    .section-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {TEXT};
        margin: 6px 0 4px 0;
        border-left: 4px solid {TEAL};
        padding-left: 10px;
    }}
    
    .section-caption {{
        font-size: 0.78rem;
        color: {SECONDARY_TEXT};
        padding-left: 14px;
        margin-bottom: 14px;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
@st.cache_data
def load_and_prepare_data():
    if RUNNING_IN_SUITE:
        df = get_outbreak_master()
    else:
        # Fallback to local files in workspace
        try:
            df = pd.read_csv("data/fact_outbreak.csv")
            dates = pd.read_csv("data/dim_date.csv")
            states = pd.read_csv("data/dim_state.csv")
            diseases = pd.read_csv("data/dim_disease.csv")
            sources = pd.read_csv("data/dim_source.csv")
            
            df = (
                df.merge(dates, on="date_id", how="left")
                .merge(states, on="state_id", how="left")
                .merge(diseases, on="disease_id", how="left")
                .merge(sources, on="source_id", how="left")
            )
        except Exception:
            # Try alternative cleaned path
            df = pd.read_csv("cleaning/fact_outbreak_clean.csv")
            dates = pd.read_csv("data/dim_date.csv")
            states = pd.read_csv("data/dim_state.csv")
            diseases = pd.read_csv("data/dim_disease.csv")
            df = (
                df.merge(dates, on="date_id", how="left")
                .merge(states, on="state_id", how="left")
                .merge(diseases, on="disease_id", how="left")
            )
            
    df["full_date"] = pd.to_datetime(df["full_date"], errors="coerce")
    df["year"] = df["year"].astype("Int64")
    df["year_month"] = df["year_month"].astype(str)
    
    # Strip spaces
    if "region" in df.columns:
        df["region"] = df["region"].astype(str).str.strip()
        
    # NOTE: these columns hold "yes"/"no" strings in the source data. A
    # plain .astype(bool) treats any non-empty string (including "no") as
    # True, which silently made every row "controlled" / "emergency" alike.
    # Map the actual yes/no values instead so downstream counts (KPIs, the
    # lifecycle funnel, and the alert queue below) are correct.
    for c in ["new_outbreak_flag", "controlled_flag", "emergency_alert_flag"]:
        if c in df.columns:
            if df[c].dtype == bool:
                continue
            df[c] = df[c].astype(str).str.strip().str.lower().map(
                {"yes": True, "no": False, "true": True, "false": False}
            ).fillna(False)
            
    return df

df = load_and_prepare_data()

# Page Title (shared header box — matches the Executive Public Health Overview)
n_states = df["state_name"].nunique()
date_min = df["year_month"].min()
date_max = df["year_month"].max()
page_header(
    "Outbreak Monitoring & Forecasting",
    f"Covering {n_states} states · {date_min} to {date_max}",
)

# ==========================================
# 2B. ALERT RESPONSE QUEUE — PERSISTENT STATE
# ==========================================
# A small operational workflow layered on top of the analytics above: the
# highest-severity, still-uncontrolled outbreak records become actionable
# "tickets" that a response coordinator can Acknowledge -> Investigate ->
# Resolve. Status is persisted to disk (like Ward Capacity / Outpatient
# Queue in the reference app) so it survives page reloads and isn't reset
# by the sidebar filters above.
ALERT_QUEUE_DB = "data/outbreak_alert_queue_state.json"
ALERT_QUEUE_SIZE = 30  # cap the queue to the most urgent records


def _load_alert_queue_state():
    if os.path.exists(ALERT_QUEUE_DB):
        try:
            with open(ALERT_QUEUE_DB, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_alert_queue_state():
    os.makedirs("data", exist_ok=True)
    with open(ALERT_QUEUE_DB, "w") as f:
        json.dump(st.session_state.alert_queue_status, f)


def _seed_alert_queue_candidates(full_df):
    """Pick the most urgent still-uncontrolled High-alert emergency outbreaks
    to seed the queue with, ranked by response delay (worst first)."""
    candidates = full_df[
        (full_df["alert_level"] == "High")
        & (~full_df["controlled_flag"].astype(bool))
        & (full_df["emergency_alert_flag"].astype(bool))
    ].sort_values("response_time_hours", ascending=False).head(ALERT_QUEUE_SIZE)
    return candidates


if "alert_queue_status" not in st.session_state:
    st.session_state.alert_queue_status = _load_alert_queue_state()

_alert_candidates = _seed_alert_queue_candidates(df)
for _oid in _alert_candidates["outbreak_id"].astype(str):
    if _oid not in st.session_state.alert_queue_status:
        st.session_state.alert_queue_status[_oid] = {
            "status": "New",
            "updated_at": None,
        }

ALERT_STATUS_FLOW = {"New": "Acknowledged", "Acknowledged": "Investigating", "Investigating": "Resolved"}
ALERT_STATUS_ACTION_LABEL = {
    "New": "✅ Acknowledge",
    "Acknowledged": "🔎 Start Investigation",
    "Investigating": "☑ Mark Resolved",
}
ALERT_STATUS_COLOR = {"New": "#C43D3D", "Acknowledged": "#C98A00", "Investigating": "#0F6B78", "Resolved": "#16855B"}

# ==========================================
# 3. FILTER STATE
# ==========================================
# The filter widgets are displayed under the KPI cards. session_state keeps
# their current selections available for the filtering/KPI calculations first.
st.session_state.setdefault("state_filter", "All")
st.session_state.setdefault("region_filter", "All")
st.session_state.setdefault("year_filter", "All")
st.session_state.setdefault("disease_filter", "All")
st.session_state.setdefault("alert_filter", "All")

def reset_filters():
    st.session_state["state_filter"] = "All"
    st.session_state["region_filter"] = "All"
    st.session_state["year_filter"] = "All"
    st.session_state["disease_filter"] = "All"
    st.session_state["alert_filter"] = "All"

region_options = ["All"] + sorted(df["region"].dropna().unique().tolist())
selected_region = st.session_state["region_filter"] if st.session_state["region_filter"] in region_options else "All"
state_pool = df if selected_region == "All" else df[df["region"] == selected_region]
state_options = ["All"] + sorted(state_pool["state_name"].dropna().unique().tolist())
selected_state = st.session_state["state_filter"] if st.session_state["state_filter"] in state_options else "All"
year_options = ["All"] + sorted([int(y) for y in df["year"].dropna().unique()])
selected_year = st.session_state["year_filter"] if st.session_state["year_filter"] in year_options else "All"
disease_options = ["All"] + sorted(df["disease_name"].dropna().unique().tolist())
selected_disease = st.session_state["disease_filter"] if st.session_state["disease_filter"] in disease_options else "All"
alert_options = ["All"] + sorted(df["alert_level"].dropna().unique().tolist())
selected_alert = st.session_state["alert_filter"] if st.session_state["alert_filter"] in alert_options else "All"

st.session_state["region_filter"] = selected_region
st.session_state["state_filter"] = selected_state
st.session_state["year_filter"] = selected_year
st.session_state["disease_filter"] = selected_disease
st.session_state["alert_filter"] = selected_alert

# Apply filters
filtered_df = df.copy()
if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]
if selected_state != "All":
    filtered_df = filtered_df[filtered_df["state_name"] == selected_state]
if selected_year != "All":
    filtered_df = filtered_df[filtered_df["year"] == int(selected_year)]
if selected_disease != "All":
    filtered_df = filtered_df[filtered_df["disease_name"] == selected_disease]
if selected_alert != "All":
    filtered_df = filtered_df[filtered_df["alert_level"] == selected_alert]

if filtered_df.empty:
    st.warning("⚠️ **No data available for the selected filter combination.** Please adjust your selections.")
    st.stop()

add_sidebar_report_button(
    "Outbreak Monitoring & Forecasting",
    filtered_df,
    {
        "Region": selected_region,
        "State": selected_state,
        "Year": selected_year,
        "Disease": selected_disease,
        "Alert Level": selected_alert,
    },
)

# ==========================================
# 4. KPI CALCULATIONS & MOM DELTAS
# ==========================================
ALERT_ORDER = {"High": 3, "Moderate": 2, "Low": 1}

def new_outbreak_count(data) -> int:
    if data.empty:
        return 0
    latest = data["year_month"].max()
    return int((data["year_month"] == latest).sum())

def calculate_kpi_deltas(filtered_data, full_data):
    unique_ym = sorted(filtered_data["year_month"].dropna().unique())
    if not unique_ym:
        return {}
    latest_ym = unique_ym[-1]
    
    all_ym = sorted(full_data["year_month"].dropna().unique())
    idx = all_ym.index(latest_ym) if latest_ym in all_ym else -1
    prev_ym = all_ym[idx - 1] if idx > 0 else None
    
    cur_df = filtered_data[filtered_data["year_month"] == latest_ym]
    
    # Apply identical non-temporal filters to full_data to isolate previous month's baseline
    ref_df = full_data.copy()
    if selected_region != "All":
        ref_df = ref_df[ref_df["region"] == selected_region]
    if selected_state != "All":
        ref_df = ref_df[ref_df["state_name"] == selected_state]
    if selected_disease != "All":
        ref_df = ref_df[ref_df["disease_name"] == selected_disease]
    if selected_alert != "All":
        ref_df = ref_df[ref_df["alert_level"] == selected_alert]
        
    prev_df = ref_df[ref_df["year_month"] == prev_ym] if prev_ym else pd.DataFrame()
    
    deltas = {}
    
    # 1. Total Outbreaks (Count in month)
    cur_outbreaks = len(cur_df)
    prev_outbreaks = len(prev_df)
    if prev_outbreaks > 0:
        deltas["total_outbreaks"] = f"{((cur_outbreaks - prev_outbreaks) / prev_outbreaks) * 100:+.1f}% MoM"
    else:
        deltas["total_outbreaks"] = None
        
    # 2. Emergency Alerts (Count in month)
    cur_emerg = int(cur_df["emergency_alert_flag"].sum())
    prev_emerg = int(prev_df["emergency_alert_flag"].sum()) if not prev_df.empty else 0
    if prev_emerg > 0:
        deltas["emergency_alerts"] = f"{((cur_emerg - prev_emerg) / prev_emerg) * 100:+.1f}% MoM"
    else:
        deltas["emergency_alerts"] = None
        
    # 3. Controlled Outbreaks (Count in month)
    cur_ctrl = int(cur_df["controlled_flag"].sum())
    prev_ctrl = int(prev_df["controlled_flag"].sum()) if not prev_df.empty else 0
    if prev_ctrl > 0:
        deltas["controlled_outbreaks"] = f"{((cur_ctrl - prev_ctrl) / prev_ctrl) * 100:+.1f}% MoM"
    else:
        deltas["controlled_outbreaks"] = None

    # 4. Containment Rate (Average)
    cur_contain = cur_df["containment_rate_pct"].mean()
    prev_contain = prev_df["containment_rate_pct"].mean() if not prev_df.empty else np.nan
    if pd.notna(prev_contain):
        deltas["containment_rate"] = f"{(cur_contain - prev_contain):+.2f}% MoM"
    else:
        deltas["containment_rate"] = None
        
    # 5. Response Time (Average)
    cur_resp = cur_df["response_time_hours"].mean()
    prev_resp = prev_df["response_time_hours"].mean() if not prev_df.empty else np.nan
    if pd.notna(prev_resp):
        deltas["response_time"] = f"{(cur_resp - prev_resp):+.2f} hrs MoM"
    else:
        deltas["response_time"] = None
        
    # 6. Readiness (Average)
    cur_ready = cur_df["hospital_readiness_score"].mean()
    prev_ready = prev_df["hospital_readiness_score"].mean() if not prev_df.empty else np.nan
    if pd.notna(prev_ready):
        deltas["hospital_readiness"] = f"{(cur_ready - prev_ready):+.2f}% MoM"
    else:
        deltas["hospital_readiness"] = None
        
    return deltas

deltas = calculate_kpi_deltas(filtered_df, df)

# Insight Banner
latest_month = filtered_df["year_month"].max()
emergency_alerts_count = int(filtered_df["emergency_alert_flag"].sum())
active_containment_avg = filtered_df["containment_rate_pct"].mean()
avg_response_time = filtered_df["response_time_hours"].mean()

insight_bits = [
    f"Active outbreaks containment rate is averaging **{active_containment_avg:.1f}%**",
    f"Emergency alerts standing at **{emergency_alerts_count}** warnings",
    f"Average containment response time is **{avg_response_time:.1f} hours**"
]
insight_text = f"{'; '.join(insight_bits)} (Filter Scope: {selected_state if selected_state != 'All' else selected_region if selected_region != 'All' else 'National'})."

insight_banner(insight_text)

# ==========================================
# 5. KPI METRICS DISPLAY
# ==========================================

# Row 1: Counts
row1 = st.columns(4)
with row1[0]:
    kpi_card_delta("Total Outbreaks", f"{len(filtered_df):,}", delta=deltas.get("total_outbreaks"), color=PRIMARY_NAVY, bg="#D6EAF8")
with row1[1]:
    kpi_card("New Outbreaks", f"{new_outbreak_count(filtered_df):,}", color=PRIMARY_NAVY, bg="#EBF5FB")
with row1[2]:
    kpi_card_delta("Emergency Alerts", f"{int(filtered_df['emergency_alert_flag'].sum()):,}", delta=deltas.get("emergency_alerts"), invert=True, color=RED, bg="#FADBD8")
with row1[3]:
    kpi_card_delta("Controlled Outbreaks", f"{int(filtered_df['controlled_flag'].sum()):,}", delta=deltas.get("controlled_outbreaks"), color=GREEN, bg="#D5F5E3")

# Row 2: Performance & Quality Metrics
row2 = st.columns(5)
with row2[0]:
    kpi_card_delta("Containment Rate", f"{filtered_df['containment_rate_pct'].mean():.2f}%", delta=deltas.get("containment_rate"), color=TEAL, bg="#E8F8F5")
with row2[1]:
    kpi_card_delta("Avg Response Time", f"{filtered_df['response_time_hours'].mean():.2f} hrs", delta=deltas.get("response_time"), invert=True, color=AMBER, bg="#FDEBD0")
with row2[2]:
    kpi_card_delta("Hospital Readiness", f"{filtered_df['hospital_readiness_score'].mean():.2f}%", delta=deltas.get("hospital_readiness"), color=PRIMARY_NAVY, bg="#D6EAF8")
with row2[3]:
    kpi_card("Resource Readiness", f"{filtered_df['resource_readiness_score'].mean():.2f}%", color=TEAL, bg="#EBF5FB")
with row2[4]:
    kpi_card("Forecast Accuracy", f"{filtered_df['forecast_accuracy_pct'].mean():.2f}%", color=TEAL, bg="#E8F8F5")

# Main-page filters — directly below the KPI cards.
with st.container(border=True):
    header_col, reset_col = st.columns([5, 1])
    with header_col:
        filter_bar_header("Selections are applied to every visual on this page")
    with reset_col:
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        st.button("♻️ Reset", on_click=reset_filters, use_container_width=True, key="outbreak_reset_filters")

    oc1, oc2, oc3, oc4, oc5 = st.columns(5, gap="medium")
    with oc1:
        st.selectbox("Region", region_options, key="region_filter")
    with oc2:
        st.selectbox("State", state_options, key="state_filter")
    with oc3:
        st.selectbox("Year", year_options, key="year_filter")
    with oc4:
        st.selectbox("Disease Name", disease_options, key="disease_filter")
    with oc5:
        st.selectbox("Alert Level", alert_options, key="alert_filter")

    active_outbreak_filters = []
    for label, value in [("Region", selected_region), ("State", selected_state), ("Year", selected_year), ("Disease", selected_disease), ("Alert", selected_alert)]:
        if value != "All":
            active_outbreak_filters.append(f'<span class="active-filter-chip">{label}: {value}</span>')
    if not active_outbreak_filters:
        active_outbreak_filters.append('<span class="active-filter-chip">All available data</span>')
    st.markdown(
        '<div class="active-filter-summary"><span class="summary-label">Active filters:</span>' + "".join(active_outbreak_filters) + '</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# ==========================================
# 6. IN-MEMORY ARIMA FORECAST GENERATOR
# ==========================================
@st.cache_data
def generate_dynamic_forecast(data):
    if data.empty:
        return pd.DataFrame(columns=["Month", "Actual_Cases"]), pd.DataFrame(columns=["Month", "Forecast_Cases"])

    # Ensure full_date is parsed
    ts_data = data.copy()
    ts_data = ts_data.dropna(subset=["year_month", "historical_cases"])
    
    monthly = (
        ts_data.groupby("year_month")["historical_cases"]
        .sum()
        .reset_index()
    )
    # Parse month dates
    monthly["Month"] = pd.to_datetime(monthly["year_month"] + "-01", format="%Y-%m-%d", errors="coerce")
    monthly = monthly.dropna(subset=["Month"]).sort_values("Month")
    monthly = monthly[["Month", "historical_cases"]].copy()
    monthly.rename(columns={"historical_cases": "Actual_Cases"}, inplace=True)
    
    if len(monthly) < 4:
        # Fallback for sparse history
        last_val = monthly["Actual_Cases"].iloc[-1] if not monthly.empty else 0
        forecast_dates = pd.date_range(
            start=(monthly["Month"].iloc[-1] if not monthly.empty else pd.Timestamp.now()) + pd.DateOffset(months=1),
            periods=6,
            freq="MS"
        )
        forecast_df = pd.DataFrame({
            "Month": forecast_dates,
            "Forecast_Cases": [int(last_val)] * 6
        })
        return monthly, forecast_df
        
    ts = monthly.set_index("Month")
    ts["log_cases"] = np.log(ts["Actual_Cases"] + 1)
    
    try:
        model = ARIMA(ts["log_cases"], order=(1, 1, 1))
        model_fit = model.fit()
        log_forecast = model_fit.forecast(steps=6)
        forecast_values = np.exp(log_forecast) - 1
        forecast_values = np.maximum(forecast_values, 0)
    except Exception:
        # Fallback to simple average baseline if ARIMA fails
        forecast_values = np.array([ts["Actual_Cases"].iloc[-3:].mean()] * 6)
        
    last_date = ts.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=6,
        freq="MS"
    )
    forecast_df = pd.DataFrame({
        "Month": forecast_dates,
        "Forecast_Cases": forecast_values
    })
    forecast_df["Forecast_Cases"] = forecast_df["Forecast_Cases"].round(0).astype(int)
    
    return monthly.reset_index(drop=True), forecast_df

# Generate dynamic ARIMA forecasts
history, forecast = generate_dynamic_forecast(filtered_df)

# ==========================================
# 7. TABS LAYOUT: MONITORING | FORECASTING | DECISION ANALYTICS
# ==========================================
tab_mon, tab_fcst, tab_dec, tab_queue = st.tabs([
    " Monitoring Summary",
    " Predictive Forecasting",
    " Decision Support Platform",
    " Alert Response Queue",
])

# Plotly Layout Helper
def apply_plotly_styling(fig, xlabel="", ylabel=""):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, sans-serif", size=11),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
            showline=False,
            title=dict(text=xlabel, font=dict(size=12, color=SECONDARY_TEXT), standoff=8) if xlabel else None
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
            showline=False,
            title=dict(text=ylabel, font=dict(size=12, color=SECONDARY_TEXT), standoff=8) if ylabel else None
        ),
        margin=dict(l=40, r=20, t=15, b=50),
        hoverlabel=dict(
            bgcolor=WHITE,
            font_size=12,
            font_family="Inter, sans-serif"
        )
    )

# ------------------------------------------
# TAB 1: MONITORING SUMMARY
# ------------------------------------------
with tab_mon:
    st.markdown('<div class="section-title">Outbreak Surveillance Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Real-time status analysis of current active outbreak events</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Outbreak trend events per month
        with st.container(border=True):
            st.markdown('<h4>Outbreak Event Trend</h4>', unsafe_allow_html=True)
            events_trend = filtered_df.groupby("year_month").size().reset_index(name="outbreaks")
            fig_trend = go.Figure(go.Scatter(
                x=events_trend["year_month"], y=events_trend["outbreaks"], mode="lines+markers",
                line=dict(color=TEAL, width=3),
                marker=dict(size=6, color=PRIMARY_NAVY),
                fill="tozeroy", fillcolor="rgba(15,107,120,0.08)"
            ))
            apply_plotly_styling(fig_trend, xlabel="Month", ylabel="Events")
            st.plotly_chart(fig_trend, use_container_width=True)
        
    with c2:
        # Alert level distribution pie chart
        with st.container(border=True):
            st.markdown('<h4>Alert Severity Profile</h4>', unsafe_allow_html=True)
            alert_counts = filtered_df["alert_level"].value_counts().reindex(["High", "Moderate", "Low"]).fillna(0)
            fig_pie = go.Figure(data=[go.Pie(
                labels=alert_counts.index, values=alert_counts.values,
                # Pastel High/Moderate/Low — same trio used by every risk
                # heatmap in the suite, so this pie agrees with them instead
                # of using the more saturated flat status colors.
                marker=dict(colors=[PIE_SEVERITY["High"], PIE_SEVERITY["Moderate"], PIE_SEVERITY["Low"]]),
                hole=0.6, textinfo="label+percent",
                textfont=dict(color="#000000", size=12),
                insidetextfont=dict(color="#000000", size=12),
                outsidetextfont=dict(color="#000000", size=12)
            )])
            fig_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=300)
            apply_plotly_styling(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True)
        
    c3, c4 = st.columns(2)
    
    with c3:
        # Outbreak funnel
        with st.container(border=True):
            st.markdown('<h4>Outbreak Management Lifecycle</h4>', unsafe_allow_html=True)
            stages = {
                "Total Outbreaks": len(filtered_df),
                "New Outbreaks": new_outbreak_count(filtered_df),
                "Emergency Alerts": int(filtered_df["emergency_alert_flag"].sum()),
                "Controlled": int(filtered_df["controlled_flag"].sum())
            }
            fig_funnel = go.Figure(go.Funnel(
                y=list(stages.keys()), x=list(stages.values()),
                textinfo="value+percent initial",
                marker=dict(color=[PRIMARY_NAVY, TEAL, AMBER, GREEN]),
                textfont=dict(color="#000000", size=11)
            ))
            apply_plotly_styling(fig_funnel)
            fig_funnel.update_layout(margin=dict(l=120, r=20, t=10, b=30))
            st.plotly_chart(fig_funnel, use_container_width=True)
        
    with c4:
        # Response time histogram
        with st.container(border=True):
            st.markdown('<h4>Response Efficiency Profile</h4>', unsafe_allow_html=True)
            fig_hist = go.Figure(data=[go.Histogram(
                x=filtered_df["response_time_hours"], nbinsx=15,
                marker=dict(color=TEAL, line=dict(color=PRIMARY_NAVY, width=0.5)),
                opacity=0.85
            )])
            apply_plotly_styling(fig_hist, xlabel="Response Time (hours)", ylabel="Events")
            st.plotly_chart(fig_hist, use_container_width=True)

    # Top Rankings Row
    st.markdown('<div class="section-title">Top Hotspots & Vector Threats</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Geographic and disease vectors driving case concentration</div>', unsafe_allow_html=True)
    
    c5, c6 = st.columns(2)
    with c5:
        with st.container(border=True):
            st.markdown('<h4>Top 10 States by Historical Cases</h4>', unsafe_allow_html=True)
            state_cases = filtered_df.groupby("state_name")["historical_cases"].sum().nlargest(10).reset_index().sort_values("historical_cases")
            fig_state_bar = go.Figure(go.Bar(
                x=state_cases["historical_cases"], y=state_cases["state_name"], orientation="h",
                marker=dict(color=state_cases["historical_cases"], colorscale=HOTSPOT_SCALE),
                text=[f"{int(v):,}" for v in state_cases["historical_cases"]], textposition="auto"
            ))
            apply_plotly_styling(fig_state_bar, xlabel="Total Cases")
            fig_state_bar.update_layout(margin=dict(l=100, r=20, t=10, b=40))
            st.plotly_chart(fig_state_bar, use_container_width=True)
        
    with c6:
        with st.container(border=True):
            st.markdown('<h4>Top 10 Diseases by Historical Cases</h4>', unsafe_allow_html=True)
            disease_cases = filtered_df.groupby("disease_name")["historical_cases"].sum().nlargest(10).reset_index().sort_values("historical_cases")
            fig_disease_bar = go.Figure(go.Bar(
                x=disease_cases["historical_cases"], y=disease_cases["disease_name"], orientation="h",
                marker=dict(color=disease_cases["historical_cases"], colorscale=HOTSPOT_SCALE),
                text=[f"{int(v):,}" for v in disease_cases["historical_cases"]], textposition="auto"
            ))
            apply_plotly_styling(fig_disease_bar, xlabel="Total Cases")
            fig_disease_bar.update_layout(margin=dict(l=120, r=20, t=10, b=40))
            st.plotly_chart(fig_disease_bar, use_container_width=True)

# ------------------------------------------
# TAB 2: PREDICTIVE FORECASTING
# ------------------------------------------
with tab_fcst:
    st.markdown('<div class="section-title">ARIMA Predictive Modeling</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">ARIMA(1,1,1) model projections for case velocities</div>', unsafe_allow_html=True)
    
    # Summary of metrics for the forecast
    f_total_cases = int(history["Actual_Cases"].sum())
    f_peak_cases = int(history["Actual_Cases"].max()) if not history.empty else 0
    f_latest_cases = int(history["Actual_Cases"].iloc[-1]) if not history.empty else 0
    f_projection = int(forecast["Forecast_Cases"].sum())
    
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Historical Cases (36 Mo)", f"{f_total_cases:,}")
    fc2.metric("Peak Month Cases", f"{f_peak_cases:,}")
    fc3.metric("Latest Month Cases", f"{f_latest_cases:,}")
    fc4.metric("6-Month Projected Cases", f"{f_projection:,}")
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns([3, 2])
    with col_f1:
        with st.container(border=True):
            st.markdown('<h4>ARIMA Trend Model (6-Mo Outlook)</h4>', unsafe_allow_html=True)
        
            fig_arima = go.Figure()
            fig_arima.add_trace(go.Scatter(
                x=history["Month"], y=history["Actual_Cases"], mode="lines+markers",
                name="Historical Cases", line=dict(color=TEAL, width=2.5),
                marker=dict(size=6, color=PRIMARY_NAVY)
            ))
            fig_arima.add_trace(go.Scatter(
                x=forecast["Month"], y=forecast["Forecast_Cases"], mode="lines+markers",
                name="Projected Cases", line=dict(color=AMBER, width=2.5, dash="dash"),
                marker=dict(size=6, symbol="diamond", color=RED)
            ))
            apply_plotly_styling(fig_arima, xlabel="Date", ylabel="Cases")
            fig_arima.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_arima, use_container_width=True)
        
    with col_f2:
        with st.container(border=True):
            st.markdown('<h4>Historical Growth Rate</h4>', unsafe_allow_html=True)
            growth = history.copy()
            growth["Growth_Rate"] = growth["Actual_Cases"].pct_change() * 100
            growth_colors = [GREEN if x <= 0 else RED for x in growth["Growth_Rate"].fillna(0)]
            fig_growth = go.Figure(go.Bar(
                x=growth["Month"], y=growth["Growth_Rate"],
                marker=dict(color=growth_colors)
            ))
            apply_plotly_styling(fig_growth, xlabel="Date", ylabel="Growth Rate %")
            st.plotly_chart(fig_growth, use_container_width=True)
        
    col_f3, col_f4 = st.columns(2)
    with col_f3:
        with st.container(border=True):
            st.markdown('<h4>Actual vs Modeled Backtest Cases</h4>', unsafe_allow_html=True)
            backtest = filtered_df.groupby("year_month")[["historical_cases", "predicted_cases"]].sum().reset_index().sort_values("year_month")
            fig_backtest = go.Figure()
            fig_backtest.add_trace(go.Scatter(
                x=backtest["year_month"], y=backtest["historical_cases"], mode="lines+markers",
                name="Actual Cases", line=dict(color=PRIMARY_NAVY, width=2)
            ))
            fig_backtest.add_trace(go.Scatter(
                x=backtest["year_month"], y=backtest["predicted_cases"], mode="lines+markers",
                name="Modeled Cases", line=dict(color=TEAL, width=2, dash="dot")
            ))
            apply_plotly_styling(fig_backtest, xlabel="Month", ylabel="Cases")
            st.plotly_chart(fig_backtest, use_container_width=True)
        
    with col_f4:
        with st.container(border=True):
            st.markdown('<h4>Backtest Model Accuracy Trend</h4>', unsafe_allow_html=True)
            acc_trend = filtered_df.groupby("year_month")["forecast_accuracy_pct"].mean().reset_index().sort_values("year_month")
            fig_acc = go.Figure(go.Scatter(
                x=acc_trend["year_month"], y=acc_trend["forecast_accuracy_pct"], mode="lines+markers",
                line=dict(color=GREEN, width=2),
                fill="tozeroy", fillcolor="rgba(22,133,91,0.08)"
            ))
            apply_plotly_styling(fig_acc, xlabel="Month", ylabel="Accuracy %")
            st.plotly_chart(fig_acc, use_container_width=True)

# ------------------------------------------
# TAB 3: DECISION SUPPORT PLATFORM
# ------------------------------------------
with tab_dec:
    st.markdown('<div class="section-title">Outbreak Priority Containment Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Outbreak Response priorities scored from 0-100 based on Case Velocity, Alert Level, Response Latency, and Facility Readiness.</div>', unsafe_allow_html=True)
    
    # Priority table calculation
    g_prio = filtered_df.groupby("state_name").agg(
        outbreaks=("outbreak_id", "count"),
        hist_cases=("historical_cases", "sum"),
        pred_cases=("predicted_cases", "sum"),
        max_alert=("alert_level", lambda s: max(s, key=lambda a: ALERT_ORDER.get(a, 0)) if len(s) > 0 else "Low"),
        response=("response_time_hours", "mean"),
        readiness=("hospital_readiness_score", "mean")
    ).reset_index()
    
    # Compute scores
    g_prio["trend_pct"] = ((g_prio["pred_cases"] - g_prio["hist_cases"]) / g_prio["hist_cases"].replace(0, np.nan) * 100).fillna(0)
    alert_w = g_prio["max_alert"].map(ALERT_ORDER).fillna(0) / 3 * 30
    resp_w = (g_prio["response"].clip(0, 48) / 48) * 20
    read_w = ((100 - g_prio["readiness"]).clip(0, 100) / 100) * 20
    trend_w = g_prio["trend_pct"].clip(0, 100) / 100 * 30
    g_prio["priority_score"] = (alert_w + resp_w + read_w + trend_w).round(1)
    
    g_prio["Priority Status"] = g_prio["priority_score"].apply(
        lambda s: "🔴 Critical" if s >= 60 else "🟠 High" if s >= 40 else "🟡 Medium" if s >= 20 else "🟢 Low"
    )
    
    g_prio = g_prio.sort_values("priority_score", ascending=False).reset_index(drop=True)
    
    # Format display columns
    display_prio = g_prio.copy()
    display_prio["trend_pct"] = display_prio["trend_pct"].map(lambda v: f"{v:+.1f}%")
    display_prio["response"] = display_prio["response"].map(lambda v: f"{v:.1f} hrs")
    display_prio["readiness"] = display_prio["readiness"].map(lambda v: f"{v:.1f}%")
    
    display_prio = display_prio.rename(columns={
        "state_name": "State Focus Area",
        "outbreaks": "Outbreaks Reported",
        "max_alert": "Peak Alert Level",
        "trend_pct": "Projected Case Growth",
        "response": "Response Latency",
        "readiness": "Hospital readiness",
        "priority_score": "Composite Priority Score",
        "Priority Status": "Priority Level"
    })
    
    st.dataframe(display_prio, use_container_width=True, hide_index=True)
    
    if not g_prio.empty:
        top = g_prio.iloc[0]
        st.markdown(
            f"""
            <div style="background-color: rgba(196, 61, 61, 0.08); border-left: 4px solid {RED}; padding: 14px 18px; border-radius: 8px; margin-top: 15px;">
                <strong style="color: {RED}; font-size: 15px;">⚠️ Top Active Priority: {top['state_name']}</strong><br/>
                <span style="font-size: 13px; color: {TEXT};">
                    Current status classified as <strong>{top['Priority Status']}</strong> (Priority Score: <strong>{top['priority_score']:.0f}/100</strong>). 
                    Reported <strong>{top['outbreaks']} outbreaks</strong> with peak alert levels of <strong>{top['max_alert']}</strong>, 
                    showing average response delay of <strong>{top['response']:.1f} hours</strong> and local hospital readiness scores at <strong>{top['readiness']:.1f}%</strong>.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------------------
# TAB 4: ALERT RESPONSE QUEUE (working feature)
# ------------------------------------------
with tab_queue:
    st.markdown('<div class="section-title">Outbreak Alert Response Queue</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">The 30 most urgent, still-uncontrolled High-alert emergencies — '
        'work each ticket through Acknowledge → Investigate → Resolve. Status is saved and persists across reloads.</div>',
        unsafe_allow_html=True,
    )

    # Build the live ticket list: candidate rows joined with their persisted status
    _queue_df = _alert_candidates.copy()
    _queue_df["outbreak_id_str"] = _queue_df["outbreak_id"].astype(str)
    _queue_df["queue_status"] = _queue_df["outbreak_id_str"].map(
        lambda oid: st.session_state.alert_queue_status.get(oid, {}).get("status", "New")
    )

    # Respect the sidebar filters already applied to the rest of the page,
    # so narrowing to a state/disease/region also narrows the queue.
    _queue_view = _queue_df.copy()
    if selected_region != "All":
        _queue_view = _queue_view[_queue_view["region"] == selected_region]
    if selected_state != "All":
        _queue_view = _queue_view[_queue_view["state_name"] == selected_state]
    if selected_disease != "All":
        _queue_view = _queue_view[_queue_view["disease_name"] == selected_disease]

    status_counts = _queue_view["queue_status"].value_counts()
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        kpi_card("New", str(int(status_counts.get("New", 0))), color=ALERT_STATUS_COLOR["New"], bg="#FADBD8")
    with qc2:
        kpi_card("Acknowledged", str(int(status_counts.get("Acknowledged", 0))), color=ALERT_STATUS_COLOR["Acknowledged"], bg="#FCF3CF")
    with qc3:
        kpi_card("Investigating", str(int(status_counts.get("Investigating", 0))), color=ALERT_STATUS_COLOR["Investigating"], bg="#D6EAF8")
    with qc4:
        kpi_card("Resolved", str(int(status_counts.get("Resolved", 0))), color=ALERT_STATUS_COLOR["Resolved"], bg="#D5F5E3")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    _active = _queue_view[_queue_view["queue_status"] != "Resolved"].sort_values("response_time_hours", ascending=False)

    if _active.empty:
        st.success("No active tickets for the current filter selection — every urgent alert has been resolved. ✅")
    else:
        for _, row in _active.iterrows():
            oid = row["outbreak_id_str"]
            status = row["queue_status"]
            badge_color = ALERT_STATUS_COLOR[status]
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(
                        f"""
                        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                            <span style="font-weight:700; color:{TEXT}; font-size:0.95rem;">{row['state_name']} · {row['disease_name']}</span>
                            <span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}55;
                                         padding:2px 10px; border-radius:12px; font-size:0.7rem; font-weight:700;">{status}</span>
                        </div>
                        <div style="color:{TEXT}; font-size:0.8rem; margin-top:6px;">
                            Response delay: <b>{row['response_time_hours']:.1f} hrs</b> &nbsp;·&nbsp;
                            Containment so far: <b>{row['containment_rate_pct']:.1f}%</b> &nbsp;·&nbsp;
                            Hospital readiness: <b>{row['hospital_readiness_score']:.1f}%</b> &nbsp;·&nbsp;
                            Ticket #{oid}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with right:
                    if st.button(ALERT_STATUS_ACTION_LABEL[status], key=f"alert_action_{oid}", use_container_width=True):
                        st.session_state.alert_queue_status[oid] = {
                            "status": ALERT_STATUS_FLOW[status],
                            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        _save_alert_queue_state()
                        st.rerun()

    _resolved = _queue_view[_queue_view["queue_status"] == "Resolved"]
    if not _resolved.empty:
        with st.expander(f"☑ Resolved tickets ({len(_resolved)})"):
            for _, row in _resolved.sort_values("response_time_hours", ascending=False).iterrows():
                oid = row["outbreak_id_str"]
                updated = st.session_state.alert_queue_status.get(oid, {}).get("updated_at", "—")
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; padding:8px 4px; border-bottom:1px solid {BORDER};">
                        <span style="color:{TEXT}; font-size:0.85rem;">{row['state_name']} · {row['disease_name']} (#{oid})</span>
                        <span style="color:{SECONDARY_TEXT}; font-size:0.78rem;">Resolved {updated}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# EXPANDER DETAILS (reference.py model details)
st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
with st.expander(" Dynamic Model Specification & Operations"):
    st.markdown(f"""
    #### Dynamic ARIMA Forecasting Model Specifications
    - **Configuration:** ARIMA(1,1,1) (1st-Order Autoregression, 1st-Difference Integrator, 1st-Order Moving Average)
    - **Input Data Scope:** Historical aggregated case volumes per month across all active filters.
    - **In-Memory Pipeline:** Filtered dataframe is grouped by `year_month` dynamically, normalized using log-transform `log(x + 1)` for variance stabilization, processed by statsmodels, and back-converted via exponential transformation.
    - **Model Fallback:** Reverts to a trailing average baseline when filtered records cover less than 4 active data months.
    """)

# ------------------------------------------------------------------------- #
# Decision Snapshot
# ------------------------------------------------------------------------- #
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
_by_state = filtered_df.groupby("state_name", as_index=False).agg(
    containment_rate_pct=("containment_rate_pct", "mean"),
    response_time_hours=("response_time_hours", "mean"),
)
_high_alert_states = filtered_df[filtered_df["alert_level"] == "High"]["state_name"]
if not _high_alert_states.empty:
    _top_high_alert = _high_alert_states.value_counts().idxmax()
    _top_high_alert_count = _high_alert_states.value_counts().max()
else:
    _top_high_alert, _top_high_alert_count = "None", 0
_low_containment = _by_state.sort_values("containment_rate_pct", ascending=True).iloc[0]
_slow_response = _by_state.sort_values("response_time_hours", ascending=False).iloc[0]

snapshot_row([
    ("Most 'High' alerts", f'{_top_high_alert} • {_top_high_alert_count}'),
    ("Lowest containment rate", f'{_low_containment["state_name"]} • {_low_containment["containment_rate_pct"]:.1f}%'),
    ("Slowest response time", f'{_slow_response["state_name"]} • {_slow_response["response_time_hours"]:.1f} hrs'),
])

# Footer Timestamp
st.markdown(
    f"""
    <div style="text-align: center; font-size: 11px; color: {SECONDARY_TEXT}; margin-top: 30px;">
        🩺 <strong>HealthSentinel Command Center</strong> |
        Last Model Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
        Multipage Dashboard Suite Integration v1.1
    </div>
    """,
    unsafe_allow_html=True
)
