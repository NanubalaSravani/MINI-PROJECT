"""
5_Upload_Custom_Analysis.py
----------------------------
Three workspaces:
  - Analyze Custom Dataset: profiles any uploaded CSV/Excel file in-memory
    and can generate a downloadable PDF analysis report. Open access.
  - Admin Data Manager Portal: allows uploading, validating, and saving/appending
    datasets directly to the data/ folder to update the entire dashboard suite.
    Open access, no login required.
  - Human Review: admin-only workspace, gated behind a login (checked against
    data/admin_credentials.csv). Each uploaded PDF report here moves through
    Pending Review -> Reviewed -> Approved, with a required reviewer comment
    at each step and a logged audit trail.
"""

import base64
import hashlib
import io
import json
import os
import sqlite3
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.styling import inject_css, page_header, kpi_card, section_title, ACCENT, SUCCESS, WARNING, DANGER, TEXT, MUTED
from src.pdf_report import build_pdf_report

# --- Admin credentials -------------------------------------------------
# Passwords are stored as SHA-256(salt + password) hashes in
# data/admin_credentials.csv (columns: admin_id, full_name, username,
# password_hash, salt, department). The file is excluded from Git via
# .gitignore — never commit credentials to version control.
# Real deployments should use SSO / OAuth / Streamlit Secrets.
ADMIN_CREDENTIALS_PATH = "data/admin_credentials.csv"


@st.cache_data(show_spinner=False, ttl=300)  # re-read at most every 5 min
def _load_admin_credentials():
    if os.path.exists(ADMIN_CREDENTIALS_PATH):
        return pd.read_csv(ADMIN_CREDENTIALS_PATH)
    return pd.DataFrame(columns=["admin_id", "full_name", "username", "password_hash", "salt", "department"])


def _authenticate_admin(username: str, password: str):
    """Returns the matching admin row (as a dict) if credentials are valid, else None.
    Compares the supplied password against SHA-256(salt + password) stored in the CSV.
    """
    import hashlib
    admins = _load_admin_credentials()
    user_rows = admins[admins["username"] == username]
    if user_rows.empty:
        return None
    row = user_rows.iloc[0]
    # Support both hashed (new) and plaintext (legacy) credential files.
    if "password_hash" in admins.columns and "salt" in admins.columns:
        salt = str(row["salt"])
        expected = hashlib.sha256((salt + password).encode()).hexdigest()
        if expected == str(row["password_hash"]):
            return row.to_dict()
    elif "password" in admins.columns:
        # Legacy plaintext fallback (not recommended for production)
        if str(row["password"]) == password:
            return row.to_dict()
    return None


def render_admin_login_form(form_key: str):
    """Shared login form used by both the Admin Data Manager Portal and
    Human Review tabs. Returns True once the admin is logged in."""
    if st.session_state.get("admin_logged_in"):
        login_status_col, logout_col = st.columns([4, 1])
        with login_status_col:
            st.success(
                f"✅ Logged in as **{st.session_state.get('admin_full_name', 'Admin')}** "
                f"({st.session_state.get('admin_department', '')})"
            )
        with logout_col:
            if st.button("Log Out", use_container_width=True, key=f"{form_key}_logout"):
                st.session_state["admin_logged_in"] = False
                st.rerun()
        return True

    st.info("🔒 Admin login is required to access this section.")
    with st.form(form_key, clear_on_submit=False):
        login_col1, login_col2 = st.columns(2)
        with login_col1:
            login_user = st.text_input("Username")
        with login_col2:
            login_pass = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button("🔑 Log In", type="primary")

    if login_submitted:
        admin_row = _authenticate_admin(login_user.strip(), login_pass)
        if admin_row:
            st.session_state["admin_logged_in"] = True
            st.session_state["admin_username"] = admin_row["username"]
            st.session_state["admin_full_name"] = admin_row["full_name"]
            st.session_state["admin_department"] = admin_row["department"]
            st.rerun()
        else:
            st.error("Invalid username or password.")
    return False


# --- Human Review workflow (for the generated PDF analysis report) -----
# All review activity (who reviewed what, which comment, which status
# change) is persisted in a small SQLite database so the audit trail
# survives across sessions/logins, not just in-memory state.
UPLOAD_REVIEW_LOG_PATH_LEGACY_JSON = "data/upload_review_log.json"  # old format, migrated once below
REVIEW_DB_PATH = "data/review_history.db"
REVIEW_STAGES = ["Pending Review", "Reviewed", "Approved"]
REVIEW_STAGE_COLOR = {"Pending Review": "#C43D3D", "Reviewed": "#C98A00", "Approved": "#16855B"}
REVIEW_STAGE_NEXT_LABEL = {"Pending Review": "🔎 Mark as Reviewed", "Reviewed": "✅ Approve"}


def _ensure_review_db():
    """Create the review_log table if it doesn't already exist."""
    os.makedirs(os.path.dirname(REVIEW_DB_PATH), exist_ok=True)
    with sqlite3.connect(REVIEW_DB_PATH, check_same_thread=False) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                admin_username TEXT,
                admin_full_name TEXT,
                file_name TEXT NOT NULL,
                file_hash TEXT,
                status TEXT NOT NULL,
                comment TEXT
            )
            """
        )
        conn.commit()


def _get_review_db_connection():
    """Return an open SQLite connection (caller is responsible for closing it)."""
    _ensure_review_db()
    return sqlite3.connect(REVIEW_DB_PATH, check_same_thread=False)


def _migrate_legacy_json_log_once():
    """One-time migration: if an old JSON review log exists and the new
    database is still empty, import its entries so history isn't lost."""
    if not os.path.exists(UPLOAD_REVIEW_LOG_PATH_LEGACY_JSON):
        return
    try:
        with _get_review_db_connection() as conn:
            existing_count = conn.execute("SELECT COUNT(*) FROM review_log").fetchone()[0]
            if existing_count == 0:
                with open(UPLOAD_REVIEW_LOG_PATH_LEGACY_JSON, "r") as f:
                    legacy_entries = json.load(f)
                for entry in legacy_entries:
                    conn.execute(
                        """INSERT INTO review_log
                           (timestamp, admin_username, admin_full_name, file_name, file_hash, status, comment)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            entry.get("timestamp", ""),
                            entry.get("admin_username", entry.get("admin", "")),
                            entry.get("admin", ""),
                            entry.get("file_name", ""),
                            entry.get("file_hash", ""),
                            entry.get("status", ""),
                            entry.get("comment", ""),
                        ),
                    )
                conn.commit()
    except Exception as exc:
        import sys
        print(f"[HealthSentinel] Warning: legacy review log migration failed: {exc}", file=sys.stderr)


def _append_upload_review_log(entry: dict):
    with _get_review_db_connection() as conn:
        conn.execute(
            """INSERT INTO review_log
               (timestamp, admin_username, admin_full_name, file_name, file_hash, status, comment)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.get("timestamp", ""),
                entry.get("admin_username", ""),
                entry.get("admin", ""),
                entry.get("file_name", ""),
                entry.get("file_hash", ""),
                entry.get("status", ""),
                entry.get("comment", ""),
            ),
        )
        conn.commit()


def _load_upload_review_log(file_name: str = None, file_hash: str = None):
    """Returns review log rows (as dicts), optionally filtered to a
    specific file (by name, or name+hash to disambiguate re-uploads that
    share a filename but have different content)."""
    with _get_review_db_connection() as conn:
        if file_name and file_hash:
            cur = conn.execute(
                "SELECT * FROM review_log WHERE file_name = ? AND file_hash = ? ORDER BY id ASC",
                (file_name, file_hash),
            )
        elif file_name:
            cur = conn.execute(
                "SELECT * FROM review_log WHERE file_name = ? ORDER BY id ASC", (file_name,)
            )
        else:
            cur = conn.execute("SELECT * FROM review_log ORDER BY id ASC")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows


def _latest_status_for_file(file_name: str, file_hash: str) -> str:
    history = _load_upload_review_log(file_name, file_hash)
    return history[-1]["status"] if history else "Pending Review"


def _render_pdf_inline(pdf_bytes: bytes, height: int = 650):
    """Renders a PDF inline in the page (no download required) via an
    embedded base64 iframe."""
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(
        f"""
        <iframe
            src="data:application/pdf;base64,{b64_pdf}"
            width="100%" height="{height}"
            style="border:1px solid #DDE3EA; border-radius:8px;"
            type="application/pdf">
        </iframe>
        """,
        unsafe_allow_html=True,
    )


_migrate_legacy_json_log_once()

# Schema specs for existing database tables
EXPECTED_COLUMNS = {
    "fact_disease_surveillance_cleaned.csv": ['fact_id', 'date_id', 'state_id', 'disease_id', 'source_id', 'population_under_surveillance', 'total_reported_cases', 'active_cases', 'recovered_cases', 'deaths', 'urban_cases', 'rural_cases', 'hospitalized_cases', 'icu_admissions', 'case_fatality_rate', 'recovery_rate', 'public_health_risk_score', 'report_date_raw'],
    "fact_environmental_cleaned.csv": ['fact_id', 'date_id', 'state_id', 'aqi', 'rainfall_mm', 'temperature_c', 'water_quality_index', 'sanitation_coverage_pct', 'healthcare_accessibility_score', 'mosquito_breeding_index', 'zoonotic_disease_incidence', 'geographic_risk_score', 'environmental_risk_score', 'case_rate_per_100k', 'hotspot_flag'],
    "fact_health_programs_cleaned.csv": ['fact_id', 'date_id', 'state_id', 'program_id', 'program_coverage_pct', 'people_screened', 'beneficiaries_reached', 'maternal_health_beneficiaries', 'child_immunization_count', 'children_population', 'adults_population', 'elderly_population', 'high_risk_individuals', 'chronic_disease_patients', 'health_vulnerability_index', 'socioeconomic_score'],
    "fact_lab_healthcare_cleaned.csv": ['fact_id', 'date_id', 'state_id', 'total_tests', 'positive_tests', 'positivity_rate', 'vaccination_coverage_pct', 'booster_coverage_pct', 'hospital_beds', 'doctors', 'phc_count', 'chc_count', 'icu_utilization_pct', 'bed_occupancy_pct', 'reporting_compliance_pct', 'turnaround_time_days', 'reporting_rate_pct'],
    "fact_outbreak_cleaned.csv": ['outbreak_id', 'date_id', 'state_id', 'disease_id', 'source_id', 'new_outbreak_flag', 'controlled_flag', 'containment_rate_pct', 'response_time_hours', 'alert_level', 'predicted_cases', 'historical_cases', 'forecast_accuracy_pct', 'hospital_readiness_score', 'resource_readiness_score', 'emergency_alert_flag', 'state_name_raw']
}

TARGET_FILES = {
    "Disease Surveillance (fact_disease_surveillance_cleaned.csv)": "fact_disease_surveillance_cleaned.csv",
    "Environmental Intelligence (fact_environmental_cleaned.csv)": "fact_environmental_cleaned.csv",
    "Laboratory & Healthcare (fact_lab_healthcare_cleaned.csv)": "fact_lab_healthcare_cleaned.csv",
    "Outbreak Monitoring (fact_outbreak_cleaned.csv)": "fact_outbreak_cleaned.csv",
    "Health Programs (fact_health_programs_cleaned.csv)": "fact_health_programs_cleaned.csv",
    "Upload New / Custom Dataset": "custom"
}

inject_css()

page_header(
    "Upload & Custom Analysis",
    "Bring your own dataset — get live KPIs, charts, and a downloadable PDF report",
    badge="LIVE",
)

TREND_PALETTE = ["#17324D", "#0F6B78", "#C43D3D", "#C98A00", "#3B8E93", "#8E44AD"]

def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

@st.cache_data(show_spinner=False)
def _load_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    for sep in (",", ";", "\t"):
        buf.seek(0)
        try:
            df_try = pd.read_csv(buf, sep=sep)
            if df_try.shape[1] > 1:
                return df_try
        except Exception:
            continue
    buf.seek(0)
    return pd.read_csv(buf)

def _detect_datetime_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    best_col, best_rate = None, 0.0
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(200)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        rate = parsed.notna().mean()
        if rate > 0.85 and rate > best_rate:
            best_col, best_rate = col, rate
    return best_col

def _classify_columns(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_col = _detect_datetime_column(df)
    categorical_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != datetime_col and 1 < df[c].nunique(dropna=True) <= 50
    ]
    return numeric_cols, categorical_cols, datetime_col

# Setup tabs
tab_analysis, tab_admin, tab_review = st.tabs(
    ["📊 Analyze Custom Dataset", "⚙️ Admin Data Manager Portal", "🩺 Human Review"]
)

# =====================================================================
# TAB 1: CUSTOM DATASET ANALYSIS & PROFILE
# =====================================================================
with tab_analysis:
    uploaded = st.file_uploader(
        "Upload a CSV or Excel file to analyze",
        type=["csv", "xlsx", "xls"],
        key="analysis_file_uploader",
        help="The file stays in this session only — nothing is written to disk permanently.",
    )

    if not uploaded:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color:{ACCENT}; padding:26px;">
                <div style="font-size:0.95rem; font-weight:700; color:{TEXT}; margin-bottom:6px;">
                    Drop in a dataset to get started
                </div>
                <div style="font-size:0.85rem; color:{MUTED}; line-height:1.55; max-width:680px;">
                    Upload any CSV or Excel file — it doesn't need to look like the bundled
                    surveillance data. The page will detect numeric, categorical, and date
                    columns automatically, generate KPIs and charts from whatever it finds,
                    and let you export the full analysis as a PDF.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            df_analysis = _load_file(uploaded.getvalue(), uploaded.name)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            st.stop()

        if df_analysis.empty or df_analysis.shape[1] == 0:
            st.warning("That file loaded but contains no usable rows/columns.")
        else:
            numeric_cols, categorical_cols, datetime_col = _classify_columns(df_analysis)
            if datetime_col:
                df_analysis[datetime_col] = pd.to_datetime(df_analysis[datetime_col], errors="coerce")

            n_rows, n_cols = df_analysis.shape
            missing_pct = round(df_analysis.isna().mean().mean() * 100, 2)
            dup_rows = int(df_analysis.duplicated().sum())

            k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
            with k1:
                kpi_card("Rows", f"{n_rows:,}", "records loaded")
            with k2:
                kpi_card("Columns", f"{n_cols:,}", f"{len(numeric_cols)} numeric · {len(categorical_cols)} categorical")
            with k3:
                kpi_card("Missing Data", f"{missing_pct}%", color=WARNING if missing_pct > 5 else SUCCESS,
                          bg="#FDEBD0" if missing_pct > 5 else "#D5F5E3")
            with k4:
                kpi_card("Duplicate Rows", f"{dup_rows:,}", color=DANGER if dup_rows else SUCCESS,
                          bg="#FADBD8" if dup_rows else "#D5F5E3")
            with k5:
                kpi_card("Date Field Found", datetime_col if datetime_col else "None",
                          color=SUCCESS if datetime_col else MUTED)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            with st.expander("🔍 Preview raw data (first 100 rows)", expanded=False):
                st.dataframe(df_analysis.head(100), use_container_width=True, height=320)

            insights: list[str] = []

            if numeric_cols:
                section_title("Distributions", "Shape of each numeric field — check spread, skew, and outliers")
                default_cols = numeric_cols[:4]
                picked_numeric = st.multiselect(
                    "Numeric columns to profile", options=numeric_cols, default=default_cols, key="numeric_picker",
                )
                if picked_numeric:
                    n = len(picked_numeric)
                    cols_per_row = 2
                    for row_start in range(0, n, cols_per_row):
                        row_cols = st.columns(cols_per_row, gap="large")
                        for offset, col_name in enumerate(picked_numeric[row_start:row_start + cols_per_row]):
                            with row_cols[offset]:
                                series = df_analysis[col_name].dropna()
                                fig = px.histogram(series, x=col_name, nbins=28, color_discrete_sequence=[ACCENT])
                                fig.update_traces(marker_line_color="white", marker_line_width=0.6, opacity=0.85)
                                fig.update_layout(
                                    height=260, margin=dict(l=10, r=10, t=10, b=10),
                                    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                                    font=dict(color=TEXT, size=11),
                                    xaxis=dict(title=col_name, tickfont=dict(color=TEXT)),
                                    yaxis=dict(title="Count", showgrid=True, gridcolor="#E4EAF0", tickfont=dict(color=TEXT)),
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                skew = series.skew()
                                if abs(skew) > 1:
                                    insights.append(
                                        f"**{col_name}** is {'right' if skew > 0 else 'left'}-skewed "
                                        f"(skew={skew:.2f}) — the mean will be pulled by outliers, prefer the median."
                                    )

            if len(numeric_cols) >= 2:
                section_title("Correlation Between Numeric Fields", "Which fields move together vs. independently")
                corr = df_analysis[numeric_cols].corr().round(2)
                # Intentionally NOT the shared Green->Yellow->Red risk scale:
                # correlation runs -1..+1 and a diverging red/blue scale is
                # the standard way to show "negative vs positive", not "low
                # vs high risk" — using the risk palette here would make a
                # strong negative correlation look "safe" (green) instead of
                # simply the opposite direction of a strong positive one.
                fig_corr = px.imshow(
                    corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                )
                fig_corr.update_traces(textfont=dict(size=10))
                fig_corr.update_layout(
                    height=min(120 + 46 * len(numeric_cols), 560), margin=dict(l=10, r=10, t=10, b=10),
                    font=dict(color=TEXT, size=11),
                    xaxis=dict(tickfont=dict(color=TEXT, size=10)),
                    yaxis=dict(tickfont=dict(color=TEXT, size=10)),
                    coloraxis_colorbar=dict(title="r", tickfont=dict(color=TEXT)),
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                pairs = []
                cols = corr.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        pairs.append((cols[i], cols[j], corr.iloc[i, j]))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                if pairs and abs(pairs[0][2]) > 0.5:
                    a, b, r = pairs[0]
                    direction = "positively" if r > 0 else "negatively"
                    insights.append(f"**{a}** and **{b}** are strongly {direction} correlated (r={r:.2f}).")

            if categorical_cols and numeric_cols:
                section_title("Compare a Metric Across Categories", "Bars show the mean; whiskers show one standard deviation")
                cc1, cc2 = st.columns(2, gap="large")
                with cc1:
                    cat_pick = st.selectbox("Category column", options=categorical_cols, key="cat_picker")
                with cc2:
                    num_pick = st.selectbox("Numeric metric", options=numeric_cols, key="num_picker")

                vc = df_analysis[cat_pick].value_counts()
                top_cats = vc[vc >= 2].head(10).index.tolist()
                grp = df_analysis[df_analysis[cat_pick].isin(top_cats)].groupby(cat_pick)[num_pick].agg(["mean", "std", "count"]).reindex(top_cats)
                if len(grp) >= 2:
                    fig_bar = go.Figure(go.Bar(
                        x=grp.index.astype(str), y=grp["mean"],
                        error_y=dict(type="data", array=grp["std"].fillna(0), color="#000000", thickness=1.3),
                        marker_color=ACCENT, marker_line_color="white", marker_line_width=0.6,
                    ))
                    fig_bar.update_layout(
                        height=380, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(color=TEXT, size=11),
                        xaxis=dict(tickfont=dict(color=TEXT, size=10)),
                        yaxis=dict(title=num_pick, showgrid=True, gridcolor="#E4EAF0", tickfont=dict(color=TEXT)),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    widest = grp["mean"].idxmax()
                    narrowest = grp["mean"].idxmin()
                    if widest != narrowest:
                        spread = grp["mean"].max() - grp["mean"].min()
                        insights.append(
                            f"For **{num_pick}**, **{widest}** averages the highest and **{narrowest}** the "
                            f"lowest across {cat_pick} — a gap of {spread:,.2f}."
                        )
                else:
                    st.info("Pick a category column with at least two groups of 2+ rows to compare.")

            if datetime_col and numeric_cols:
                section_title("Trend Over Time", f"Monthly average of a selected metric by {datetime_col}")
                trend_metric = st.selectbox("Metric to trend", options=numeric_cols, key="trend_metric_picker")
                ts = df_analysis[[datetime_col, trend_metric]].dropna().sort_values(datetime_col)
                ts_monthly = ts.groupby(pd.Grouper(key=datetime_col, freq="MS"))[trend_metric].mean().dropna()
                if len(ts_monthly) >= 2:
                    color = TREND_PALETTE[0]
                    fig_trend = go.Figure(go.Scatter(
                        x=ts_monthly.index, y=ts_monthly.values, mode="lines+markers",
                        line=dict(width=3, color=color, shape="spline"),
                        marker=dict(size=6, color=color, line=dict(width=1, color="white")),
                        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.16),
                    ))
                    fig_trend.update_layout(
                        height=360, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(color=TEXT, size=11),
                        xaxis=dict(tickfont=dict(color=TEXT, size=10)),
                        yaxis=dict(title=trend_metric, showgrid=True, gridcolor="#E4EAF0", tickfont=dict(color=TEXT)),
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Not enough distinct time points to plot a trend.")

            section_title("Key Insights", "Auto-generated from the patterns detected above")
            if insights:
                for line in insights:
                    st.markdown(f"- {line}")
            else:
                st.caption("No strong patterns (skew, correlation, or category gaps) were detected in this file.")

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            pdf_col1, pdf_col2 = st.columns([1, 3])
            with pdf_col1:
                generate = st.button("📄 Generate PDF Report", use_container_width=True, type="primary")
            with pdf_col2:
                st.caption("Builds a shareable PDF with the KPIs, summary stats, and charts above, computed from this file.")

            # Persist the generated PDF in session_state and render the
            # download button unconditionally (outside `if generate:`).
            # Otherwise the download_button's own click-triggered rerun makes
            # `generate` False again on the next run, the bytes disappear
            # from this block, and the download that was in flight gets
            # served empty/stale content instead of the real report.
            if generate:
                with st.spinner("Building PDF report..."):
                    plain_insights = [line.replace("**", "") for line in insights]
                    st.session_state["custom_analysis_pdf_bytes"] = build_pdf_report(
                        df=df_analysis, file_name=uploaded.name,
                        numeric_cols=numeric_cols, categorical_cols=categorical_cols,
                        datetime_col=datetime_col, insights=plain_insights,
                    )
                    st.session_state["custom_analysis_pdf_name"] = f"analysis_report_{uploaded.name.rsplit('.', 1)[0]}.pdf"

            if st.session_state.get("custom_analysis_pdf_bytes"):
                st.download_button(
                    "⬇️ Download analysis_report.pdf",
                    data=st.session_state["custom_analysis_pdf_bytes"],
                    file_name=st.session_state.get("custom_analysis_pdf_name", "analysis_report.pdf"),
                    mime="application/pdf", use_container_width=True,
                    key="custom_analysis_pdf_download_btn",
                )

# =====================================================================
# TAB 2: ADMIN DATA MANAGER PORTAL
# =====================================================================
with tab_admin:
    st.write("### 🗄️ Save & Append Uploaded Datasets directly to Database")
    st.caption("Admin login required. Perform direct file operations on the `data/` folder and dynamically reload the active cache.")

    # --- Login gate (Issue #2 fix: portal was previously open to everyone) ---
    if not render_admin_login_form("admin_login_form_portal"):
        st.stop()

    uploaded_admin = st.file_uploader(
        "Upload a dataset to update/save",
        type=["csv"],
        key="admin_file_uploader",
    )

    if uploaded_admin:
        try:
            df_admin = pd.read_csv(uploaded_admin)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

        target_select = st.selectbox(
            "Select target destination file",
            options=list(TARGET_FILES.keys()),
            key="admin_target_select"
        )

        target_filename = TARGET_FILES[target_select]

        # If new custom filename — strip path components to prevent traversal
        custom_name = ""
        if target_filename == "custom":
            raw_name = st.text_input("Enter filename (e.g. fact_vaccinations_cleaned.csv)", value="fact_custom_dataset.csv")
            # Issue #3 fix: use Path.name to strip any directory traversal (e.g. ../../app.py)
            import pathlib
            safe_name = pathlib.Path(raw_name.strip()).name
            if safe_name != raw_name.strip():
                st.warning(f"⚠️ Filename sanitized to `{safe_name}` — directory traversal components removed.")
            custom_name = safe_name
            target_filename = custom_name

        if target_filename:
            st.write(f"**Target file:** `data/{target_filename}`")

            # Validation block
            validation_success = False
            registered_filename = custom_name if target_select == "Upload New / Custom Dataset" else target_filename
            if target_select != "Upload New / Custom Dataset" and target_filename in EXPECTED_COLUMNS:
                expected_cols = EXPECTED_COLUMNS[target_filename]
                uploaded_cols = df_admin.columns.tolist()

                missing = [c for c in expected_cols if c not in uploaded_cols]

                if not missing:
                    st.markdown(
                        f"""
                        <div style="background-color:#D5F5E3; border-left:4px solid #16855B; padding:10px; border-radius:4px; margin-bottom:12px;">
                            <strong style="color:#16855B;">✅ Schema Validation Successful</strong><br>
                            All {len(expected_cols)} required columns are present in the uploaded dataset.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    validation_success = True
                else:
                    st.markdown(
                        f"""
                        <div style="background-color:#FADBD8; border-left:4px solid #C43D3D; padding:10px; border-radius:4px; margin-bottom:12px;">
                            <strong style="color:#C43D3D;">⚠️ Schema Validation Failure</strong><br>
                            The uploaded dataset is missing the following columns:<br>
                            <code>{', '.join(missing)}</code>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("ℹ️ Custom or unregistered filename. Manual structure check suggested.")
                validation_success = True  # allow uploads of new arbitrary files

            # Selection of operation mode
            write_mode = st.radio("Operation Mode", ["Append to Existing (Merge)", "Overwrite Existing (Replace)"])

            if st.button("💾 Save & Update Dashboards", type="primary"):
                import pathlib as _pl
                target_path = _pl.Path("data") / pathlib.Path(target_filename).name

                try:
                    if write_mode == "Overwrite Existing (Replace)" or not target_path.exists():
                        df_admin.to_csv(target_path, index=False)
                        st.success(f"Successfully wrote {len(df_admin)} rows to `data/{target_filename}` (Overwritten).")
                    else:
                        existing_df = pd.read_csv(target_path)
                        combined_df = pd.concat([existing_df, df_admin], ignore_index=True)
                        pk = "outbreak_id" if "outbreak_id" in combined_df.columns else "fact_id"
                        if pk in combined_df.columns:
                            combined_df.drop_duplicates(subset=[pk], keep="last", inplace=True)
                        else:
                            combined_df.drop_duplicates(keep="last", inplace=True)
                        combined_df.to_csv(target_path, index=False)
                        st.success(f"Successfully merged data. Total rows in database: {len(combined_df)}")

                    # Clear Streamlit's cache so dashboards reload fresh data
                    st.cache_data.clear()
                    st.info("🔄 Streamlit cache invalidated. Dashboards will reload the clean data on next view.")

                except Exception as e:
                    st.error(f"Failed to write to file: {e}")

# =====================================================================
# TAB 3: HUMAN REVIEW
# =====================================================================
with tab_review:
    st.write("### 🩺 Human Review")
    st.caption(
        "Admin login required. Upload one or more PDF reports — each moves through its own "
        "Pending Review → Reviewed → Approved workflow. All comments and reviewer activity are "
        "saved to a database, so nothing is lost between sessions."
    )

    if render_admin_login_form("admin_login_form_review"):
        uploaded_review_files = st.file_uploader(
            "Upload one or more PDF reports for review",
            type=["pdf"],
            accept_multiple_files=True,
            key="review_file_uploader",
        )

        if uploaded_review_files:
            st.success(f"{len(uploaded_review_files)} report(s) uploaded successfully!")

            # Build a stable identity for each uploaded file (name + content
            # hash) so two different files that happen to share a filename
            # are still tracked — and reviewed — separately.
            review_files = []
            for f in uploaded_review_files:
                pdf_bytes = f.getvalue()
                file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]
                review_files.append({
                    "name": f.name,
                    "bytes": pdf_bytes,
                    "hash": file_hash,
                    "identity": f"{f.name}::{file_hash}",
                })

            # One tab per uploaded PDF, so each file's viewer, status, and
            # comment box are clearly scoped to that file only.
            file_tab_labels = [
                f"📄 {rf['name']}" if len(rf["name"]) <= 28 else f"📄 {rf['name'][:25]}…"
                for rf in review_files
            ]
            file_tabs = st.tabs(file_tab_labels)

            for rf, file_tab in zip(review_files, file_tabs):
                with file_tab:
                    st.write(f"**File:** `{rf['name']}` &nbsp;·&nbsp; **Size:** {len(rf['bytes']) / 1024:.1f} KB")

                    # View the PDF inline — no download needed.
                    _render_pdf_inline(rf["bytes"])

                    # ---------------------------------------------------
                    # Review workflow — Pending Review -> Reviewed ->
                    # Approved. Current status is derived from the most
                    # recent database entry for this exact file (by name +
                    # content hash), so it's consistent across reruns and
                    # logins, not just session state.
                    # ---------------------------------------------------
                    review_status = _latest_status_for_file(rf["name"], rf["hash"])

                    section_title("Review Status", "This upload must be reviewed and approved by an admin")
                    st.markdown(
                        f"""
                        <span style="
                            display:inline-block; padding:4px 12px; border-radius:999px;
                            background:{REVIEW_STAGE_COLOR[review_status]}1A;
                            color:{REVIEW_STAGE_COLOR[review_status]}; font-weight:700; font-size:0.85rem;
                            border:1px solid {REVIEW_STAGE_COLOR[review_status]};
                        ">● {review_status}</span>
                        """,
                        unsafe_allow_html=True,
                    )

                    review_action_col1, review_action_col2 = st.columns([1, 3])
                    with review_action_col1:
                        if review_status in REVIEW_STAGE_NEXT_LABEL:
                            advance_clicked = st.button(
                                REVIEW_STAGE_NEXT_LABEL[review_status],
                                type="primary", use_container_width=True,
                                key=f"advance_review::{rf['identity']}",
                            )
                        else:
                            advance_clicked = False
                            st.success("✅ Approved.")
                    with review_action_col2:
                        if review_status == "Pending Review":
                            st.caption("Add a comment below, then mark this upload as reviewed.")
                        elif review_status == "Reviewed":
                            st.caption("Approve once you're confident this report is correct and complete.")

                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    reviewer_comment = st.text_area(
                        "Reviewer comment",
                        placeholder="Add a note or feedback about this report...",
                        key=f"reviewer_comment::{rf['identity']}",
                    )

                    if advance_clicked:
                        if not reviewer_comment.strip():
                            st.warning("Please add a comment before advancing this upload's status.")
                        else:
                            next_status = REVIEW_STAGES[REVIEW_STAGES.index(review_status) + 1]
                            _append_upload_review_log({
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "admin_username": st.session_state.get("admin_username", ""),
                                "admin": st.session_state.get("admin_full_name", st.session_state.get("admin_username", "")),
                                "file_name": rf["name"],
                                "file_hash": rf["hash"],
                                "status": next_status,
                                "comment": reviewer_comment.strip(),
                            })
                            st.rerun()

                    # Review history for this specific file only.
                    file_history = _load_upload_review_log(rf["name"], rf["hash"])
                    if file_history:
                        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                        with st.expander(f"📜 Review history for this file ({len(file_history)})"):
                            for h in reversed(file_history):
                                st.markdown(
                                    f"**{h['status']}** — {h['admin_full_name']} · {h['timestamp']}<br>"
                                    f"<span style='color:{MUTED};'>{h['comment']}</span>",
                                    unsafe_allow_html=True,
                                )

            # =============================================================
            # Combined history across ALL reviewed files — makes it clear
            # at a glance which comment belongs to which PDF, who wrote it,
            # and when. Backed by the same SQLite database.
            # =============================================================
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            section_title("📚 Full Review History", "Every comment ever logged, across all uploaded reports")

            all_history = _load_upload_review_log()
            if all_history:
                uploaded_names = sorted({rf["name"] for rf in review_files})
                filter_choice = st.selectbox(
                    "Filter by file",
                    ["All currently uploaded files"] + uploaded_names + ["All files ever reviewed"],
                    key="review_history_filter",
                )

                if filter_choice == "All currently uploaded files":
                    identities = {rf["identity"] for rf in review_files}
                    filtered = [h for h in all_history if f"{h['file_name']}::{h['file_hash']}" in identities]
                elif filter_choice == "All files ever reviewed":
                    filtered = all_history
                else:
                    filtered = [h for h in all_history if h["file_name"] == filter_choice]

                if filtered:
                    history_df = pd.DataFrame(filtered)[
                        ["file_name", "status", "admin_full_name", "timestamp", "comment"]
                    ].rename(columns={
                        "file_name": "PDF File",
                        "status": "Status",
                        "admin_full_name": "Reviewer",
                        "timestamp": "Timestamp",
                        "comment": "Comment",
                    }).sort_values("Timestamp", ascending=False)
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No review history matches this filter yet.")
            else:
                st.caption("No review history logged yet — add a comment above to get started.")

