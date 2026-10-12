# HealthSentinel: Public Health Analytics & Epidemiological Research Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Benchmark](https://img.shields.io/badge/Research_Grade-Peer_Reviewed-green.svg)](#research-methodology--statistical-validation-suite)

**HealthSentinel** is an enterprise-grade, peer-review ready **Public Health Analytics and Epidemiological Decision Support Suite** tailored for disease surveillance, capacity planning, and outbreak management across 32 Indian States and Union Territories.

Built on an analytical star-schema data warehouse, the platform integrates **multi-model time-series forecasting, CDC EARS outbreak aberration detection, distribution-free conformal surge bounds, causal policy counterfactual simulation, algorithmic subgroup fairness auditing, and an offline-grounded / Gemini LLM epidemiological copilot**.

---

## 🧭 Dashboard Architecture & Capabilities

`app.py` serves as the central router connecting the suite via Streamlit's unified navigation system (`st.navigation`):

| # | Dashboard | Status | Capabilities & Research Features |
|---|---|---|---|
| **0** | **Home** | ✅ Active | Platform mission, executive summary, and multi-dashboard navigation directory. |
| **1** | **Executive Public Health Overview** | ✅ Active | Dual-tab national burden tracking, volume-weighted CFR/Recovery KPIs, and disease category heatmaps. |
| **2** | **Geographic & Environmental Intelligence** | ✅ Active | Spatial transmission risk, AQI, monsoon rainfall, and sanitation correlation analytics across Indian states. |
| **3** | **Laboratory & Healthcare Capacity** | ✅ Active | Diagnostic test throughput, lab positivity rates, vaccination curves, and critical care (ICU/bed) saturation stress. |
| **4** | **Outbreak Monitoring & Forecasting** | ✅ Upgraded | **6 Integrated Tabs**: Outbreak Monitoring, Multi-Model Forecasting, Anomaly Surge Detection, Decision Matrix, Alert Response Queue, and **🔬 Research Benchmark & Validation**. |
| **5** | **Health Programs & Population Vulnerability** | ✅ Active | Welfare scheme coverage (PM-JAY, NHM), high-risk cohort tracking, and demographic vulnerability indices. |
| **6** | **Upload & Custom Analysis** | ✅ Active | Data profiler, authenticated schema validation portal, SQLite audit review logs, and automated PDF report export. |
| **7** | **Ask Sentinel (AI Copilot)** | ✅ Active | Zero-dependency offline semantic synthesis engine, Google Gemini LLM bridge, and instant Executive Situation Report (SITREP) generator. |

---

## 🔬 Research Methodology & Statistical Validation Suite

The predictive core in [`src/ml_models.py`](src/ml_models.py) implements rigorous statistical, epidemiological, and causal machine learning protocols:

### 1. Multi-Model Epidemiological Forecasting
* **Log-Stabilized Box-Jenkins $\text{ARIMA}(1,1,1)$**:
  $$\Delta^d \log(Y_t + 1) = c + \sum_{i=1}^p \phi_i \Delta^d \log(Y_{t-i} + 1) + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$
* **Holt-Winters Additive Damped Trend** (Gardner & McKenzie, 1985):
  $$\hat{Y}_{t+h|t} = \ell_t + \sum_{i=1}^h \phi^i b_t$$
* **Diebold-Mariano Predictive Equivalence Hypothesis Test** (Diebold & Mariano, 1995):
  Tests $H_0: \mathbb{E}[d_t] = 0$ (equal predictive accuracy) with the Harvey-Leybourne-Newbold (HLN, 1997) finite-sample correction:
  $$\text{HLN} = \sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}}, \quad \text{DM}^* = \text{HLN} \cdot \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \sim t_{T-1}$$
* **Non-Parametric Empirical Bootstrap Confidence Intervals**:
  Resamples hold-out forecast residuals over $B = 1,000$ iterations to compute empirical $[q_{\alpha/2}, q_{1 - \alpha/2}]$ 95% confidence intervals on out-of-sample MAE, RMSE, and MAPE.
* **Gneiting-Raftery Proper Scoring Rules (Weighted Interval Score)**:
  $$\text{IS}_\alpha(L, U, y) = (U - L) + \frac{2}{\alpha}(L - y)\mathbb{I}(y < L) + \frac{2}{\alpha}(y - U)\mathbb{I}(y > U)$$

### 2. CDC EARS & Conformal Surge Detection
* **CDC Early Aberration Reporting System (EARS)** (Hutwagner et al., 2003):
  * **$C_1(t)$**: Immediate 7-period baseline $(t-7 \dots t-1)$ sensitive to acute spikes.
  * **$C_2(t)$**: 7-period baseline with a 2-period protective buffer $(t-9 \dots t-3)$ to prevent outbreak spillover from contaminating the reference mean.
  * **$C_3(t)$**: 3-period cumulative aberration $\sum_{k=0}^2 \max(0, C_2(t-k) - 1)$ sensitive to insidious, slow-building transmission clusters.
* **Conformalized Quantile Residual Bounds** (Vovk et al., 2005):
  Distribution-free $(1 - \alpha)$ non-conformity threshold with certified finite-sample false-discovery control:
  $$\text{Threshold}_t = \hat{\mu}_t + \text{Quantile}_{1-\alpha}(R) \cdot \left(1 + \frac{1}{k}\right)$$

### 3. Inverse Policy Optimization & Algorithmic Fairness
* **Causal Policy Counterfactual Optimization** (Wachter et al., 2017):
  $$\arg\min_{\Delta x} \|\Delta x\|_2 \quad \text{s.t.} \quad f(x + \Delta x) = y_{\text{target}}$$
  Calculates the minimum necessary intervention on operational levers (`response_time_hours`, `hospital_readiness_score`) required to de-escalate an outbreak from **High Alert** to **Low Alert**.
* **Algorithmic Subgroup Fairness Audit** (Barocas et al., 2019):
  Audits alert distribution equity across administrative jurisdictions:
  * **Disparate Impact**: $\frac{\min P(\hat{Y}=\text{High}|A=a)}{\max P(\hat{Y}=\text{High}|A=a)} \ge 0.80$ (EEOC Four-Fifths Rule).
  * **Equal Opportunity Parity**: $\max_a \text{TPR}_a - \min_a \text{TPR}_a \le 0.20$.

---

## 📁 Repository Structure

```
HEALTH SENTINAL FINALL/
├── .streamlit/
│   └── config.toml                         # Visual design tokens & theme configuration
├── assets/                                 # Brand assets, logos, and UI hero graphics
├── dashboards/                             # Multipage dashboards routed via app.py
│   ├── 00_Home.py                          # Landing hero portal & real-time advisory ticker
│   ├── 0_Executive_Public_Health_Overview.py # Executive Summary & Disease Surveillance KPIs
│   ├── 1_Geographic_Environmental_Intelligence.py # Geospatial maps, AQI & climate correlations
│   ├── 2_Laboratory_Healthcare_Capacity.py # Lab testing, ICU/bed occupancy & clinical capacity
│   ├── 3_Outbreak_Monitoring_Forecasting.py # Forecasts, EARS, Counterfactuals & Research Validation
│   ├── 4_Health_Programs_Population_Vulnerability.py # Welfare scheme coverage & vulnerability scoring
│   ├── 5_Upload_Custom_Analysis.py         # Open data profiler, ingestion portal & gated admin review
│   └── 6_Ask_Sentinel_AI.py                # Dual-mode AI copilot chat interface & SITREP briefing
├── data/                                   # Cleaned star-schema CSV tables & audit logs
│   ├── dim_dates_cleaned.csv               # Monthly date dimension (2021–2023)
│   ├── dim_state_cleaned.csv               # Geographic dimension (32 Indian states/UTs)
│   ├── dim_disease_cleaned.csv             # Disease master dimension (12 diseases)
│   ├── dim_source_cleaned.csv              # Reporting source dimension (6 channels)
│   ├── dim_program_cleaned.csv             # Public health schemes dimension
│   ├── fact_disease_surveillance_cleaned.csv # Granular monthly disease surveillance records (13,824 rows)
│   ├── fact_environmental_cleaned.csv      # Environmental & climatic metrics per state/month
│   ├── fact_lab_healthcare_cleaned.csv     # Diagnostic testing, bed & ICU capacity metrics
│   ├── fact_health_programs_cleaned.csv    # Program coverage & demographic reach
│   ├── fact_outbreak_cleaned.csv           # Outbreak tracking, containment & alert levels
│   ├── review_history.db                   # SQLite database for audit lifecycle logging
│   └── outbreak_alert_queue_state.json     # Incident triage state machine persistence
├── src/                                    # Core analytical logic & reusable modules
│   ├── ml_models.py                        # Research-grade ML, Diebold-Mariano, EARS & Counterfactuals
│   ├── ai_copilot.py                       # Offline semantic synthesis & Gemini LLM bridge
│   ├── data_loader.py                      # Cached CSV ingest & star-schema join engine
│   ├── geographic.py                       # Geospatial calculations, Plotly map & gauge builders
│   ├── kpis.py                             # Public health rate calculations (unbiased aggregation)
│   ├── pdf_report.py                       # ReportLab PDF compilation engine
│   └── styling.py                          # CSS injection, KPI cards, banners & layout
├── app.py                                  # Main application entrypoint (st.navigation)
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation & scientific specification
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies in a virtual environment:
```bash
git clone https://github.com/NanubalaSravani/MINI-PROJECT.git
cd MINI-PROJECT
pip install -r requirements.txt
```

### 2. Launch the Application
Start the interactive Streamlit command center:
```bash
streamlit run app.py
```
The suite will open at `http://localhost:8501`.

### 3. Run the Research Benchmark Suite via Python CLI
Execute the master scientific benchmark programmatically:
```python
import pandas as pd
from src.data_loader import get_surveillance_master, get_outbreak_master
from src.ml_models import run_research_benchmark_suite

surveillance_df = get_surveillance_master()
outbreak_df = get_outbreak_master()

results = run_research_benchmark_suite(surveillance_df, outbreak_df)
print("Diebold-Mariano Result:", results["diebold_mariano_test"])
print("Aberrations Summary:", results["aberration_detection_summary"])
print("Fairness Audit:", results["fairness_audit"]["fairness_compliant"])
```

---

## 📊 Analytical Star Schema

All rate-based metrics (Case Fatality Rate, Recovery Rate, Bed/ICU Occupancy) are dynamically recomputed from aggregated sums rather than averaged row-level percentages to eliminate population weighting distortion:

| Fact Table | Grain | Attributes |
|---|---|---|
| `fact_disease_surveillance` | State × Date × Disease × Source | Cases, deaths, CFR, recovery rate, hospitalization, ICU admissions, public health risk score |
| `fact_environmental` | State × Date | AQI, rainfall, temperature, water quality, mosquito breeding index, environmental risk index |
| `fact_lab_healthcare` | State × Date | Tests conducted, positive tests, positivity rate, vaccination rates, hospital beds, ICU utilization % |
| `fact_health_programs` | State × Date × Program | Scheme coverage %, beneficiaries reached, high-risk screening, health vulnerability index |
| `fact_outbreak` | State × Date × Disease × Source | Alert level (High/Moderate/Low), containment rate %, response time (hours), predicted cases |

---

## 📝 Reporting Standards & Scientific Citation

HealthSentinel's experimental design conforms to the **TRIPOD+AI** (*Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis + AI*) reporting guidelines.

If utilizing this codebase or methodology in academic research, please cite:
```bibtex
@software{healthsentinel2026,
  author = {Nanubala, Sravani},
  title = {HealthSentinel: Public Health Analytics & Epidemiological Research Suite},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/NanubalaSravani/MINI-PROJECT}
}
```
