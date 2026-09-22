# HealthSentinel: Public Health Analytics & Epidemiological Decision Support Suite

A Streamlit multipage application presenting an end-to-end Public Health Analytics suite for India, built on a star-schema data model (dimension + fact CSV extracts) with integrated machine learning forecasting and generative AI copilot capabilities.

`app.py` links the entire suite together with a unified left-hand navigation menu (`st.navigation`), **Home** loading first by default.

| # | Dashboard | Status | Description & Capabilities |
|---|---|---|---|
| 0 | Home | ✅ Built | Landing hero portal, platform mission, and dashboard navigation directory |
| 1 | Executive Public Health Overview | ✅ Built | 2 tabs: Executive Summary + Disease Surveillance, national burden & KPIs |
| 2 | Geographic & Environmental Intelligence | ✅ Built | Spatial risks, AQI/sanitation/rainfall correlation against epidemiological trends |
| 3 | Laboratory & Healthcare Capacity | ✅ Built | Testing volumes, positivity rates, vaccination progress & hospital/ICU bed stress |
| 4 | Outbreak Monitoring & Forecasting | ✅ Built | 4 tabs: Outbreak Surveillance, Multi-Model Forecasting (ARIMA/Holt-Winters with 80%/95% CI), Anomaly/Surge Detection (Z-Score), and Decision Matrix with Random Forest risk driver classification |
| 5 | Health Programs & Population Vulnerability | ✅ Built | Health program coverage, beneficiary reach, and demographic vulnerability indices |
| 6 | Upload & Custom Analysis | ✅ Built | Authenticated data ingestion portal, strict schema validation, SQLite review log, and automated PDF report generation |
| 7 | Ask Sentinel (AI Copilot) | ✅ Built | Offline-grounded natural language epidemiological Q&A, optional Gemini LLM reasoning, and one-click Executive Situation Report (SITREP) generator |

## Project Structure

```
dashboard/
├── app.py                                              # Main entry point — wires all dashboards into unified st.navigation
├── dashboards/
│   ├── 00_Home.py                                      # Landing hero page & dashboard directory
│   ├── 0_Executive_Public_Health_Overview.py           # Executive Summary + Disease Surveillance
│   ├── 1_Geographic_Environmental_Intelligence.py      # Spatial & environmental risk analytics
│   ├── 2_Laboratory_Healthcare_Capacity.py             # Lab testing, positivity, ICU & bed capacity
│   ├── 3_Outbreak_Monitoring_Forecasting.py            # Outbreak tracking, ARIMA/Holt-Winters, Anomaly detection, ML risk drivers
│   ├── 4_Health_Programs_Population_Vulnerability.py   # Health scheme coverage & vulnerability scoring
│   ├── 5_Upload_Custom_Analysis.py                     # Secure upload portal, custom review, automated PDF export
│   └── 6_Ask_Sentinel_AI.py                            # Natural language query copilot & automated SITREP generator
├── src/
│   ├── data_loader.py             # Cached CSV loading + star-schema joins
│   ├── ml_models.py               # Time-series forecasting, surge detection, RF feature importance
│   ├── ai_copilot.py              # Natural language synthesis & Executive SITREP generator
│   ├── report_generator.py        # PDF & analytical report generation logic
│   ├── filters.py                 # Shared sidebar filter panel
│   ├── kpis.py                    # KPI calculation logic (unit-testable)
│   └── styling.py                 # Shared CSS, theme tokens, & reusable UI components
├── data/                          # Cleaned CSV extracts (dim_*, fact_*) & audit logs
├── assets/                        # Logos, icons, and hero graphics
├── .streamlit/
│   └── config.toml                # Corporate theme configuration
├── requirements.txt
└── README.md
```

> Note: the pages folder is named `dashboards/`, not `pages/` — Streamlit
> reserves the literal `pages/` folder name for its older auto-navigation
> feature, which conflicts with the `st.navigation` API used in `app.py`.

## Adding a new dashboard later

1. Drop a new file in `dashboards/`.
2. Add one `st.Page("dashboards/your_file.py", title="Your Title")` line
   in `app.py` and include it in the list passed to `st.navigation([...])`.

No other file needs to change.

## Running Locally

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Use the left sidebar to switch
between dashboards, and to filter by Region, State, Year, Month, Disease
Type, Disease, and Primary Source (on dashboards that use
`render_sidebar_filters()`).

## Data Model

| Table | Grain | Key Columns |
|---|---|---|
| `dim_dates` | 1 row per month | `date_id`, `year`, `month_name`, `quarter` |
| `dim_state` | 1 row per state | `state_id`, `state_name`, `region`, `population` |
| `dim_disease` | 1 row per disease | `disease_id`, `disease_name`, `disease_category` |
| `dim_source` | 1 row per reporting source | `source_id`, `source_name` |
| `dim_program` | 1 row per health program | `program_id`, `program_name` |
| `fact_disease_surveillance` | state × date × disease × source | cases, deaths, CFR, recovery rate, risk score |
| `fact_outbreak` | state × date × disease × source | outbreak/alert/containment metrics |
| `fact_environmental` | state × date | AQI, rainfall, sanitation, environmental risk |
| `fact_health_programs` | state × date × program | coverage, beneficiaries, vulnerability index |
| `fact_lab_healthcare` | state × date | testing, positivity, vaccination, infrastructure |

## KPI Definitions (Executive Public Health Overview)

| KPI | Formula |
|---|---|
| Total Population Under Surveillance | Sum of distinct `population_under_surveillance` per state/date |
| Total Reported Cases | Sum of `total_reported_cases` |
| Active Cases | Sum of `active_cases` |
| Recovered Cases | Sum of `recovered_cases` |
| Deaths (Monthly) | Sum of `deaths` |
| Case Fatality Rate | Deaths ÷ Total Reported Cases × 100 |
| Recovery Rate | Recovered ÷ Total Reported Cases × 100 |
| Public Health Risk Score | Mean of `public_health_risk_score` across filtered records |

## Notes

- Rates (CFR, Recovery Rate, Hospitalization/ICU Rate) are **recomputed from
  aggregated totals** rather than averaged row-level percentages, to avoid
  bias when aggregating across states/diseases of very different case volume.
- Conditional formatting on the State Ranking table uses `pandas.Styler`
  background gradients (Cases → Blue, CFR → Red, Recovery Rate → Green).
