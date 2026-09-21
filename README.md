# Public Health Analytics Dashboard

Streamlit multipage application presenting the Public Health Analytics
dashboard suite from a single entry point, built on a star-schema data
model (dimension + fact CSV extracts).

`app.py` links all 5 dashboards together with a single left-hand
navigation menu (`st.navigation`), **Executive Public Health Overview**
loading first by default.

| # | Dashboard | Status |
|---|---|---|
| 1 | Executive Public Health Overview | ✅ Built — 2 tabs: Executive Summary + Disease Surveillance |
| 2 | Geographic & Environmental Intelligence | ⬜ Empty |
| 3 | Laboratory & Healthcare Capacity | ⬜ Empty |
| 4 | Outbreak Monitoring & Forecasting | ⬜ Empty |
| 5 | Health Programs & Population Vulnerability | ⬜ Empty |

Disease Surveillance content (disease-wise trends, state heatmap, outbreak
alerts, testing/positivity) now lives **inside** the Executive Public Health
Overview page as its own tab, sharing the same sidebar filter panel as the
Executive Summary tab — no separate nav entry, no duplicate filters.

The four empty dashboards already have their data available through
`src/data_loader.py` (one loader function per fact table: `get_outbreak_master()`,
`get_environmental_master()`, `get_programs_master()`, `get_lab_master()`) —
build the visuals directly on top of that, no data plumbing needed.

## Project Structure

```
dashboard/
├── app.py                                              # Entry point — wires all 5 dashboards into one nav menu
├── dashboards/
│   ├── 0_Executive_Public_Health_Overview.py           # ✅ Built — 2 tabs: Executive Summary + Disease Surveillance
│   ├── 1_Geographic_Environmental_Intelligence.py      # ⬜ Empty placeholder
│   ├── 2_Laboratory_Healthcare_Capacity.py             # ⬜ Empty placeholder
│   ├── 3_Outbreak_Monitoring_Forecasting.py            # ⬜ Empty placeholder
│   └── 4_Health_Programs_Population_Vulnerability.py   # ⬜ Empty placeholder
├── src/
│   ├── data_loader.py             # Cached CSV loading + star-schema joins (all 5 fact tables ready)
│   ├── filters.py                 # Shared sidebar filter panel
│   ├── kpis.py                    # KPI calculation logic (unit-testable)
│   └── styling.py                 # Shared CSS + reusable UI components
├── data/                          # Cleaned CSV extracts (dim_*, fact_*)
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
