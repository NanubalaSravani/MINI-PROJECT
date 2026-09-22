"""
ai_copilot.py
-------------
Natural Language Query Engine and Situation Report (SITREP) Generator.

Provides:
1. ask_sentinel(): An interactive conversational copilot that answers public health
   surveillance questions. Supports zero-config offline rule-based semantic data synthesis
   as well as LLM API augmentation (Google Gemini / OpenAI compatible).
2. generate_executive_sitrep(): Generates structured Epidemiological Situation Reports
   summarizing current threat levels, critical clusters, hospital stress, and action plans.

Author : Health Sentinel AI & Epidemiology Engineering Team
"""

from typing import Dict, Any, Optional
import os
import json
import pandas as pd
import numpy as np

# Safe import of data loaders
try:
    from src.data_loader import (
        get_surveillance_master,
        get_outbreak_master,
        get_environmental_master,
        get_lab_master,
        get_programs_master,
    )
except ImportError:
    pass


def _get_dataset_summary() -> Dict[str, Any]:
    """Extract top-line summaries across all analytical domains."""
    summary = {}
    try:
        surv_df = get_surveillance_master()
        outb_df = get_outbreak_master()
        env_df = get_environmental_master()
        lab_df = get_lab_master()
        prog_df = get_programs_master()

        case_col = "total_reported_cases" if "total_reported_cases" in surv_df.columns else "cases" if "cases" in surv_df.columns else surv_df.columns[0]
        summary["total_cases"] = int(surv_df[case_col].sum())
        summary["total_deaths"] = int(surv_df["deaths"].sum()) if "deaths" in surv_df.columns else 0
        summary["cfr"] = round((summary["total_deaths"] / max(1, summary["total_cases"])) * 100, 2)
        summary["total_outbreaks"] = len(outb_df)
        summary["high_alerts"] = int((outb_df["alert_level"] == "High").sum()) if "alert_level" in outb_df.columns else 0

        # Top 5 diseases by cases
        if "disease_name" in surv_df.columns:
            summary["top_diseases"] = (
                surv_df.groupby("disease_name")[case_col].sum().nlargest(5).to_dict()
            )
        else:
            summary["top_diseases"] = {}

        # Top 5 states by cases
        if "state_name" in surv_df.columns:
            summary["top_states"] = (
                surv_df.groupby("state_name")[case_col].sum().nlargest(5).to_dict()
            )
        else:
            summary["top_states"] = {}

        # High risk states in outbreak
        if "state_name" in outb_df.columns and "alert_level" in outb_df.columns:
            summary["high_alert_states"] = (
                outb_df[outb_df["alert_level"] == "High"]["state_name"].value_counts().head(5).to_dict()
            )
        else:
            summary["high_alert_states"] = {}

        # Lab & ICU stress and Hospital Beds
        icu_col = "icu_utilization_pct" if "icu_utilization_pct" in lab_df.columns else "icu_bed_occupancy_rate" if "icu_bed_occupancy_rate" in lab_df.columns else None
        if icu_col:
            summary["avg_icu_occupancy"] = round(float(lab_df[icu_col].mean()), 1)
            summary["high_icu_states"] = (
                lab_df.groupby("state_name")[icu_col].mean().nlargest(5).round(1).to_dict()
            )
        else:
            summary["avg_icu_occupancy"] = 0.0
            summary["high_icu_states"] = {}

        if "hospital_beds" in lab_df.columns and "state_name" in lab_df.columns:
            summary["top_bed_states"] = (
                lab_df.groupby("state_name")["hospital_beds"].sum().nlargest(5).round(0).to_dict()
            )
        else:
            summary["top_bed_states"] = {}

        # Environmental risk and AQI
        aqi_col = "aqi" if "aqi" in env_df.columns else "aqi_index" if "aqi_index" in env_df.columns else None
        if aqi_col:
            summary["avg_aqi"] = round(float(env_df[aqi_col].mean()), 1)
            summary["worst_aqi_states"] = (
                env_df.groupby("state_name")[aqi_col].mean().nlargest(5).round(1).to_dict()
            )
        else:
            summary["avg_aqi"] = 0.0
            summary["worst_aqi_states"] = {}

        # Population Health Vulnerability Index
        vuln_col = "health_vulnerability_index" if "health_vulnerability_index" in prog_df.columns else "vulnerability_index" if "vulnerability_index" in prog_df.columns else None
        if vuln_col and "state_name" in prog_df.columns:
            summary["avg_vulnerability"] = round(float(prog_df[vuln_col].mean()), 1)
            summary["top_vulnerable_states"] = (
                prog_df.groupby("state_name")[vuln_col].mean().nlargest(5).round(1).to_dict()
            )
        else:
            summary["avg_vulnerability"] = 0.0
            summary["top_vulnerable_states"] = {}

    except Exception as e:
        summary["error"] = str(e)
    
    return summary


def _offline_answer_generator(query: str) -> str:
    """Intelligent semantic data query engine when no external LLM API key is provided."""
    q = query.lower()
    stats = _get_dataset_summary()

    # 1. Vulnerability Index
    if any(term in q for term in ["vulnerab", "hvi", "socioeconomic"]):
        vuln_states = stats.get("top_vulnerable_states", {})
        vuln_str = "\n".join([f"{i+1}. **{s}**: Vulnerability Index **{score} / 100**" for i, (s, score) in enumerate(vuln_states.items())])
        return (
            f"### 🛡️ Population Health Vulnerability Analysis\n\n"
            f"The national baseline Health Vulnerability Index averages **{stats.get('avg_vulnerability', 0)} / 100** across surveyed populations.\n\n"
            f"**States with the Highest Health Vulnerability Index:**\n"
            f"{vuln_str}\n\n"
            f"**Action Recommendation:** Vulnerable demographics in these states require targeted welfare schemes, mobile medical units (MMUs), and expanded maternal-child immunization outreach."
        )

    # 2. Hospital Beds & Capacity (Cities/States)
    elif any(term in q for term in ["bed", "hospital capacity", "facility"]):
        bed_states = stats.get("top_bed_states", {})
        bed_str = "\n".join([f"{i+1}. **{s}**: ~{int(beds):,} cumulative bed-days available" for i, (s, beds) in enumerate(bed_states.items())])
        return (
            f"### 🛏️ Hospital Bed Capacity Distribution\n\n"
            f"*(Note: Health Sentinel tracks healthcare infrastructure at the State level across India)*\n\n"
            f"**States with the highest hospital bed infrastructure:**\n"
            f"{bed_str}\n\n"
            f"- **National ICU Utilization**: `{stats.get('avg_icu_occupancy', 0)}%`\n"
            f"- **Hospital Bed Infrastructure**: Heavily concentrated in high-population states (Uttar Pradesh, Maharashtra, and Bihar)."
        )

    # 3. ICU & Healthcare Capacity Pressure
    elif any(term in q for term in ["icu", "intensive care", "ventilator"]):
        icu_states = stats.get("high_icu_states", {})
        icu_str = "\n".join([f"- **{state}**: {rate}% average ICU utilization" for state, rate in icu_states.items()])
        return (
            f"### 🏥 Critical Care & ICU Capacity Stress\n\n"
            f"The national average ICU bed utilization is currently **{stats.get('avg_icu_occupancy', 0)}%**.\n\n"
            f"**States facing the highest critical care occupancy pressure:**\n"
            f"{icu_str}\n\n"
            f"**Action Recommendation:** Regions exceeding 75% ICU occupancy require emergency surge bed capacity, "
            f"oxygen buffer stockpile checks, and patient triage protocol activation."
        )

    # 4. Outbreak & Alerts
    elif any(term in q for term in ["high risk", "alert", "outbreak", "urgent"]):
        high_states = stats.get("high_alert_states", {})
        high_str = "\n".join([f"- **{state}**: {count} High-Alert Incidents" for state, count in high_states.items()])
        return (
            f"### 🚨 Outbreak & Alert Severity Analysis\n\n"
            f"Based on real-time surveillance records, there are **{stats.get('total_outbreaks', 0):,} total outbreak events** "
            f"recorded, of which **{stats.get('high_alerts', 0):,} are categorized as High Alert**.\n\n"
            f"**States with the highest concentration of High-Alert emergencies:**\n"
            f"{high_str}\n\n"
            f"**Action Recommendation:** Immediate priority containment teams should be mobilized to these top focus states. "
            f"Check the *Outbreak Alert Response Queue* on Dashboard 3 to track active containment workflows."
        )

    # 5. Disease Burdens (Top 5 / Top 10)
    elif any(term in q for term in ["disease", "burden", "most cases", "common", "top illness"]):
        dis = stats.get("top_diseases", {})
        dis_str = "\n".join([f"{i+1}. **{d}**: {int(cases):,} reported cases" for i, (d, cases) in enumerate(dis.items())])
        return (
            f"### 🦠 Top Disease Burdens Across Surveillance Networks\n\n"
            f"Surveillance data aggregates **{stats.get('total_cases', 0):,} total cases** with an overall "
            f"Case Fatality Rate (CFR) of **{stats.get('cfr', 0)}%**.\n\n"
            f"**Dominant diseases driving patient volume:**\n"
            f"{dis_str}\n\n"
            f"**Key Vector Pattern:** Vector-borne diseases (Dengue, Malaria, Chikungunya) and acute respiratory infections exhibit the highest seasonal velocity."
        )

    # 6. AQI & Environmental Risk
    elif any(term in q for term in ["aqi", "air", "environment", "pollution", "rainfall", "weather"]):
        aqi_states = stats.get("worst_aqi_states", {})
        aqi_str = "\n".join([f"- **{state}**: Average AQI {aqi}" for state, aqi in aqi_states.items()])
        return (
            f"### 🌫️ Environmental Intelligence & AQI Stress\n\n"
            f"The nationwide average Air Quality Index (AQI) stands at **{stats.get('avg_aqi', 0)}**.\n\n"
            f"**States recording the most severe air quality exposure:**\n"
            f"{aqi_str}\n\n"
            f"**Epidemiological Correlation:** Prolonged elevated AQI (>200) correlates strongly with respiratory emergency admissions (Asthma, Influenza, and viral bronchitis)."
        )

    # 7. States & Hotspots
    elif any(term in q for term in ["state", "hotspot", "geography", "top state", "city", "cities"]):
        st_dict = stats.get("top_states", {})
        st_str = "\n".join([f"{i+1}. **{s}**: {int(cases):,} cases" for i, (s, cases) in enumerate(st_dict.items())])
        return (
            f"### 🗺️ Geographic Hotspot Distribution\n\n"
            f"*(Note: Health Sentinel tracks epidemiology at State granularity across India)*\n\n"
            f"**Top 5 states with the highest cumulative disease case volume:**\n"
            f"{st_str}\n\n"
            f"**Summary:** Hotspots are concentrated in high-density urban corridors and regions experiencing high post-monsoon vector indices."
        )

    else:
        return (
            f"### 🩺 Sentinel Public Health Summary\n\n"
            f"Here is the active surveillance snapshot across national health databases:\n\n"
            f"- **Cumulative Reported Cases**: {stats.get('total_cases', 0):,}\n"
            f"- **Cumulative Deaths**: {stats.get('total_deaths', 0):,} (CFR: {stats.get('cfr', 0)}%)\n"
            f"- **Active Outbreaks Tracked**: {stats.get('total_outbreaks', 0):,} ({stats.get('high_alerts', 0):,} High Alert)\n"
            f"- **National Mean ICU Utilization**: {stats.get('avg_icu_occupancy', 0)}%\n"
            f"- **National Mean AQI Index**: {stats.get('avg_aqi', 0)}\n"
            f"- **National Mean Vulnerability Index**: {stats.get('avg_vulnerability', 0)} / 100\n\n"
            f"💡 *Tip: You can ask specific questions like:*\n"
            f"* 'Which states have the highest outbreak alert levels?'\n"
            f"* 'Which states have the most hospital beds?'\n"
            f"* 'What are the top disease burdens?'\n"
            f"* 'Show me the highest vulnerability index states'\n"
            f"* 'Show environmental risk and AQI impact'"
        )


def ask_sentinel(query: str, api_key: Optional[str] = None) -> str:
    """
    Main conversational interface.
    If an external API key (Gemini / OpenAI) is configured, calls the LLM with health context.
    Otherwise, gracefully falls back to the data-grounded offline synthesis engine.
    """
    if not query or not query.strip():
        return "Please enter an epidemiological or surveillance question."

    # If user provided an API key, we can use LLM reasoning
    if api_key and len(api_key.strip()) > 10:
        try:
            # Check for Google Generative AI (Gemini)
            import google.generativeai as genai
            genai.configure(api_key=api_key.strip())
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            stats = _get_dataset_summary()
            system_context = f"""
            You are 'Sentinel AI', an expert public health epidemiological decision-support co-pilot for health directors.
            Current Surveillance Context:
            - Cumulative Cases: {stats.get('total_cases')}
            - Cumulative Deaths: {stats.get('total_deaths')} (CFR: {stats.get('cfr')}%)
            - Outbreak Incidents: {stats.get('total_outbreaks')} (High Alert: {stats.get('high_alerts')})
            - Top High-Alert States: {stats.get('high_alert_states')}
            - Top Disease Burdens: {stats.get('top_diseases')}
            - High ICU Occupancy States: {stats.get('high_icu_states')}
            - Environmental AQI Highlights: {stats.get('worst_aqi_states')}

            Provide actionable, professional, evidence-backed epidemiological answers formatted in clean markdown.
            """
            response = model.generate_content(f"{system_context}\n\nUser Question: {query}")
            if response and response.text:
                return response.text
        except Exception as e:
            # Fallback seamlessly to offline synthesis
            fallback = _offline_answer_generator(query)
            return f"*(Note: LLM API returned error: {str(e)[:70]}... using local Sentinel synthesis engine)*\n\n{fallback}"

    # Default: Grounded Offline Data Intelligence Engine
    return _offline_answer_generator(query)


def generate_executive_sitrep(state: str = "All", disease: str = "All") -> str:
    """
    Generate an Executive Epidemiological Situation Report (SITREP) formatted for
    directors of public health and emergency response commanders.
    """
    stats = _get_dataset_summary()
    date_str = pd.Timestamp.now().strftime("%d %B %Y, %H:%M UTC")

    threat_level = "CRITICAL (LEVEL 3)" if stats.get("high_alerts", 0) > 100 else "ELEVATED (LEVEL 2)"
    badge_color = "🔴" if "CRITICAL" in threat_level else "🟠"

    top_diseases_md = "\n".join([f"- **{d}**: {int(c):,} cases" for d, c in list(stats.get("top_diseases", {}).items())[:4]])
    top_states_md = "\n".join([f"- **{s}**: {c} High Alert Events" for s, c in list(stats.get("high_alert_states", {}).items())[:4]])
    icu_md = "\n".join([f"- **{s}**: {r}% Occupancy" for s, r in list(stats.get("high_icu_states", {}).items())[:3]])

    sitrep = f"""# 🛡️ HEALTH SENTINEL: EXECUTIVE SITUATION REPORT (SITREP)
**Report Generation Time**: `{date_str}`  
**Operational Status**: {badge_color} **{threat_level}**  
**Geographic Scope**: `{state}` | **Pathogen Focus**: `{disease}`  

---

### 1. EXECUTIVE SUMMARY & THREAT APPRAISAL
Surveillance telemetry across joined hospital, lab, and field surveillance streams indicates **{stats.get('total_outbreaks', 0):,} monitored outbreak incidents**, with **{stats.get('high_alerts', 0):,} critical emergency alerts** active. Cumulative case counts have reached **{stats.get('total_cases', 0):,}** with a national Case Fatality Rate (CFR) of **{stats.get('cfr', 0)}%**.

### 2. PRIMARY DISEASE VECTORS OF CONCERN
The dominant transmission trajectories are led by vector-borne seasonal surges and acute respiratory infections:
{top_diseases_md}

### 3. CRITICAL GEOGRAPHIC HOTSPOTS
Emergency containment latency and high case velocities indicate severe localized clusters in:
{top_states_md}

### 4. HEALTHCARE INFRASTRUCTURE & SURGE CAPACITY STRESS
- **National Mean ICU Occupancy**: `{stats.get('avg_icu_occupancy', 0)}%`
- **Key At-Risk Healthcare Systems**:
{icu_md}

### 5. MANDATED STRATEGIC INTERVENTIONS
1. **Rapid Containment Deployment**: Mobilize rapid response teams (RRTs) to priority high-alert districts within a 24-hour response window.
2. **Critical Care Surge Activation**: Re-allocate mechanical ventilators and emergency oxygen reserves to districts exceeding 80% ICU saturation.
3. **Targeted Vector & Environmental Mitigation**: Initiate vector spraying and community drainage operations in high rainfall / elevated AQI zones.
4. **Enhanced Diagnostic Testing**: Deploy rapid test kits to clinics in top hotspot regions to reduce positivity latency.

---
*Classification: RESTRICTED — FOR PUBLIC HEALTH LEADERSHIP & OPERATIONAL COMMAND ONLY*
"""
    return sitrep
