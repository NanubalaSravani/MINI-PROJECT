"""
6_Ask_Sentinel_AI.py
--------------------
Interactive Generative AI Public Health Copilot and Situation Report Station.

Features:
- Natural Language Question Answering over surveillance, outbreak, lab capacity,
  and environmental datasets.
- Interactive Query Suggestions (Chips) for immediate decision-support.
- Automated Executive Epidemiological Situation Report (SITREP) generator
  with single-click briefing download.
- Optional Google Gemini / LLM API Key integration for advanced open-ended reasoning.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.styling import inject_css, page_header, kpi_card, section_title
from src.ai_copilot import ask_sentinel, generate_executive_sitrep

# ==========================================
# 1. SETUP & STYLING
# ==========================================
inject_css()

PRIMARY_NAVY = "#17324D"
TEAL = "#0F6B78"
RED = "#C43D3D"
AMBER = "#C98A00"
GREEN = "#16855B"

# Page Header
page_header(
    title="Ask Sentinel (AI Copilot)",
    subtitle="Intelligent epidemiology co-pilot & automated executive situation briefing system",
)

# ==========================================
# 2. SIDEBAR CONFIGURATION (OPTIONAL LLM KEY)
# ==========================================
with st.sidebar:
    st.markdown("### 🤖 Sentinel AI Engine")
    st.caption("Running with local data synthesis engine (Offline Grounded).")
    
    with st.expander("🔑 Optional: Configure LLM API Key", expanded=False):
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Optional: Enter a Google Gemini API key to enable open-ended epidemiological conversational reasoning. If omitted, Sentinel's built-in offline data synthesis engine is used."
        )
        if api_key_input:
            st.session_state["sentinel_api_key"] = api_key_input
            st.success("API key configured!")

# Retrieve key from session state if set
current_api_key = st.session_state.get("sentinel_api_key", None)

# ==========================================
# 3. TABS: CHAT COPILOT | EXECUTIVE SITREP
# ==========================================
tab_chat, tab_sitrep = st.tabs([
    "💬 Ask Sentinel Chat Copilot",
    "📋 Automated Executive Situation Report (SITREP)"
])

# ------------------------------------------
# TAB 1: INTERACTIVE CHAT COPILOT
# ------------------------------------------
with tab_chat:
    st.markdown('<div class="section-title">Epidemiological Intelligence Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Ask questions about disease trends, alert levels, ICU capacities, or environmental correlations across all surveillance datasets.</div>', unsafe_allow_html=True)

    # Initialize chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am Sentinel AI**, your epidemiological decision-support co-pilot.\n\n"
                    "I continuously analyze joined records across disease surveillance, outbreak alerts, "
                    "laboratory capacity, and environmental indices. Ask me anything about current outbreak hotspots, "
                    "hospital capacity, or vector surges!"
                )
            }
        ]

    # Quick Suggestion Chips
    st.markdown("<p style='font-size: 13px; font-weight: 600; margin-bottom: 6px;'>Suggested Inquiries:</p>", unsafe_allow_html=True)
    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
    
    prompt_to_submit = None
    with chip_col1:
        if st.button("🚨 Outbreak Alerts & Hotspots", use_container_width=True):
            prompt_to_submit = "Which states have the highest outbreak alert levels?"
    with chip_col2:
        if st.button("🏥 ICU & Bed Occupancy", use_container_width=True):
            prompt_to_submit = "Show me healthcare capacity and ICU bed occupancy stress."
    with chip_col3:
        if st.button("🦠 Top Disease Burdens", use_container_width=True):
            prompt_to_submit = "What are the top 5 diseases driving patient volume?"
    with chip_col4:
        if st.button("🌫️ AQI & Environmental Risk", use_container_width=True):
            prompt_to_submit = "Which states have the worst AQI pollution and environmental stress?"

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Display Chat Conversation
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    # User Input Field
    user_input = st.chat_input("Ask a question about disease outbreaks, hospital readiness, or trends...")

    active_prompt = prompt_to_submit or user_input

    if active_prompt:
        # Append User Message
        st.session_state.chat_messages.append({"role": "user", "content": active_prompt})
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(active_prompt)

        # Generate Assistant Response
        with chat_container:
            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("Synthesizing epidemiological surveillance data..."):
                    bot_reply = ask_sentinel(active_prompt, api_key=current_api_key)
                    st.markdown(bot_reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": bot_reply})
        st.rerun()

    # Clear Chat Button
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Conversation History"):
        st.session_state.chat_messages = []
        st.rerun()

# ------------------------------------------
# TAB 2: EXECUTIVE SITUATION REPORT (SITREP)
# ------------------------------------------
with tab_sitrep:
    st.markdown('<div class="section-title">Automated Situation Report (SITREP) Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Generate an automated, structured briefing summarizing the real-time threat posture, hospital bottlenecks, and mandated interventions for leadership.</div>', unsafe_allow_html=True)

    gen_col1, gen_col2 = st.columns([3, 1])
    with gen_col1:
        st.info("Click the button below to synthesize live surveillance streams into an official Epidemiological Situation Report.")
    with gen_col2:
        generate_btn = st.button("⚡ Generate Live SITREP", use_container_width=True, type="primary")

    if generate_btn or "cached_sitrep" in st.session_state:
        if generate_btn or "cached_sitrep" not in st.session_state:
            with st.spinner("Compiling cross-domain metrics and executive briefing..."):
                sitrep_text = generate_executive_sitrep()
                st.session_state["cached_sitrep"] = sitrep_text
        else:
            sitrep_text = st.session_state["cached_sitrep"]

        with st.container(border=True):
            st.markdown(sitrep_text)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Executive SITREP (.md)",
            data=sitrep_text,
            file_name=f"HEALTH_SENTINEL_SITREP_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=False,
        )
