"""
Tab 1: Dashboard
Overview and quick access to main features
"""

import streamlit as st
import pandas as pd
from datetime import datetime


def render(settings: dict) -> None:
    """
    Render the Dashboard tab.
    
    Args:
        settings: Dict of sidebar settings
    """
    st.markdown("### 🏠 Dashboard")
    st.markdown("Welcome to the RAG PDF Field Extractor. Get started by uploading a PDF and building an index.")
    
    # Status Overview
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "client" in st.session_state:
            st.metric("🔑 API Status", "Connected", delta="✓")
        else:
            st.metric("🔑 API Status", "Not Connected", delta="⚠")
    
    with col2:
        if "index" in st.session_state:
            st.metric("📦 Index Status", "Built", delta="✓")
        else:
            st.metric("📦 Index Status", "Not Built", delta="⚠")
    
    with col3:
        if "chunks" in st.session_state:
            st.metric("🧩 Total Chunks", len(st.session_state.chunks))
        else:
            st.metric("🧩 Total Chunks", "N/A")
    
    with col4:
        if "chat" in st.session_state:
            st.metric("💬 Extractions", len(st.session_state.chat))
        else:
            st.metric("💬 Extractions", 0)
    
    # Quick Actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📄 Document Extraction
        - Upload a PDF document
        - Build the FAISS index
        - Extract fields using natural language
        
        *Go to the **Document Extraction** tab →*
        """)
    
    with col2:
        st.markdown("""
        #### ⚖️ Flow Comparison
        - Compare Zero-shot vs RAG extraction
        - Measure accuracy and hallucination
        - Export detailed reports
        
        *Go to the **Flow Comparison** tab →*
        """)
    
    with col3:
        st.markdown("""
        #### 📊 Benchmarking
        - Test multiple fields at once
        - Compare chunking algorithms
        - Analyze performance metrics
        
        *Go to the **Benchmarking** tab →*
        """)
    
    # Current Configuration
    if "index" in st.session_state:
        st.markdown("---")
        st.markdown("### ⚙️ Current Configuration")
        
        config_data = {
            "Setting": ["Chunking Algorithm", "Chunk Mode", "Chunk Size", "Overlap", "Top-K"],
            "Value": [
                settings.get("chunking_algorithm", "N/A"),
                settings.get("chunk_mode", "N/A"),
                settings.get("chunk_size", "N/A"),
                settings.get("overlap", "N/A"),
                settings.get("top_k", "N/A")
            ]
        }
        st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)
    
    # Recent Activity
    if "chat" in st.session_state and st.session_state.chat:
        st.markdown("---")
        st.markdown("### 📋 Recent Extractions")
        
        recent = st.session_state.chat[-5:]  # Last 5 extractions
        for item in reversed(recent):
            # Support both old format (q, a, c) and new format (q, a, c, r)
            if len(item) == 4:
                query, value, conf, reason = item
            else:
                query, value, conf = item
                reason = ""
            with st.expander(f"🔍 {query[:50]}..."):
                st.markdown(f"**Value:** `{value}`")
                st.markdown(f"**Confidence:** {conf}%")
                if reason:
                    st.caption(f"💡 *{reason}*")
