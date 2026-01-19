"""
Tab 2: Document Analysis
Automatic field identification and extraction
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime

from analysis.document_analyzer import analyze_document
from core.retrieval import retrieve
from core.extraction import extract_field_value
from ui.components import render_confidence_table, render_metrics_cards


def render(settings: dict) -> None:
    """
    Render the Document Analysis tab.
    
    Args:
        settings: Dict of sidebar settings
    """
    st.markdown("### 🔍 Automatic Document Analysis")
    st.markdown("Automatically identify and extract all key fields from your document using AI.")
    
    if "index" not in st.session_state:
        st.warning("⚠️ Please build an index first from the sidebar.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎯 Analysis Controls")
        st.info("Click the button below to automatically analyze the document and extract all key fields.")
        
        analyze_btn = st.button("🔍 Analyze Document", type="primary", use_container_width=True)
        
        if "analysis_results" in st.session_state and st.session_state.analysis_results:
            st.success(f"✅ Analysis complete! Found {len(st.session_state.analysis_results)} fields")
    
    with col2:
        st.markdown("#### ℹ️ How It Works")
        st.markdown("""
        1. **AI identifies key fields** in your document
        2. **Extracts values** for each field using RAG
        3. **Calculates confidence** scores for each extraction
        4. **Displays results** in an organized table
        """)
    
    # Run Analysis
    if analyze_btn:
        _run_analysis(settings)
    
    # Display Results
    if "analysis_results" in st.session_state and st.session_state.analysis_results:
        _display_results(st.session_state.analysis_results)


def _run_analysis(settings: dict) -> None:
    """Run the document analysis workflow."""
    st.markdown("---")
    st.info("🤖 Step 1: Analyzing document to identify key fields...")
    
    try:
        fields = analyze_document(
            st.session_state.client,
            st.session_state.pdf_text
        )
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        return
    
    if not fields:
        st.error("❌ Could not identify any fields in the document.")
        return
    
    st.success(f"✅ Identified {len(fields)} key fields")
    
    # Show identified fields
    with st.expander("📋 View Identified Fields"):
        for i, field in enumerate(fields, 1):
            st.markdown(f"{i}. **{field.get('field_name', 'Unknown')}**")
    
    st.info(f"🤖 Step 2: Extracting values for {len(fields)} fields...")
    
    # Extract all fields
    results = _extract_fields(fields, settings)
    
    if results:
        st.session_state.analysis_results = results
        st.success("✅ Analysis complete!")
        st.rerun()


def _extract_fields(fields: list, settings: dict) -> list:
    """Extract values for all identified fields."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    api_delay = settings.get("api_delay", 2.0)
    top_k = settings.get("top_k", 5)
    
    import time
    
    for i, field in enumerate(fields):
        field_name = field.get("field_name", f"Field {i+1}")
        query = field.get("query", "")
        
        status_text.text(f"Extracting {i+1}/{len(fields)}: {field_name}...")
        
        try:
            # Retrieve relevant chunks
            retrieved = retrieve(
                st.session_state.client,
                st.session_state.index,
                st.session_state.chunks,
                query,
                top_k
            )
            context_text = "\n\n".join([r["chunk"] for r in retrieved])
            
            # Extract with confidence
            extracted_value, confidence, reason = extract_field_value(
                st.session_state.client,
                query,
                context_text
            )
            
            results.append({
                "field_name": field_name,
                "value": extracted_value,
                "confidence": round(confidence, 1),
                "confidence_reason": reason,
                "query": query
            })
            
            # Rate limiting
            if i < len(fields) - 1:
                time.sleep(api_delay)
                
        except Exception as e:
            st.warning(f"⚠️ Error extracting {field_name}: {e}")
            results.append({
                "field_name": field_name,
                "value": "ERROR",
                "confidence": 0,
                "confidence_reason": str(e),
                "query": query
            })
            if i < len(fields) - 1:
                time.sleep(api_delay * 2)
        
        progress_bar.progress((i + 1) / len(fields))
    
    progress_bar.empty()
    status_text.empty()
    
    return results


def _display_results(results: list) -> None:
    """Display analysis results."""
    st.markdown("---")
    st.markdown("### 📊 Extraction Results")
    
    df = pd.DataFrame(results)
    
    # Summary metrics
    metrics = [
        {"label": "Total Fields", "value": len(results)},
        {"label": "Extracted", "value": len([r for r in results if r["value"] not in ["N/A", "ERROR"]])},
        {"label": "Avg Confidence", "value": f"{df[df['value'] != 'ERROR']['confidence'].mean():.1f}%"},
        {"label": "High Confidence (≥70%)", "value": len(df[df["confidence"] >= 70])}
    ]
    render_metrics_cards(metrics)
    
    st.markdown("---")
    
    # Results table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 Detailed Results")
        display_df = df[["field_name", "value", "confidence"]].copy()
        display_df.columns = ["Field Name", "Value", "Confidence (%)"]
        render_confidence_table(display_df, "Value", "Confidence (%)")
    
    with col2:
        st.markdown("#### 📥 Export Options")
        
        # CSV Export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"document_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # JSON Export
        json_data = json.dumps(results, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"document_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("#### 📈 Confidence Distribution")
        
        conf_bins = pd.cut(df["confidence"], bins=[0, 50, 70, 100], labels=["Low", "Medium", "High"])
        conf_counts = conf_bins.value_counts()
        st.bar_chart(conf_counts)
    
    # Detailed view with queries and confidence reasoning
    st.markdown("---")
    st.markdown("#### 🔍 Detailed Field Information")
    
    for i, result in enumerate(results, 1):
        value_preview = result['value'][:50] + ('...' if len(result['value']) > 50 else '')
        with st.expander(f"{i}. {result['field_name']} - {value_preview}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Field Name:** {result['field_name']}")
                st.markdown(f"**Extracted Value:** `{result['value']}`")
                st.markdown(f"**Confidence:** {result['confidence']}%")
            with col2:
                st.markdown(f"**Query Used:** {result.get('query', 'N/A')}")
                if "confidence_reason" in result:
                    st.info(f"**Why this confidence?** {result['confidence_reason']}")
