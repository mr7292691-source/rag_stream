"""
UI - Components Module
Reusable UI components for the application
"""

import streamlit as st
import pandas as pd
import base64
from html import escape
from typing import List, Dict, Optional, Callable

from config import PDF_EMBED_SIZE_THRESHOLD_MB
from .styles import PDF_PREVIEW_STYLE, PDF_IFRAME_STYLE, get_confidence_color


def render_pdf_preview(
    pdf_bytes: bytes, 
    pdf_text: str,
    container_height: int = 600
) -> None:
    """
    Render PDF preview with fallback options for large files.
    
    Args:
        pdf_bytes: Raw PDF bytes
        pdf_text: Extracted text content
        container_height: Height of the preview container
    """
    pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
    
    if pdf_size_mb > PDF_EMBED_SIZE_THRESHOLD_MB:
        st.warning(f"⚠️ PDF is large ({pdf_size_mb:.1f} MB). Preview disabled to prevent browser issues.")
        st.info("💡 Use the download button below to view the PDF externally.")
        
        st.download_button(
            label="📥 Download Original PDF",
            data=pdf_bytes,
            file_name="document.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        if st.checkbox("Show Text Preview", value=False):
            st.markdown(
                PDF_PREVIEW_STYLE.format(
                    content=escape(pdf_text[:5000]) + 
                    ('...' if len(pdf_text) > 5000 else '')
                ),
                unsafe_allow_html=True
            )
    else:
        try:
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            st.markdown(
                PDF_IFRAME_STYLE.format(pdf_base64=base64_pdf),
                unsafe_allow_html=True
            )
            st.caption(f"📊 PDF Size: {pdf_size_mb:.2f} MB")
        except Exception as e:
            st.error(f"❌ Error displaying PDF: {e}")
            st.download_button(
                label="📥 Download Original PDF",
                data=pdf_bytes,
                file_name="document.pdf",
                mime="application/pdf",
                use_container_width=True
            )


def render_confidence_table(
    df: pd.DataFrame,
    value_col: str = "value",
    confidence_col: str = "confidence",
    height: int = 400
) -> None:
    """
    Render a styled dataframe with confidence color coding.
    
    Args:
        df: DataFrame with extraction results
        value_col: Column name for values
        confidence_col: Column name for confidence scores
        height: Table height in pixels
    """
    def highlight_confidence(row):
        conf = row[confidence_col]
        color = get_confidence_color(conf)
        return [f'background-color: {color}'] * len(row)
    
    styled_df = df.style.apply(highlight_confidence, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=height)
    
    st.caption("🟢 Green: High confidence (≥70%) | 🟡 Yellow: Medium (50-70%) | 🔴 Red: Low (<50%)")


def render_metrics_cards(metrics: List[Dict]) -> None:
    """
    Render a row of metric cards.
    
    Args:
        metrics: List of dicts with 'label', 'value', and optional 'delta'
    """
    cols = st.columns(len(metrics))
    
    for col, metric in zip(cols, metrics):
        with col:
            if "delta" in metric:
                st.metric(
                    label=metric["label"],
                    value=metric["value"],
                    delta=metric.get("delta")
                )
            else:
                st.metric(
                    label=metric["label"],
                    value=metric["value"]
                )


def render_progress_tracker(
    current: int,
    total: int,
    status_text: str = ""
) -> tuple:
    """
    Render a progress bar with status text.
    
    Args:
        current: Current progress value
        total: Total value
        status_text: Optional status message
        
    Returns:
        Tuple of (progress_bar, status_placeholder) for updates
    """
    progress_bar = st.progress(current / total if total > 0 else 0)
    status = st.empty()
    
    if status_text:
        status.text(status_text)
    
    return progress_bar, status


def render_comparison_table(
    field_data: List[Dict],
    height: int = 400
) -> pd.DataFrame:
    """
    Render a comparison table for zero-shot vs RAG results.
    
    Args:
        field_data: List of field comparison dicts
        height: Table height
        
    Returns:
        DataFrame for export
    """
    from .styles import get_match_icon
    
    formatted_data = []
    for field in field_data:
        zs_icon = get_match_icon(field.get("zero_shot_match", "mismatch"))
        rag_icon = get_match_icon(field.get("rag_match", "mismatch"))
        
        formatted_data.append({
            "Field": field["field_name"],
            "Master Value": str(field["master_value"])[:30],
            "Zero-Shot": f"{zs_icon} {str(field.get('zero_shot_value', 'N/A'))[:25]}",
            "ZS Conf": f"{field.get('zero_shot_confidence', 0)}%",
            "RAG": f"{rag_icon} {str(field.get('rag_value', 'N/A'))[:25]}",
            "RAG Conf": f"{field.get('rag_confidence', 0)}%"
        })
    
    df = pd.DataFrame(formatted_data)
    st.dataframe(df, use_container_width=True, height=height)
    
    return df


def render_export_buttons(
    data: Dict,
    prefix: str = "export",
    include_csv: bool = True
) -> None:
    """
    Render export/download buttons for data.
    
    Args:
        data: Dict with 'json' and optionally 'csv' data
        prefix: Filename prefix
        include_csv: Whether to include CSV download option
    """
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    cols = st.columns(2 if include_csv else 1)
    
    with cols[0]:
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(data.get("json", data), indent=2),
            file_name=f"{prefix}_{timestamp}.json",
            mime="application/json",
            use_container_width=True
        )
    
    if include_csv and "csv" in data:
        with cols[1]:
            st.download_button(
                label="📥 Download CSV",
                data=data["csv"],
                file_name=f"{prefix}_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )


def render_chat_message(
    query: str,
    answer: str,
    confidence: float,
    reason: str = ""
) -> None:
    """
    Render a single chat message with confidence reasoning.
    
    Args:
        query: User's query
        answer: Extracted answer
        confidence: Confidence percentage
        reason: Reason for the confidence score
    """
    st.markdown(f"**🧑 You:** {query}")
    st.markdown(f"**🤖 Extracted:** `{answer}`")
    st.markdown(f"**📊 Confidence:** `{confidence}%`")
    if reason:
        st.caption(f"💡 *{reason}*")
    st.divider()
