"""
Tab 5: Algorithm Comparison
Compare chunking algorithms side-by-side
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime

from analysis.benchmarking import compare_chunking_algorithms
from ui.components import render_metrics_cards


def render(settings: dict) -> None:
    """
    Render the Algorithm Comparison tab.
    
    Args:
        settings: Dict of sidebar settings
    """
    st.markdown("### ⚖️ Compare Chunking Algorithms")
    st.markdown("Compare Sliding Window vs Recursive chunking algorithms side-by-side.")
    
    if "pdf_text" not in st.session_state:
        st.warning("⚠️ Please build an index first from the sidebar.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        comparison_query = st.text_input(
            "Enter your comparison query",
            placeholder="e.g., What is the total amount?",
            key="comparison_query_tab"
        )
    
    with col2:
        st.markdown("####  ")  # Spacing
        compare_btn = st.button("⚖️ Compare Algorithms", type="primary", use_container_width=True)
    
    if compare_btn and comparison_query:
        _run_comparison(comparison_query, settings)


def _run_comparison(query: str, settings: dict):
    """Run algorithm comparison."""
    num_runs = settings.get("num_benchmark_runs", 5)
    api_delay = settings.get("api_delay", 2.0)
    chunk_mode = settings.get("chunk_mode", "token")
    chunk_size = settings.get("chunk_size", 200)
    overlap = settings.get("overlap", 20)
    top_k = settings.get("top_k", 5)
    
    st.markdown("---")
    st.info(f"🕒 Running {num_runs} iterations per algorithm with {api_delay}s delay")
    
    def progress_callback(current, total, status):
        st.info(status)
    
    comparison = compare_chunking_algorithms(
        query=query,
        document_text=st.session_state.pdf_text,
        client=st.session_state.client,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        num_runs=num_runs,
        delay_seconds=api_delay,
        progress_callback=progress_callback
    )
    
    if comparison and len(comparison) == 2:
        _display_comparison_results(comparison)


def _display_comparison_results(comparison: dict):
    """Display algorithm comparison results."""
    # Summary Comparison
    st.markdown("#### 📊 Summary Comparison")
    
    summary_data = []
    for algo, data in comparison.items():
        summary_data.append({
            "Algorithm": algo,
            "Chunks Created": data["num_chunks"],
            "Avg Confidence": f"{data['avg_confidence']}%",
            "Avg Time (ms)": data["avg_time_ms"],
            "Unique Values": data["consistency"],
            "Most Common Value": data["most_common_value"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    st.markdown("---")
    
    # Winner Analysis
    st.markdown("#### 🏆 Winner Analysis")
    
    sw_score = comparison["Sliding Window"]["avg_confidence"] - (comparison["Sliding Window"]["avg_time_ms"] / 100)
    rec_score = comparison["Recursive"]["avg_confidence"] - (comparison["Recursive"]["avg_time_ms"] / 100)
    
    winner = "Sliding Window" if sw_score > rec_score else "Recursive"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success(f"🏆 **Winner: {winner}**")
        st.caption("Based on confidence-to-speed ratio")
    
    st.markdown("---")
    
    # Detailed Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Sliding Window Results")
        if "Sliding Window" in comparison:
            sw_df = pd.DataFrame(comparison["Sliding Window"]["results"])
            st.dataframe(sw_df, use_container_width=True, height=250)
    
    with col2:
        st.markdown("#### 📊 Recursive Results")
        if "Recursive" in comparison:
            rec_df = pd.DataFrame(comparison["Recursive"]["results"])
            st.dataframe(rec_df, use_container_width=True, height=250)
    
    st.markdown("---")
    
    # Performance Charts
    st.markdown("#### 📈 Performance Comparison Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Confidence Comparison**")
        chart_data = pd.DataFrame({
            "Sliding Window": [r["confidence"] for r in comparison["Sliding Window"]["results"]],
            "Recursive": [r["confidence"] for r in comparison["Recursive"]["results"]]
        })
        st.line_chart(chart_data)
    
    with col2:
        st.markdown("**Time Comparison**")
        time_data = pd.DataFrame({
            "Sliding Window": [r["time_ms"] for r in comparison["Sliding Window"]["results"]],
            "Recursive": [r["time_ms"] for r in comparison["Recursive"]["results"]]
        })
        st.line_chart(time_data)
    
    # Download Report
    st.markdown("---")
    full_report = {
        "summary": summary_data,
        "sliding_window_results": comparison["Sliding Window"]["results"],
        "recursive_results": comparison["Recursive"]["results"],
        "winner": winner,
        "timestamp": datetime.now().isoformat()
    }
    
    st.download_button(
        label="📥 Download Full Comparison Report (JSON)",
        data=json.dumps(full_report, indent=2),
        file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )
