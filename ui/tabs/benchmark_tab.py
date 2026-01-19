"""
Tab: Benchmarking
Multi-field benchmarking with chunking strategy comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

from core.retrieval import retrieve
from core.extraction import extract_field_value_simple
from core.chunking import chunk_text_sliding_window, chunk_text_recursive
from core.embeddings import embed_documents
from core.faiss_index import build_faiss_index
from ui.components import render_metrics_cards


def render(settings: dict) -> None:
    """
    Render the Benchmarking tab with multi-field and dual-algorithm support.
    
    Args:
        settings: Dict of sidebar settings
    """
    st.markdown("### 📊 Multi-Field Benchmarking")
    st.markdown("Test extraction performance with multiple fields and compare chunking algorithms.")
    
    if "index" not in st.session_state:
        st.warning("⚠️ Please build an index first from the sidebar.")
        return
    
    # Multi-Field Input Section
    st.markdown("#### 📝 Fields to Benchmark")
    
    # Initialize session state for fields
    if "benchmark_fields" not in st.session_state:
        st.session_state.benchmark_fields = [{"query": "", "expected": ""}]
    
    # Add field button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Add Field", use_container_width=True):
            st.session_state.benchmark_fields.append({"query": "", "expected": ""})
            st.rerun()
    
    # Display field inputs
    fields_to_remove = []
    for i, field in enumerate(st.session_state.benchmark_fields):
        col1, col2, col3 = st.columns([3, 2, 0.5])
        
        with col1:
            st.session_state.benchmark_fields[i]["query"] = st.text_input(
                f"Query {i+1}",
                value=field.get("query", ""),
                placeholder="e.g., What is the contract date?",
                key=f"bm_query_{i}"
            )
        
        with col2:
            st.session_state.benchmark_fields[i]["expected"] = st.text_input(
                f"Expected {i+1}",
                value=field.get("expected", ""),
                placeholder="e.g., 2024-01-15 (optional)",
                key=f"bm_expected_{i}"
            )
        
        with col3:
            st.markdown("####")  # Spacing
            if len(st.session_state.benchmark_fields) > 1:
                if st.button("🗑️", key=f"bm_remove_{i}"):
                    fields_to_remove.append(i)
    
    # Remove marked fields
    for i in reversed(fields_to_remove):
        st.session_state.benchmark_fields.pop(i)
        st.rerun()
    
    st.markdown("---")
    
    # Algorithm Comparison Option
    st.markdown("#### 🔧 Chunking Strategy Comparison")
    
    compare_algorithms = st.checkbox("Compare two chunking algorithms", value=False)
    
    if compare_algorithms:
        col1, col2 = st.columns(2)
        with col1:
            algo1 = st.selectbox(
                "Algorithm 1",
                ["Sliding Window", "Recursive"],
                index=0,
                key="algo1"
            )
        with col2:
            algo2 = st.selectbox(
                "Algorithm 2",
                ["Sliding Window", "Recursive"],
                index=1,
                key="algo2"
            )
    else:
        algo1 = settings.get("chunking_algorithm", "Sliding Window")
        algo2 = None
    
    st.markdown("---")
    
    # Run Button
    run_btn = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)
    
    if run_btn:
        # Filter valid fields
        valid_fields = [f for f in st.session_state.benchmark_fields if f["query"].strip()]
        
        if not valid_fields:
            st.error("Please enter at least one query to benchmark.")
            return
        
        if compare_algorithms:
            _run_algorithm_comparison(valid_fields, algo1, algo2, settings)
        else:
            _run_single_benchmark(valid_fields, settings)


def _run_single_benchmark(fields: list, settings: dict):
    """Run benchmark with current algorithm."""
    api_delay = settings.get("api_delay", 2.0)
    top_k = settings.get("top_k", 5)
    num_runs = settings.get("num_benchmark_runs", 1)
    
    st.markdown("---")
    st.info(f"🔄 Running {num_runs} iteration(s) for {len(fields)} field(s)...")
    
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(fields) * num_runs
    current_step = 0
    
    for run in range(num_runs):
        for i, field in enumerate(fields):
            query = field["query"]
            expected = field.get("expected", "")
            
            status_text.text(f"Run {run+1}/{num_runs} - Field {i+1}/{len(fields)}: {query[:40]}...")
            start_time = time.time()
            
            try:
                retrieved = retrieve(
                    st.session_state.client,
                    st.session_state.index,
                    st.session_state.chunks,
                    query,
                    top_k
                )
                
                context_text = "\n\n".join([r["chunk"] for r in retrieved])
                extracted_value, match_confidence, match_reason = extract_field_value_simple(
                    st.session_state.client,
                    query,
                    context_text
                )
                
                elapsed_time = time.time() - start_time
                
                is_correct = expected and extracted_value.strip().lower() == expected.strip().lower()
                
                all_results.append({
                    "run": run + 1,
                    "query": query[:40],
                    "expected": expected or "N/A",
                    "extracted": extracted_value,
                    "match": "✓" if is_correct else ("✗" if expected else "—"),
                    "confidence": round(match_confidence, 1),
                    "reason": match_reason[:30],
                    "time_ms": round(elapsed_time * 1000, 2)
                })
                
                if current_step < total_steps - 1:
                    time.sleep(api_delay)
                    
            except Exception as e:
                all_results.append({
                    "run": run + 1,
                    "query": query[:40],
                    "expected": expected or "N/A",
                    "extracted": f"ERROR: {str(e)[:30]}",
                    "match": "✗",
                    "confidence": 0,
                    "time_ms": 0
                })
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)
    
    progress_bar.empty()
    status_text.empty()
    
    _display_results(all_results, fields)


def _run_algorithm_comparison(fields: list, algo1: str, algo2: str, settings: dict):
    """Compare two chunking algorithms."""
    api_delay = settings.get("api_delay", 2.0)
    top_k = settings.get("top_k", 5)
    chunk_mode = settings.get("chunk_mode", "token")
    chunk_size = settings.get("chunk_size", 200)
    overlap = settings.get("overlap", 20)
    
    st.markdown("---")
    st.info(f"🔄 Comparing {algo1} vs {algo2} on {len(fields)} field(s)...")
    
    comparison_results = {}
    
    for algo in [algo1, algo2]:
        st.markdown(f"##### Testing {algo}...")
        
        # Build index for this algorithm
        if algo == "Sliding Window":
            chunks = chunk_text_sliding_window(
                st.session_state.pdf_text,
                chunk_mode,
                chunk_size,
                overlap
            )
        else:
            chunks = chunk_text_recursive(
                st.session_state.pdf_text,
                chunk_mode,
                chunk_size,
                overlap
            )
        
        with st.spinner(f"Building {algo} index..."):
            try:
                embeddings = embed_documents(st.session_state.client, chunks)
                temp_index = build_faiss_index(embeddings)
            except Exception as e:
                st.error(f"Failed to build {algo} index: {e}")
                continue
        
        # Run extractions
        results = []
        progress_bar = st.progress(0)
        
        for i, field in enumerate(fields):
            query = field["query"]
            expected = field.get("expected", "")
            
            start_time = time.time()
            
            try:
                retrieved = retrieve(
                    st.session_state.client,
                    temp_index,
                    chunks,
                    query,
                    top_k
                )
                
                context_text = "\n\n".join([r["chunk"] for r in retrieved])
                extracted_value, match_confidence, match_reason = extract_field_value_simple(
                    st.session_state.client,
                    query,
                    context_text
                )
                
                elapsed_time = time.time() - start_time
                
                is_correct = expected and extracted_value.strip().lower() == expected.strip().lower()
                
                results.append({
                    "query": query[:40],
                    "expected": expected or "N/A",
                    "extracted": extracted_value,
                    "correct": is_correct if expected else None,
                    "confidence": round(match_confidence, 1),
                    "reason": match_reason[:30],
                    "time_ms": round(elapsed_time * 1000, 2)
                })
                
                if i < len(fields) - 1:
                    time.sleep(api_delay)
                    
            except Exception as e:
                results.append({
                    "query": query[:40],
                    "expected": expected or "N/A",
                    "extracted": f"ERROR",
                    "correct": False,
                    "confidence": 0,
                    "time_ms": 0
                })
            
            progress_bar.progress((i + 1) / len(fields))
        
        progress_bar.empty()
        
        comparison_results[algo] = {
            "results": results,
            "num_chunks": len(chunks),
            "avg_confidence": np.mean([r["confidence"] for r in results]),
            "avg_time_ms": np.mean([r["time_ms"] for r in results]),
            "accuracy": sum(1 for r in results if r["correct"]) / len([r for r in results if r["correct"] is not None]) * 100 if any(r["correct"] is not None for r in results) else None
        }
        
        # Delay between algorithms
        if algo == algo1:
            time.sleep(api_delay * 2)
    
    _display_comparison(comparison_results, algo1, algo2, fields)


def _display_results(results: list, fields: list):
    """Display single-algorithm benchmark results."""
    st.markdown("### 📊 Results")
    
    df = pd.DataFrame(results)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Fields", len(fields))
    with col2:
        st.metric("Avg Confidence", f"{df['confidence'].mean():.1f}%")
    with col3:
        st.metric("Avg Time", f"{df['time_ms'].mean():.0f} ms")
    with col4:
        correct_count = len(df[df['match'] == '✓'])
        total_with_expected = len(df[df['match'] != '—'])
        if total_with_expected > 0:
            st.metric("Accuracy", f"{correct_count}/{total_with_expected}")
        else:
            st.metric("Accuracy", "N/A")
    
    st.markdown("---")
    st.dataframe(df, use_container_width=True, height=400)
    
    # Export
    st.download_button(
        "📥 Download Results (CSV)",
        data=df.to_csv(index=False),
        file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


def _display_comparison(results: dict, algo1: str, algo2: str, fields: list):
    """Display algorithm comparison results."""
    st.markdown("### 📊 Algorithm Comparison Results")
    
    # Summary table
    summary_data = []
    for algo, data in results.items():
        summary_data.append({
            "Algorithm": algo,
            "Chunks": data["num_chunks"],
            "Avg Confidence": f"{data['avg_confidence']:.1f}%",
            "Avg Time (ms)": f"{data['avg_time_ms']:.0f}",
            "Accuracy": f"{data['accuracy']:.1f}%" if data['accuracy'] is not None else "N/A"
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    # Winner determination
    st.markdown("---")
    
    score1 = results[algo1]["avg_confidence"] - (results[algo1]["avg_time_ms"] / 100)
    score2 = results[algo2]["avg_confidence"] - (results[algo2]["avg_time_ms"] / 100)
    
    if score1 > score2:
        st.success(f"🏆 **Winner: {algo1}** (Score: {score1:.1f} vs {score2:.1f})")
    elif score2 > score1:
        st.success(f"🏆 **Winner: {algo2}** (Score: {score2:.1f} vs {score1:.1f})")
    else:
        st.info("🤝 **Tie** - Both algorithms performed equally")
    
    # Detailed results
    st.markdown("---")
    st.markdown("#### 📋 Detailed Results")
    
    tab1, tab2 = st.tabs([algo1, algo2])
    
    with tab1:
        df1 = pd.DataFrame(results[algo1]["results"])
        st.dataframe(df1, use_container_width=True)
    
    with tab2:
        df2 = pd.DataFrame(results[algo2]["results"])
        st.dataframe(df2, use_container_width=True)
    
    # Charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Confidence Comparison**")
        chart_data = pd.DataFrame({
            algo1: [r["confidence"] for r in results[algo1]["results"]],
            algo2: [r["confidence"] for r in results[algo2]["results"]]
        })
        st.bar_chart(chart_data)
    
    with col2:
        st.markdown("**Time Comparison**")
        time_data = pd.DataFrame({
            algo1: [r["time_ms"] for r in results[algo1]["results"]],
            algo2: [r["time_ms"] for r in results[algo2]["results"]]
        })
        st.bar_chart(time_data)
    
    # Export
    export_data = {
        "comparison": summary_data,
        algo1: results[algo1],
        algo2: results[algo2],
        "winner": algo1 if score1 > score2 else algo2 if score2 > score1 else "Tie",
        "timestamp": datetime.now().isoformat()
    }
    
    st.download_button(
        "📥 Download Full Report (JSON)",
        data=json.dumps(export_data, indent=2, default=str),
        file_name=f"algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )
