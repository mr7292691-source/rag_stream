"""
UI - Sidebar Module
Renders the sidebar with all configuration options
"""

import streamlit as st

from core.llm import create_client
from config import (
    CHUNKING_ALGORITHMS,
    CHUNKING_MODES,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_BENCHMARK_RUNS,
    MAX_BENCHMARK_RUNS,
    DEFAULT_API_DELAY,
    MIN_API_DELAY,
    MAX_API_DELAY,
)


def render_sidebar() -> dict:
    """
    Render the sidebar configuration panel.
    
    Returns:
        Dict with all sidebar settings
    """
    st.sidebar.header("⚙️ Settings")
    
    # Initialize client from config (API key is in config.py)
    if "client" not in st.session_state:
        try:
            create_client()  # Uses API key from config.py
            st.session_state.client = True  # Just a flag, LiteLLM uses env vars
        except ValueError as e:
            st.sidebar.error(f"❌ {str(e)}")
            st.sidebar.info("💡 Edit config.py to set your API key")
    
    # File Upload
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    
    # Chunking Algorithm Selection
    st.sidebar.markdown("### 🔧 Chunking Algorithm")
    chunking_algorithm = st.sidebar.radio(
        "Select Algorithm",
        CHUNKING_ALGORITHMS
    )
    
    chunk_mode = st.sidebar.selectbox(
        "Chunking Strategy",
        CHUNKING_MODES
    )
    
    chunk_size = st.sidebar.number_input(
        "Chunk Size", 
        min_value=50,
        max_value=1000,
        value=DEFAULT_CHUNK_SIZE,
        step=10,
        help="Number of tokens per chunk"
    )
    
    overlap = st.sidebar.number_input(
        "Overlap", 
        min_value=0,
        max_value=500,
        value=DEFAULT_OVERLAP,
        step=5,
        help="Number of overlapping tokens between chunks"
    )
    
    top_k = st.sidebar.number_input(
        "Top-K Retrieval", 
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        step=1,
        help="Number of chunks to retrieve for each query"
    )
    
    st.sidebar.markdown("---")
    build_index = st.sidebar.button("🚀 Build Index")
    
    # Benchmarking Section
    st.sidebar.markdown("### 📊 Benchmarking")
    benchmark_query = st.sidebar.text_input(
        "Benchmark Query", 
        placeholder="e.g., What is the date?"
    )
    
    num_benchmark_runs = st.sidebar.slider(
        "Number of Runs", 
        1, MAX_BENCHMARK_RUNS, 
        DEFAULT_BENCHMARK_RUNS,
        help="Fewer runs for free API tier"
    )
    
    api_delay = st.sidebar.slider(
        "Delay Between Calls (seconds)", 
        MIN_API_DELAY, MAX_API_DELAY, 
        DEFAULT_API_DELAY, 0.5,
        help="Increase for free API tier"
    )
    
    run_benchmark = st.sidebar.button("🏃 Run Benchmark")
    compare_algorithms = st.sidebar.button("⚖️ Compare Algorithms")
    
    return {
        "uploaded_file": uploaded_file,
        "chunking_algorithm": chunking_algorithm,
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k,
        "build_index": build_index,
        "benchmark_query": benchmark_query,
        "num_benchmark_runs": num_benchmark_runs,
        "api_delay": api_delay,
        "run_benchmark": run_benchmark,
        "compare_algorithms": compare_algorithms,
    }
