"""
Tab: Settings
Application settings and information
"""

import streamlit as st

from config import LITE_GENERATION_MODEL


def render(settings: dict) -> None:
    """
    Render the Settings tab.
    
    Args:
        settings: Dict of sidebar settings
    """
    st.markdown("### ⚙️ Application Settings & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Current Configuration")
        
        if "index" in st.session_state:
            st.success("✅ Index Status: Built")
            st.info(f"📊 Total Chunks: {len(st.session_state.chunks)}")
            st.info(f"🔧 Algorithm: {settings.get('chunking_algorithm', 'N/A')}")
            st.info(f"📏 Chunk Size: {settings.get('chunk_size', 'N/A')}")
            st.info(f"🔄 Overlap: {settings.get('overlap', 'N/A')}")
            st.info(f"🎯 Top-K: {settings.get('top_k', 'N/A')}")
        else:
            st.warning("⚠️ No index built yet")
        
        st.markdown("---")
        
        st.markdown("#### 🔑 API Configuration")
        if "client" in st.session_state:
            st.success("✅ API Key: Configured")
        else:
            st.error("❌ API Key: Not configured")
        
        st.info(f"🤖 Model: {LITE_GENERATION_MODEL}")
        st.info(f"📊 Benchmark Runs: {settings.get('num_benchmark_runs', 'N/A')}")
        st.info(f"⏱️ API Delay: {settings.get('api_delay', 'N/A')}s")
    
    with col2:
        st.markdown("#### 📚 How to Use")
        
        st.markdown("""
        **1. Upload & Build Index**
        - Upload a PDF from the sidebar
        - Configure chunking settings
        - Click "🚀 Build Index"
        
        **2. Extract Fields**
        - Go to "Document Extraction" tab
        - Ask questions in natural language
        - View extracted values and confidence
        
        **3. Compare Flows**
        - Go to "Flow Comparison" tab
        - Compare Zero-shot vs RAG extraction
        - Analyze accuracy and hallucination metrics
        
        **4. Benchmark Performance**
        - Go to "Benchmarking" tab
        - Add multiple fields to test
        - Compare chunking algorithms side-by-side
        """)
        
        st.markdown("---")
        
        st.markdown("#### ℹ️ About")
        st.markdown("""
        **RAG PDF Field Extractor**
        
        A production-grade document field extraction system using:
        - 🤖 Google Gemini AI
        - 🔍 FAISS Vector Search
        - 📊 Advanced Chunking Algorithms
        - ⚡ Rate Limit Protection
        """)
        
        st.markdown("---")
        
        st.markdown("#### 📁 Project Structure")
        st.code("""
stream/
├── app.py              # Main entry point
├── config.py           # Configuration
├── core/               # Core logic
├── analysis/           # Analysis modules
├── ui/                 # UI components
│   └── tabs/           # Tab pages
└── utils/              # Utilities
        """, language="text")
