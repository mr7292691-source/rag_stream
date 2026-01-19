"""
RAG PDF  Field Extractor
A production-grade document field extraction system using Gemini AI and FAISS.

This is the main entry point that orchestrates all modules.
"""

import streamlit as st
import nltk

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Import configuration
from config import PAGE_TITLE, PAGE_ICON, LAYOUT

# Import core modules
from core.pdf_reader import read_pdf
from core.chunking import chunk_text
from core.embeddings import embed_documents
from core.faiss_index import build_faiss_index

# Import UI modules
from ui.sidebar import render_sidebar
from ui.tabs import (
    render_dashboard_tab,
    render_extraction_tab,
    render_comparison_tab,
    render_benchmark_tab,
    render_settings_tab,
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT
)

st.title(f"{PAGE_ICON} RAG PDF Field Extractor (Gemini + FAISS)")


# ============================================================================
# SIDEBAR
# ============================================================================

settings = render_sidebar()


# ============================================================================
# INDEX BUILDING
# ============================================================================

if settings["build_index"] and settings["uploaded_file"]:
    if "client" not in st.session_state:
        st.error("Please set API key in config.py first.")
        st.stop()

    # Store PDF bytes for preview
    settings["uploaded_file"].seek(0)
    st.session_state.pdf_bytes = settings["uploaded_file"].read()
    settings["uploaded_file"].seek(0)
    
    pdf_filename = settings["uploaded_file"].name

    with st.spinner("📄 Reading PDF..."):
        text = read_pdf(settings["uploaded_file"])

    if not text.strip():
        st.error("❌ No readable text found. PDF might be scanned.")
        st.stop()
    
    # Check if index already exists for this document
    from core import check_index_exists, load_index, save_index, get_document_hash
    
    existing_hash = check_index_exists(text)
    
    if existing_hash:
        st.info(f"📦 Found existing index for this document (hash: {existing_hash[:8]}...)")
        with st.spinner("⚡ Loading saved index..."):
            loaded_data = load_index(existing_hash)
            
            if loaded_data:
                index, chunks, metadata = loaded_data
                st.session_state.pdf_text = text
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.chat = []
                st.session_state.last_extracted = []
                
                st.success(f"✅ Loaded index with {len(chunks)} chunks (created: {metadata.get('created_at', 'unknown')})")
                st.info("💡 Index loaded from disk - no API calls needed!")
                st.stop()

    # Build new index if not found
    with st.spinner("✂️ Chunking text..."):
        chunks = chunk_text(
            text,
            algorithm=settings["chunking_algorithm"],
            mode=settings["chunk_mode"],
            size=settings["chunk_size"],
            overlap=settings["overlap"]
        )

    if not chunks:
        st.error("❌ Chunking produced no chunks.")
        st.stop()

    st.info(f"🧩 Total chunks: {len(chunks)}")

    with st.spinner("🧠 Creating embeddings using Gemini..."):
        try:
            embeddings = embed_documents(st.session_state.client, chunks)
        except Exception as e:
            st.error(f"❌ Embedding failed: {e}")
            st.stop()

    if embeddings.size == 0:
        st.error("❌ No embeddings created. Check API key or quota.")
        st.stop()

    with st.spinner("📦 Building FAISS index..."):
        index = build_faiss_index(embeddings)

    # Store in session state
    st.session_state.pdf_text = text
    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.chat = []
    st.session_state.last_extracted = []
    
    # Save index and metadata to disk
    with st.spinner("💾 Saving index to disk..."):
        chunking_config = {
            "algorithm": settings["chunking_algorithm"],
            "mode": settings["chunk_mode"],
            "size": settings["chunk_size"],
            "overlap": settings["overlap"]
        }
        
        doc_hash = save_index(
            index,
            chunks,
            text,
            pdf_filename,
            chunking_config
        )
        
        st.success(f"✅ Index saved! (hash: {doc_hash[:8]}...)")
        st.info("💡 Next time you upload this PDF, the index will load instantly!")

    st.success("✅ FAISS index built successfully!")


# ============================================================================
# MAIN TABS
# ============================================================================

TAB_NAMES = [
    "🏠 Dashboard",
    "📄 Document Extraction",
    "⚖️ Flow Comparison",
    "📊 Benchmarking",
    "⚙️ Settings"
]

tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_NAMES)

with tab1:
    render_dashboard_tab(settings)

with tab2:
    render_extraction_tab(settings)

with tab3:
    render_comparison_tab(settings)

with tab4:
    render_benchmark_tab(settings)

with tab5:
    render_settings_tab(settings)
