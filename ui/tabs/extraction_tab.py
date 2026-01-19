"""
Tab 1: Document Extraction
Interactive field extraction with chat interface
"""

import streamlit as st

from core.retrieval import retrieve
from core.extraction import extract_field_value_simple
from ui.components import render_pdf_preview, render_chat_message
from config import LITE_GENERATION_MODEL


def render(settings: dict) -> None:
    """
    Render the Document Extraction tab.
    
    Args:
        settings: Dict of sidebar settings
    """
    st.markdown("### Extract Fields from Your Document")
    st.markdown("Upload a PDF, build the index, and extract specific fields using natural language queries.")
    
    if "index" not in st.session_state:
        st.info("👈 Please upload a PDF and build the index from the sidebar to get started.")
        return
    
    col1, col2 = st.columns([1, 1.5])
    
    # LEFT: CHAT
    with col1:
        st.markdown("#### 💬 Field Extraction Chat")
        
        chat_box = st.container(height=400)
        with chat_box:
            if "chat" in st.session_state and st.session_state.chat:
                for item in st.session_state.chat:
                    # Support both old format (q, a, c) and new format (q, a, c, r)
                    if len(item) == 4:
                        q, a, c, r = item
                    else:
                        q, a, c = item
                        r = ""
                    render_chat_message(q, a, c, r)
            else:
                st.info("No extractions yet. Ask a question below!")
        
        st.markdown("---")
        query = st.text_input(
            "Ask a field extraction question", 
            placeholder="e.g., What is the Agreement Date?",
            key="extraction_query"
        )
        extract_btn = st.button("🎯 Extract Field", type="primary", use_container_width=True)
    
    # RIGHT: DOCUMENT PREVIEW
    with col2:
        st.markdown("#### 📄 Document Preview")
        
        if "pdf_bytes" in st.session_state:
            render_pdf_preview(
                st.session_state.pdf_bytes,
                st.session_state.get("pdf_text", "")
            )
        elif "pdf_text" in st.session_state:
            st.info("📄 Original PDF preview not available. Please rebuild the index.")
    
    # EXTRACTION LOGIC
    if extract_btn and query:
        with st.spinner("🔍 Retrieving context..."):
            results = retrieve(
                st.session_state.client,
                st.session_state.index,
                st.session_state.chunks,
                query,
                settings.get("top_k", 5)
            )
        
        context_text = "\n\n".join([r["chunk"] for r in results])
        st.session_state.last_chunks = results
        
        # Extract value with confidence scoring
        extracted_value, confidence, reason = extract_field_value_simple(
            st.session_state.client,
            query,
            context_text
        )
        
        # Store chat with reasoning (query, value, confidence, reason)
        st.session_state.chat.append((query, extracted_value, confidence, reason))
        st.session_state.last_extracted = [extracted_value]
        st.rerun()
    
    # MATCHED CHUNKS EVIDENCE
    if "last_chunks" in st.session_state and st.session_state.last_chunks:
        st.markdown("---")
        st.markdown("#### 🔍 Matched Chunks (Evidence)")
        
        for i, r in enumerate(st.session_state.last_chunks, 1):
            with st.expander(f"Chunk {i} - Retrieval Similarity: {r['confidence']}%"):
                st.markdown(f"**Distance:** {round(r['distance'], 4)}")
                st.markdown("**Content:**")
                st.text(r['chunk'])
