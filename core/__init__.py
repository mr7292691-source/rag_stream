# Core module - PDF processing, chunking, embeddings, LLM, and retrieval
from .pdf_reader import read_pdf
from .chunking import chunk_text, chunk_text_sliding_window, chunk_text_recursive
from .embeddings import embed_documents, embed_query
from .faiss_index import build_faiss_index
from .retrieval import retrieve
from .extraction import extract_field_value, extract_field_value_simple, extract_all_fields
from .index_persistence import (
    save_index,
    load_index,
    check_index_exists,
    get_document_hash,
    list_saved_indexes,
    delete_index
)
from .llm import (
    create_client,
    get_embeddings,
    get_single_embedding,
    generate_text,
    generate_json,
    generate_with_retry,
    generate_json_with_retry,
    generate_with_pydantic,
    calculate_cost
)

__all__ = [
    # PDF
    "read_pdf",
    # Chunking
    "chunk_text",
    "chunk_text_sliding_window", 
    "chunk_text_recursive",
    # Embeddings
    "embed_documents",
    "embed_query",
    "get_embeddings",
    "get_single_embedding",
    # Index
    "build_faiss_index",
    # Retrieval
    "retrieve",
    # Extraction
    "extract_field_value",
    "extract_field_value_simple",
    "extract_all_fields",
    # Index Persistence
    "save_index",
    "load_index",
    "check_index_exists",
    "get_document_hash",
    "list_saved_indexes",
    "delete_index",
    # LLM
    "create_client",
    "generate_text",
    "generate_json",
    "generate_with_retry",
    "generate_json_with_retry",
    "generate_with_pydantic",
    "calculate_cost",
]
