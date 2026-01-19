"""
Index Persistence Module
Save and load FAISS indexes with metadata for processed documents
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import faiss
import numpy as np


def get_document_hash(text: str) -> str:
    """
    Generate a unique hash for a document based on its content.
    
    Args:
        text: Document text
        
    Returns:
        SHA256 hash of the document
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def get_index_directory(base_dir: str = "indexes") -> str:
    """
    Get or create the directory for storing indexes.
    
    Args:
        base_dir: Base directory name for indexes
        
    Returns:
        Absolute path to index directory
    """
    index_dir = os.path.abspath(base_dir)
    os.makedirs(index_dir, exist_ok=True)
    return index_dir


def get_index_path(document_hash: str, base_dir: str = "indexes") -> Tuple[str, str]:
    """
    Get file paths for index and metadata.
    
    Args:
        document_hash: Hash of the document
        base_dir: Base directory for indexes
        
    Returns:
        Tuple of (index_path, metadata_path)
    """
    index_dir = get_index_directory(base_dir)
    index_path = os.path.join(index_dir, f"{document_hash}.faiss")
    metadata_path = os.path.join(index_dir, f"{document_hash}_metadata.json")
    return index_path, metadata_path


def save_index(
    index: faiss.Index,
    chunks: List[str],
    document_text: str,
    pdf_filename: str,
    chunking_config: Dict,
    base_dir: str = "indexes"
) -> str:
    """
    Save FAISS index and metadata to disk.
    
    Args:
        index: FAISS index object
        chunks: List of text chunks
        document_text: Original document text
        pdf_filename: Name of the PDF file
        chunking_config: Configuration used for chunking
        base_dir: Base directory for indexes
        
    Returns:
        Document hash (identifier for this index)
    """
    doc_hash = get_document_hash(document_text)
    index_path, metadata_path = get_index_path(doc_hash, base_dir)
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    
    # Save metadata
    metadata = {
        "document_hash": doc_hash,
        "pdf_filename": pdf_filename,
        "created_at": datetime.now().isoformat(),
        "num_chunks": len(chunks),
        "chunks": chunks,
        "chunking_config": chunking_config,
        "document_length": len(document_text),
        "index_dimension": index.d,
        "index_total": index.ntotal
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return doc_hash


def load_index(
    document_hash: str,
    base_dir: str = "indexes"
) -> Optional[Tuple[faiss.Index, List[str], Dict]]:
    """
    Load FAISS index and metadata from disk.
    
    Args:
        document_hash: Hash of the document
        base_dir: Base directory for indexes
        
    Returns:
        Tuple of (index, chunks, metadata) or None if not found
    """
    index_path, metadata_path = get_index_path(document_hash, base_dir)
    
    # Check if files exist
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None
    
    try:
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        chunks = metadata.get("chunks", [])
        
        return index, chunks, metadata
    
    except Exception as e:
        print(f"Error loading index: {e}")
        return None


def check_index_exists(
    document_text: str,
    base_dir: str = "indexes"
) -> Optional[str]:
    """
    Check if an index exists for a document.
    
    Args:
        document_text: Document text to check
        base_dir: Base directory for indexes
        
    Returns:
        Document hash if index exists, None otherwise
    """
    doc_hash = get_document_hash(document_text)
    index_path, metadata_path = get_index_path(doc_hash, base_dir)
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        return doc_hash
    
    return None


def list_saved_indexes(base_dir: str = "indexes") -> List[Dict]:
    """
    List all saved indexes with their metadata.
    
    Args:
        base_dir: Base directory for indexes
        
    Returns:
        List of metadata dictionaries
    """
    index_dir = get_index_directory(base_dir)
    indexes = []
    
    for filename in os.listdir(index_dir):
        if filename.endswith("_metadata.json"):
            metadata_path = os.path.join(index_dir, filename)
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    indexes.append(metadata)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return indexes


def delete_index(
    document_hash: str,
    base_dir: str = "indexes"
) -> bool:
    """
    Delete a saved index and its metadata.
    
    Args:
        document_hash: Hash of the document
        base_dir: Base directory for indexes
        
    Returns:
        True if deleted successfully, False otherwise
    """
    index_path, metadata_path = get_index_path(document_hash, base_dir)
    
    try:
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        return True
    except Exception as e:
        print(f"Error deleting index: {e}")
        return False
