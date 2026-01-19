"""
FAISS Index Module
Handles FAISS vector index creation and management
"""

import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from embeddings.
    
    Args:
        embeddings: numpy array of embeddings (N x D)
        
    Returns:
        FAISS index
        
    Raises:
        ValueError: If embeddings array is empty
    """
    if embeddings.size == 0:
        raise ValueError("No embeddings provided. Cannot build index.")
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    return index
