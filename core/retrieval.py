"""
Retrieval Module
Handles RAG retrieval from FAISS index with confidence scoring
"""

import numpy as np
from typing import List, Dict

from .embeddings import embed_query


def retrieve(
    client,
    index,
    chunks: List[str],
    query: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieve relevant chunks for a query using FAISS index.
    
    Args:
        client: Gemini API client for query embedding
        index: FAISS index
        chunks: List of text chunks
        query: Query string
        top_k: Number of results to return
        
    Returns:
        List of dicts with chunk, distance, and confidence
    """
    q_emb = embed_query(client, query)
    
    D, I = index.search(q_emb, top_k)
    
    results = []
    for dist, idx in zip(D[0], I[0]):
        chunk_text = chunks[idx]
        
        # Convert L2 distance to cosine similarity for better confidence scoring
        # For normalized embeddings: L2_distance² = 2(1 - cosine_similarity)
        # Therefore: cosine_similarity = 1 - (L2_distance² / 2)
        # Clamp to [0, 1] range for safety
        cosine_sim = max(0, min(1, 1 - (dist / 2)))
        
        # Convert to percentage confidence
        confidence = round(cosine_sim * 100, 2)
        
        results.append({
            "chunk": chunk_text,
            "distance": float(dist),
            "confidence": confidence  # %
        })

    return results
