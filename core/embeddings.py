"""
Embeddings Module
Handles vector embedding generation using Gemini API
"""

import numpy as np
from google.genai import types
from typing import List, Union

from config import EMBEDDING_MODEL


def embed_documents(client, texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of document texts.
    
    Args:
        client: Gemini API client
        texts: List of text strings to embed
        
    Returns:
        numpy array of embeddings (float32)
    """
    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        embeddings = np.array([e.values for e in result.embeddings]).astype("float32")
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")


def embed_query(client, text: str) -> np.ndarray:
    """
    Generate embedding for a query text.
    
    Args:
        client: Gemini API client  
        text: Query text to embed
        
    Returns:
        numpy array of query embedding (float32)
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )
    return np.array([result.embeddings[0].values]).astype("float32")
