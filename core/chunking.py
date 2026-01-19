"""
Chunking Module
Provides different text chunking algorithms for RAG
"""

import tiktoken
from nltk.tokenize import sent_tokenize
from typing import List

from config import TIKTOKEN_ENCODING


def chunk_text_sliding_window(
    text: str, 
    mode: str, 
    size: int, 
    overlap: int
) -> List[str]:
    """
    Sliding window chunking algorithm.
    
    Args:
        text: Input text to chunk
        mode: Chunking mode - 'token', 'sentence', or 'paragraph'
        size: Chunk size (tokens/sentences/words depending on mode)
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    if mode == "paragraph":
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    if mode == "sentence":
        sents = sent_tokenize(text)
        chunks, buf, word_count = [], [], 0
        
        for s in sents:
            buf.append(s)
            word_count += len(s.split())
            if word_count >= size:
                chunks.append(" ".join(buf))
                buf, word_count = [], 0
        
        if buf:
            chunks.append(" ".join(buf))
        return chunks

    # Token level sliding window
    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    tokens = enc.encode(text)
    chunks = []
    step = max(size - overlap, 1)
    
    for i in range(0, len(tokens), step):
        part = tokens[i:i + size]
        if part:
            chunks.append(enc.decode(part))
    
    return chunks


def chunk_text_recursive(
    text: str, 
    mode: str, 
    size: int, 
    overlap: int
) -> List[str]:
    """
    Recursive chunking algorithm - splits text hierarchically.
    
    Args:
        text: Input text to chunk
        mode: Chunking mode - 'token', 'sentence', or 'paragraph'
        size: Target chunk size
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
    
    # Define separators for recursive splitting
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators"""
        if not separators:
            return [text]
        
        separator = separators[0]
        if separator == "":
            # Character-level split as last resort
            return list(text)
        
        splits = text.split(separator)
        result = []
        
        for split in splits:
            if len(split) > size:
                # Recursively split with next separator
                result.extend(split_text(split, separators[1:]))
            elif split.strip():
                result.append(split)
        
        return result
    
    # Get initial splits based on mode
    if mode == "paragraph":
        initial_chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    elif mode == "sentence":
        initial_chunks = sent_tokenize(text)
    else:
        # Token-based recursive
        enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
        tokens = enc.encode(text)
        
        # Recursively split tokens
        chunks = []
        i = 0
        while i < len(tokens):
            end = min(i + size, len(tokens))
            chunk_tokens = tokens[i:end]
            chunks.append(enc.decode(chunk_tokens))
            i += size - overlap if overlap > 0 else size
        
        return chunks
    
    # Merge small chunks and split large ones
    final_chunks = []
    current_chunk = ""
    
    for chunk in initial_chunks:
        if len(current_chunk.split()) + len(chunk.split()) <= size:
            current_chunk += (" " if current_chunk else "") + chunk
        else:
            if current_chunk:
                final_chunks.append(current_chunk)
            if len(chunk.split()) > size:
                # Split large chunk recursively
                sub_chunks = split_text(chunk, separators)
                final_chunks.extend([c for c in sub_chunks if c.strip()])
                current_chunk = ""
            else:
                current_chunk = chunk
    
    if current_chunk:
        final_chunks.append(current_chunk)
    
    return final_chunks


def chunk_text(
    text: str,
    algorithm: str = "Sliding Window",
    mode: str = "token",
    size: int = 200,
    overlap: int = 20
) -> List[str]:
    """
    Main chunking function that routes to the selected algorithm.
    
    Args:
        text: Input text to chunk
        algorithm: 'Sliding Window' or 'Recursive'
        mode: Chunking mode - 'token', 'sentence', or 'paragraph'
        size: Chunk size
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if algorithm == "Sliding Window":
        return chunk_text_sliding_window(text, mode, size, overlap)
    else:
        return chunk_text_recursive(text, mode, size, overlap)
