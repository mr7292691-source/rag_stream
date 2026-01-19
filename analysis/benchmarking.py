"""
Benchmarking Module
Performance testing and algorithm comparison utilities
"""

import time
import numpy as np
from typing import List, Dict, Optional, Callable

from core.chunking import chunk_text_sliding_window, chunk_text_recursive
from core.embeddings import embed_documents
from core.faiss_index import build_faiss_index


def run_benchmark_test(
    query: str,
    extractor: Callable,
    num_runs: int = 5,
    delay_seconds: float = 2.0,
    progress_callback: Optional[Callable] = None
) -> List[Dict]:
    """
    Run extraction multiple times and collect metrics.
    
    Args:
        query: The extraction query to benchmark
        extractor: Function that takes query and returns (value, confidence, chunks)
        num_runs: Number of iterations
        delay_seconds: Delay between API calls
        progress_callback: Optional callback(current, total, status)
        
    Returns:
        List of result dicts with run metrics
    """
    results = []
    
    for i in range(num_runs):
        if progress_callback:
            progress_callback(i, num_runs, f"Running iteration {i+1}/{num_runs}")
        
        start_time = time.time()
        
        try:
            extracted_value, confidence, chunks = extractor(query)
            elapsed_time = time.time() - start_time
            
            results.append({
                "run": i + 1,
                "value": extracted_value,
                "confidence": confidence,
                "time_ms": round(elapsed_time * 1000, 2),
                "num_chunks": len(chunks) if chunks else 0
            })
            
            # Add delay between runs (except for last run)
            if i < num_runs - 1:
                if progress_callback:
                    progress_callback(i, num_runs, f"Waiting {delay_seconds}s (rate limit protection)")
                time.sleep(delay_seconds)
                
        except Exception as e:
            results.append({
                "run": i + 1,
                "value": "ERROR",
                "confidence": 0,
                "time_ms": 0,
                "num_chunks": 0,
                "error": str(e)
            })
            # Longer delay after error
            if i < num_runs - 1:
                time.sleep(delay_seconds * 2)
    
    return results


def compare_chunking_algorithms(
    query: str,
    document_text: str,
    client,
    chunk_mode: str,
    chunk_size: int,
    overlap: int,
    top_k: int,
    num_runs: int = 5,
    delay_seconds: float = 2.0,
    benchmark_runner: Callable = None,
    progress_callback: Optional[Callable] = None
) -> Optional[Dict]:
    """
    Compare both chunking algorithms on the same query.
    
    Args:
        query: The extraction query
        document_text: Full document text
        client: Gemini API client
        chunk_mode: Chunking mode (token/sentence/paragraph)
        chunk_size: Chunk size parameter
        overlap: Overlap parameter
        top_k: Number of chunks to retrieve
        num_runs: Number of benchmark runs per algorithm
        delay_seconds: Delay between API calls
        benchmark_runner: Function to run benchmark with given index/chunks
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dict with comparison results for both algorithms
    """
    from core.retrieval import retrieve
    from core.extraction import extract_field_value_simple
    
    comparison_results = {}
    
    for algo_idx, algo in enumerate(["Sliding Window", "Recursive"]):
        if progress_callback:
            progress_callback(algo_idx, 2, f"Testing {algo} algorithm...")
        
        # Chunk text with this algorithm
        if algo == "Sliding Window":
            chunks = chunk_text_sliding_window(
                document_text, 
                chunk_mode, 
                chunk_size, 
                overlap
            )
        else:
            chunks = chunk_text_recursive(
                document_text, 
                chunk_mode, 
                chunk_size, 
                overlap
            )
        
        if not chunks:
            continue
        
        # Build embeddings and index
        try:
            embeddings = embed_documents(client, chunks)
            if embeddings.size == 0:
                continue
            
            temp_index = build_faiss_index(embeddings)
        except Exception:
            continue
        
        # Create extractor function for this index
        def make_extractor(idx, chnks):
            def extractor(q):
                results = retrieve(client, idx, chnks, q, top_k)
                context = "\n".join([r["chunk"] for r in results])
                value, confidence, reason = extract_field_value_simple(client, q, context)
                return value, round(confidence, 1), results
            return extractor
        
        # Run benchmark
        results = run_benchmark_test(
            query,
            make_extractor(temp_index, chunks),
            num_runs=num_runs,
            delay_seconds=delay_seconds,
            progress_callback=progress_callback
        )
        
        if results:
            values = [r["value"] for r in results]
            comparison_results[algo] = {
                "results": results,
                "num_chunks": len(chunks),
                "avg_confidence": round(np.mean([r["confidence"] for r in results]), 2),
                "avg_time_ms": round(np.mean([r["time_ms"] for r in results]), 2),
                "consistency": len(set(values)),
                "most_common_value": max(set(values), key=values.count)
            }
        
        # Add delay between algorithms
        if algo_idx < 1:
            time.sleep(delay_seconds * 2)
    
    return comparison_results if comparison_results else None
