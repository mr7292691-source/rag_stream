"""
Flow Comparison Module
Compare Zero-shot vs RAG extraction approaches with accurate token tracking
"""

import time
import json
from typing import List, Dict, Tuple, Optional, Callable

from config import GENERATION_MODEL, MAX_DOCUMENT_LENGTH
from .hallucination import calculate_hallucination_score
from utils.token_counter import TokenTracker, track_llm_usage


def zero_shot_extraction(
    client,
    document_text: str,
    fields: List[Dict],
    delay_seconds: float = 2.0,
    custom_prompt: Optional[str] = None
) -> Tuple[Optional[List[Dict]], Dict]:
    """
    Zero-shot extraction: Send entire document + prompt to LLM.
    
    Args:
        client: Gemini API client
        document_text: Full document text
        fields: List of field dicts with 'field_name'
        delay_seconds: API delay (unused, single call)
        custom_prompt: Optional custom prompt with {FIELDS} and {DOCUMENT} placeholders
        
    Returns:
        Tuple of (results list, metrics dict)
    """
    start_time = time.time()
    tracker = TokenTracker()
    
    # Build field list for prompt
    field_list = "\n".join([f"- {f['field_name']}" for f in fields])
    
    # Use custom prompt if provided, otherwise use default
    if custom_prompt:
        extraction_prompt = custom_prompt.replace(
            "{FIELDS}", field_list
        ).replace(
            "{DOCUMENT}", document_text[:MAX_DOCUMENT_LENGTH]
        )
    else:
        extraction_prompt = f"""You are a document field extraction expert. Extract the following fields and provide confidence scores.

FIELDS TO EXTRACT:
{field_list}

DOCUMENT:
{document_text[:MAX_DOCUMENT_LENGTH]}

INSTRUCTIONS:
1. Extract exact values for each field from the document
2. For EACH field, provide:
   - The extracted value (or "N/A" if not found)
   - Your confidence score (0-100) based on how clearly the value appears
   - A brief reason explaining your confidence level
3. Be precise - extract only what's in the document

Return your response in this EXACT JSON format:
{{
  "Field Name 1": {{"value": "extracted value", "confidence": 85, "reason": "why you chose this"}},
  "Field Name 2": {{"value": "N/A", "confidence": 10, "reason": "field not found in document"}},
  ...
}}

Return ONLY the JSON object, no other text."""

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=extraction_prompt
        )
        
        elapsed_time = time.time() - start_time
        
        # Track token usage
        tracker.add_llm_usage(extraction_prompt, response.text)
        
        # Parse response
        response_text = response.text.strip()
        if response_text.startswith("```"):
            parts = response_text.split("```")
            if len(parts) >= 2:
                response_text = parts[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
        
        response_text = response_text.strip()
        
        try:
            extracted = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            # If JSON parsing fails, return error with the response text for debugging
            error_dict = {
                "error": f"JSON parsing failed: {str(json_err)}. Response: {response_text[:200]}",
                "total_time": round(elapsed_time, 2),
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
                "llm_total_tokens": 0,
                "embedding_tokens": 0,
                "total_tokens": 0,
                "api_calls": 1,
                "llm_calls": 1,
                "embedding_calls": 0
            }
            return None, error_dict
        
        # Get token summary
        token_summary = tracker.get_summary()
        
        metrics = {
            "total_time": round(elapsed_time, 2),
            "llm_input_tokens": token_summary["llm_input_tokens"],
            "llm_output_tokens": token_summary["llm_output_tokens"],
            "llm_total_tokens": token_summary["llm_total_tokens"],
            "embedding_tokens": 0,  # Zero-shot doesn't use embeddings
            "total_tokens": token_summary["llm_total_tokens"],
            "api_calls": 1,
            "llm_calls": 1,
            "embedding_calls": 0
        }
        
        # Format results to match field structure
        results = []
        for field in fields:
            field_name = field.get("field_name", "")
            field_data = extracted.get(field_name, {})
            
            if isinstance(field_data, dict):
                value = str(field_data.get("value", "N/A"))
                confidence = float(field_data.get("confidence", 50))
                reason = field_data.get("reason", "No reason provided")
            else:
                value = str(field_data) if field_data else "N/A"
                confidence = 50
                reason = "Legacy format - no confidence provided"
            
            results.append({
                "field_name": field_name,
                "value": value,
                "confidence": round(confidence, 1),
                "confidence_reason": reason,
                "source": "zero-shot"
            })
        
        return results, metrics
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "total_time": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "llm_total_tokens": 0,
            "embedding_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "llm_calls": 0,
            "embedding_calls": 0
        }
        return None, error_dict


def rag_extraction(
    client,
    fields: List[Dict],
    retriever: Callable,
    delay_seconds: float = 2.0,
    progress_callback: Optional[Callable] = None
) -> Tuple[Optional[List[Dict]], Dict]:
    """
    RAG extraction: Use chunking + embedding + retrieval for each field.
    
    Args:
        client: Gemini API client
        fields: List of field dicts with 'field_name' and 'query'
        retriever: Function to retrieve context for a query
        delay_seconds: Delay between API calls
        progress_callback: Optional callback(current, total, field_name)
        
    Returns:
        Tuple of (results list, metrics dict)
    """
    start_time = time.time()
    tracker = TokenTracker()
    results = []
    
    for i, field in enumerate(fields):
        field_name = field.get("field_name", f"Field {i+1}")
        query = field.get("query", f"What is the {field_name}?")
        
        if progress_callback:
            progress_callback(i, len(fields), field_name)
        
        try:
            # Retrieve relevant chunks
            retrieved = retriever(query)
            
            # OPTIMIZATION: Use only top 2 most relevant chunks (not 3 or 5)
            # Each chunk is ~200 tokens, so 2 chunks = ~400 tokens max
            top_chunks = retrieved[:min(2, len(retrieved))]
            
            # OPTIMIZATION: Limit each chunk to 200 chars (~50 tokens) to minimize context
            # This ensures RAG uses less tokens than zero-shot
            context_parts = []
            for chunk_data in top_chunks:
                chunk_text = chunk_data["chunk"]
                # Aggressively truncate to keep only most relevant part
                if len(chunk_text) > 200:
                    chunk_text = chunk_text[:200] + "..."
                context_parts.append(chunk_text)
            
            context_text = "\n".join(context_parts)  # Single newline, not double
            
            # Track embedding usage (query embedding)
            tracker.add_embedding_usage([query])
            
            # OPTIMIZATION: Ultra-concise prompt to minimize tokens
            extraction_prompt = f"""Context: {context_text}

Extract {field_name} for: {query}

JSON: {{"value":"...","confidence":0-100,"reason":"..."}}"""

            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=extraction_prompt
            )
            
            # Track LLM usage
            tracker.add_llm_usage(extraction_prompt, response.text)
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            try:
                parsed = json.loads(response_text)
                extracted_value = str(parsed.get("value", "N/A"))
                llm_confidence = float(parsed.get("confidence", 50))
                confidence_reason = parsed.get("reason", "No reason provided")
            except json.JSONDecodeError:
                extracted_value = response.text.strip()
                llm_confidence = 50
                confidence_reason = "Could not parse LLM confidence response"
            
            results.append({
                "field_name": field_name,
                "value": extracted_value,
                "confidence": round(llm_confidence, 1),
                "confidence_reason": confidence_reason,
                "source": "rag",
                "chunks_used": len(top_chunks)
            })
            
            if i < len(fields) - 1:
                time.sleep(delay_seconds)
                
        except Exception as e:
            results.append({
                "field_name": field_name,
                "value": "ERROR",
                "confidence": 0,
                "confidence_reason": str(e),
                "source": "rag",
                "error": str(e)
            })
            if i < len(fields) - 1:
                time.sleep(delay_seconds * 2)
    
    elapsed_time = time.time() - start_time
    
    # Get token summary
    token_summary = tracker.get_summary()
    
    metrics = {
        "total_time": round(elapsed_time, 2),
        "llm_input_tokens": token_summary["llm_input_tokens"],
        "llm_output_tokens": token_summary["llm_output_tokens"],
        "llm_total_tokens": token_summary["llm_total_tokens"],
        "embedding_tokens": token_summary["embedding_tokens"],
        "total_tokens": token_summary["total_tokens"],
        "api_calls": token_summary["total_calls"],
        "llm_calls": token_summary["llm_calls"],
        "embedding_calls": token_summary["embedding_calls"],
        "avg_time_per_field": round(elapsed_time / len(fields), 2) if fields else 0
    }
    
    return results, metrics


def compare_outputs(
    master_output: List[Dict],
    zero_shot_results: Optional[List[Dict]],
    rag_results: Optional[List[Dict]],
    document_text: str = ""
) -> Dict:
    """
    Compare zero-shot and RAG results against master output.
    
    Args:
        master_output: Ground truth field values
        zero_shot_results: Zero-shot extraction results
        rag_results: RAG extraction results
        document_text: Original document text for hallucination check
        
    Returns:
        Comprehensive comparison metrics dict
    """
    comparison = {
        "fields": [],
        "zero_shot_summary": {},
        "rag_summary": {}
    }
    
    # Field-level comparison
    zs_correct = 0
    zs_partial = 0
    zs_hallucination_total = 0
    rag_correct = 0
    rag_partial = 0
    rag_hallucination_total = 0
    
    for master_field in master_output:
        field_name = master_field.get("field_name", "")
        master_value = str(master_field.get("value", ""))
        
        # Find corresponding results
        zs_result = next(
            (r for r in zero_shot_results if r["field_name"] == field_name), 
            None
        ) if zero_shot_results else None
        rag_result = next(
            (r for r in rag_results if r["field_name"] == field_name), 
            None
        ) if rag_results else None
        
        zs_value = zs_result["value"] if zs_result else "N/A"
        rag_value = rag_result["value"] if rag_result else "N/A"
        
        # Calculate match scores
        def calc_match(extracted, master):
            if not master or master.lower() == "n/a":
                return "N/A", 0
            extracted_lower = extracted.lower().strip()
            master_lower = master.lower().strip()
            
            if extracted_lower == master_lower:
                return "exact", 100
            elif master_lower in extracted_lower or extracted_lower in master_lower:
                return "partial", 70
            elif any(word in extracted_lower for word in master_lower.split()):
                return "fuzzy", 40
            else:
                return "mismatch", 0
        
        zs_match_type, zs_score = calc_match(zs_value, master_value)
        rag_match_type, rag_score = calc_match(rag_value, master_value)
        
        if zs_score == 100:
            zs_correct += 1
        elif zs_score > 0:
            zs_partial += 1
            
        if rag_score == 100:
            rag_correct += 1
        elif rag_score > 0:
            rag_partial += 1
        
        # Calculate hallucination scores
        zs_halluc = calculate_hallucination_score(zs_value, master_value, document_text)
        rag_halluc = calculate_hallucination_score(rag_value, master_value, document_text)
        zs_hallucination_total += zs_halluc
        rag_hallucination_total += rag_halluc
        
        comparison["fields"].append({
            "field_name": field_name,
            "master_value": master_value,
            "zero_shot_value": zs_value,
            "zero_shot_match": zs_match_type,
            "zero_shot_score": zs_score,
            "zero_shot_confidence": zs_result.get("confidence", 0) if zs_result else 0,
            "zero_shot_hallucination": zs_halluc,
            "rag_value": rag_value,
            "rag_match": rag_match_type,
            "rag_score": rag_score,
            "rag_confidence": rag_result.get("confidence", 0) if rag_result else 0,
            "rag_hallucination": rag_halluc
        })
    
    total_fields = len(master_output)
    
    # Summary metrics
    comparison["zero_shot_summary"] = {
        "accuracy": round((zs_correct / total_fields) * 100, 1) if total_fields > 0 else 0,
        "partial_match": round((zs_partial / total_fields) * 100, 1) if total_fields > 0 else 0,
        "exact_matches": zs_correct,
        "partial_matches": zs_partial,
        "mismatches": total_fields - zs_correct - zs_partial,
        "avg_hallucination": round(zs_hallucination_total / total_fields, 1) if total_fields > 0 else 0,
        "field_coverage": round(
            len([r for r in zero_shot_results if r["value"] not in ["N/A", "ERROR"]]) / total_fields * 100, 1
        ) if zero_shot_results and total_fields > 0 else 0
    }
    
    comparison["rag_summary"] = {
        "accuracy": round((rag_correct / total_fields) * 100, 1) if total_fields > 0 else 0,
        "partial_match": round((rag_partial / total_fields) * 100, 1) if total_fields > 0 else 0,
        "exact_matches": rag_correct,
        "partial_matches": rag_partial,
        "mismatches": total_fields - rag_correct - rag_partial,
        "avg_hallucination": round(rag_hallucination_total / total_fields, 1) if total_fields > 0 else 0,
        "field_coverage": round(
            len([r for r in rag_results if r["value"] not in ["N/A", "ERROR"]]) / total_fields * 100, 1
        ) if rag_results and total_fields > 0 else 0
    }
    
    return comparison
