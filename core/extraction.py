"""
Extraction Module
Handles field value extraction from retrieved context with LLM-powered confidence reasoning
"""

import time
import re
import json
from typing import List, Dict, Tuple, Optional, Callable

from config import LITE_GENERATION_MODEL, GENERATION_MODEL, DEFAULT_RETRY_COUNT


def extract_field_value(
    client,
    query: str,
    context: str,
    retry_count: int = DEFAULT_RETRY_COUNT
) -> Tuple[str, float, str]:
    """
    Extract a field value from context with LLM-powered confidence and reasoning.
    Uses the full model for better quality.
    
    Args:
        client: Gemini API client
        query: The extraction query
        context: Context text from retrieved chunks
        retry_count: Number of retries on failure
        
    Returns:
        Tuple of (extracted_value, confidence, reasoning)
    """
    extraction_prompt = f"""You are a document field extraction expert. Extract the value and explain your reasoning.

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Extract the EXACT value that answers the question from the context
2. Rate your confidence (0-100) based on:
   - How clearly the value appears in the context
   - How well it matches what the question is asking for
   - Whether the value is complete and unambiguous
3. Explain WHY you chose this specific value and why you assigned this confidence level

Return your response in this EXACT JSON format:
{{"value": "extracted value or N/A if not found", "confidence": 85, "reasoning": "I found this value because... My confidence is X% because..."}}

Return ONLY the JSON object, no other text."""

    for attempt in range(retry_count):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=extraction_prompt
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            try:
                parsed = json.loads(response_text)
                extracted_value = str(parsed.get("value", "N/A"))
                confidence = float(parsed.get("confidence", 50))
                reasoning = parsed.get("reasoning", "No reasoning provided")
                return extracted_value, confidence, reasoning
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return response.text.strip(), 50.0, "Could not parse LLM response"
                
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a quota error and parse retry delay
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                retry_match = re.search(r'retry in (\d+\.?\d*)s', error_msg)
                if retry_match:
                    retry_delay = float(retry_match.group(1))
                    if attempt < retry_count - 1:
                        time.sleep(min(retry_delay, 60))  # Cap at 60s
                        continue
                raise RuntimeError("Daily quota exceeded. Please try again later or upgrade your API plan.")
            
            if attempt < retry_count - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                time.sleep(wait_time)
            else:
                raise e
    
    return "N/A", 0.0, "Extraction failed after retries"


def extract_field_value_simple(
    client,
    query: str,
    context: str
) -> Tuple[str, float, str]:
    """
    Simple but effective field extraction with LLM reasoning.
    Uses a simpler prompt that works better with the lite model.
    
    Args:
        client: Gemini API client
        query: The extraction query
        context: Context text from retrieved chunks
        
    Returns:
        Tuple of (extracted_value, confidence, reasoning)
    """
    # Step 1: Simple extraction prompt (works better for accuracy)
    extraction_prompt = f"""Extract the answer from the context below.

Context:
{context}

Question: {query}

Instructions:
- Find and return the exact value that answers the question
- If the answer contains multiple parts, include all relevant information
- If not found, say "N/A"
- Be thorough - look for related terms and synonyms

Answer:"""

    try:
        response = client.models.generate_content(
            model=LITE_GENERATION_MODEL,
            contents=extraction_prompt
        )
        
        extracted_value = response.text.strip()
        
        # Step 2: Get confidence and reasoning for the extracted value
        confidence_prompt = f"""Rate your confidence and explain why.

Context: {context[:2000]}

Question: {query}
Extracted Answer: {extracted_value}

Provide a JSON response:
{{"confidence": 0-100, "reasoning": "why this value and confidence"}}

JSON only:"""

        try:
            conf_response = client.models.generate_content(
                model=LITE_GENERATION_MODEL,
                contents=confidence_prompt
            )
            
            conf_text = conf_response.text.strip()
            if conf_text.startswith("```"):
                conf_text = conf_text.split("```")[1]
                if conf_text.startswith("json"):
                    conf_text = conf_text[4:]
            
            parsed = json.loads(conf_text)
            confidence = float(parsed.get("confidence", 70))
            reasoning = parsed.get("reasoning", "Value extracted from context")
            
        except:
            # Fallback: calculate basic confidence
            confidence = 75.0 if extracted_value.lower() not in ["n/a", "not found", ""] else 20.0
            reasoning = "Value found in document context" if confidence > 50 else "Value not clearly found"
        
        return extracted_value, confidence, reasoning
            
    except Exception as e:
        return "ERROR", 0.0, str(e)


def extract_all_fields(
    client,
    fields: List[Dict],
    retriever: Callable,
    delay_seconds: float = 2.0,
    progress_callback: Optional[Callable] = None
) -> List[Dict]:
    """
    Extract values for all identified fields.
    
    Args:
        client: Gemini API client
        fields: List of field dicts with 'field_name' and 'query'
        retriever: Function to retrieve context for a query
        delay_seconds: Delay between API calls
        progress_callback: Optional callback(current, total, field_name)
        
    Returns:
        List of extraction results
    """
    if not fields:
        return []
    
    results = []
    
    for i, field in enumerate(fields):
        field_name = field.get("field_name", f"Field {i+1}")
        query = field.get("query", "")
        
        if progress_callback:
            progress_callback(i, len(fields), field_name)
        
        try:
            # Retrieve relevant chunks
            retrieved = retriever(query)
            context_text = "\n\n".join([r["chunk"] for r in retrieved])
            
            # Extract with confidence
            extracted_value, confidence, reasoning = extract_field_value(
                client, query, context_text
            )
            
            results.append({
                "field_name": field_name,
                "value": extracted_value,
                "confidence": round(confidence, 1),
                "reasoning": reasoning,
                "query": query
            })
            
            # Rate limiting
            if i < len(fields) - 1:
                time.sleep(delay_seconds)
                
        except Exception as e:
            results.append({
                "field_name": field_name,
                "value": "ERROR",
                "confidence": 0,
                "reasoning": str(e),
                "query": query
            })
            # Longer delay after error
            if i < len(fields) - 1:
                time.sleep(delay_seconds * 2)
    
    return results
