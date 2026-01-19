"""

Document Analyzer Module
AI-powered automatic field identification from documents
"""

import json
from typing import List, Dict, Optional

from config import GENERATION_MODEL, MAX_ANALYSIS_SAMPLE


def analyze_document(
    client,
    document_text: str,
    sample_length: int = MAX_ANALYSIS_SAMPLE
) -> Optional[List[Dict]]:
    """
    Analyze document to identify all key fields using LLM.
    
    Args:
        client: Gemini API client
        document_text: Full document text
        sample_length: Length of document sample to analyze
        
    Returns:
        List of field dicts with 'field_name' and 'query', or None on failure
    """
    # Get a sample of the document
    sample_text = document_text[:sample_length]
    
    analysis_prompt = f"""You are a document analysis expert. Analyze this document and identify ALL key fields that should be extracted.

Document Sample:
{sample_text}

Instructions:
1. Identify all important fields (dates, amounts, names, addresses, IDs, etc.)
2. Return ONLY a JSON array of field objects
3. Each field should have: "field_name" (descriptive name) and "query" (question to extract it)
4. Be comprehensive - include 10-20 fields depending on document type
5. Format: [{{"field_name": "Invoice Date", "query": "What is the invoice date?"}}, ...]

Return ONLY the JSON array, no other text."""

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=analysis_prompt
        )
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        fields = json.loads(response_text)
        return fields
        
    except Exception as e:
        raise RuntimeError(f"Document analysis failed: {e}")
