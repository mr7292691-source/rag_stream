"""
LLM Module - TCS Version
Centralized functions for TCS BFSI GenAI client and LLM usage
All LLM-related changes should be made in this file only.

To switch to TCS version:
1. Rename this file to llm.py (backup the original as llm_personal.py)
2. Or update imports in other files to use llm_tcs instead of llm
"""

import time
import re
import json
from typing import List, Optional, Dict, Any, Tuple

from langchain_tcs_bfsi_genai import APIClient, Auth, TCSLLMs

from config import DEFAULT_RETRY_COUNT


# ============================================================================
# DEFAULT MODEL CONFIGURATION (can be overridden)
# ============================================================================

DEFAULT_MODEL = "gpt-4o-mini"
LITE_MODEL = "gpt-4o-mini"  # Can use a lighter model if available


# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

def create_client(api_key: str = None):
    """
    Create and return TCS BFSI GenAI client with authentication.
    
    Args:
        api_key: Not used for TCS (uses internal auth), kept for compatibility
        
    Returns:
        Tuple of (client, llm) for TCS usage
    """
    client = APIClient()
    auth = Auth(client)
    llm = TCSLLMs(client=client, model_name=DEFAULT_MODEL)
    return llm  # Return LLM directly for compatibility


def create_tcs_client(model_name: str = None):
    """
    Create TCS client with specific model.
    
    Args:
        model_name: Model to use (default: gpt-4o-mini)
        
    Returns:
        Configured TCSLLMs instance
    """
    client = APIClient()
    auth = Auth(client)
    model = model_name or DEFAULT_MODEL
    return TCSLLMs(client=client, model_name=model)


# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

def get_embeddings(
    client,
    texts: List[str],
    batch_size: int = 50
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    Note: TCS may use different embedding approach - adjust as needed.
    
    Args:
        client: TCS LLM client
        texts: List of text strings to embed
        batch_size: Number of texts per API call
        
    Returns:
        List of embedding vectors
    """
    # TCS embedding implementation - adjust based on available APIs
    # This is a placeholder - replace with actual TCS embedding call
    all_embeddings = []
    
    for text in texts:
        # Use LLM to generate a pseudo-embedding or call TCS embedding API
        # Replace with actual TCS embedding API when available
        embedding = [0.0] * 768  # Placeholder
        all_embeddings.append(embedding)
    
    return all_embeddings


def get_single_embedding(client, text: str) -> List[float]:
    """
    Generate embedding for a single text.
    
    Args:
        client: TCS LLM client
        text: Text string to embed
        
    Returns:
        Embedding vector
    """
    embeddings = get_embeddings(client, [text])
    return embeddings[0] if embeddings else [0.0] * 768


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_text(
    client,
    prompt: str,
    model: str = None,
    use_lite_model: bool = False,
    system_message: str = None
) -> str:
    """
    Generate text response from TCS LLM.
    
    Args:
        client: TCS LLM client (TCSLLMs instance)
        prompt: The prompt to send
        model: Not used (model set at client creation)
        use_lite_model: Not used for TCS
        system_message: Optional system message
        
    Returns:
        Generated text response
    """
    system_msg = system_message or "You are a helpful document analysis assistant."
    
    messages = [
        ("system", system_msg),
        ("human", prompt),
    ]
    
    response = client.invoke(messages)
    
    # Handle different response formats
    if hasattr(response, 'content'):
        return response.content.strip()
    elif isinstance(response, str):
        return response.strip()
    else:
        return str(response).strip()


def generate_json(
    client,
    prompt: str,
    model: str = None,
    use_lite_model: bool = False,
    system_message: str = None
) -> Optional[Dict[str, Any]]:
    """
    Generate and parse JSON response from TCS LLM.
    
    Args:
        client: TCS LLM client
        prompt: The prompt to send (should request JSON output)
        model: Not used
        use_lite_model: Not used
        system_message: Optional system message
        
    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    json_system = system_message or "You are a helpful assistant. Return responses in valid JSON format only."
    response_text = generate_text(client, prompt, system_message=json_system)
    return parse_json_response(response_text)


def generate_with_retry(
    client,
    prompt: str,
    model: str = None,
    use_lite_model: bool = False,
    retry_count: int = DEFAULT_RETRY_COUNT,
    system_message: str = None
) -> str:
    """
    Generate text with automatic retry on errors.
    
    Args:
        client: TCS LLM client
        prompt: The prompt to send
        model: Not used
        use_lite_model: Not used
        retry_count: Number of retries on failure
        system_message: Optional system message
        
    Returns:
        Generated text response
        
    Raises:
        Exception: If all retries fail
    """
    system_msg = system_message or "You are a helpful document analysis assistant."
    
    messages = [
        ("system", system_msg),
        ("human", prompt),
    ]
    
    for attempt in range(retry_count):
        try:
            response = client.invoke(messages)
            
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for rate limit errors
            if "429" in error_msg or "rate" in error_msg.lower():
                if attempt < retry_count - 1:
                    time.sleep(min((attempt + 1) * 5, 60))
                    continue
                raise RuntimeError("Rate limit exceeded. Please try again later.")
            
            if attempt < retry_count - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
            else:
                raise e
    
    raise RuntimeError("Generation failed after retries")


def generate_json_with_retry(
    client,
    prompt: str,
    model: str = None,
    use_lite_model: bool = False,
    retry_count: int = DEFAULT_RETRY_COUNT,
    system_message: str = None
) -> Optional[Dict[str, Any]]:
    """
    Generate and parse JSON with automatic retry.
    
    Args:
        client: TCS LLM client
        prompt: The prompt to send (should request JSON output)
        model: Not used
        use_lite_model: Not used
        retry_count: Number of retries on failure
        system_message: Optional system message
        
    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    json_system = system_message or "You are a helpful assistant. Return responses in valid JSON format only."
    response_text = generate_with_retry(client, prompt, retry_count=retry_count, system_message=json_system)
    return parse_json_response(response_text)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    text = response_text.strip()
    
    # Clean up markdown code blocks if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def get_model_name(use_lite: bool = False) -> str:
    """
    Get the appropriate model name.
    
    Args:
        use_lite: If True, return lite model name
        
    Returns:
        Model name string
    """
    return LITE_MODEL if use_lite else DEFAULT_MODEL
