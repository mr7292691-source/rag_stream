"""
LLM Module - LiteLLM with Pydantic
Centralized LLM access with cost tracking
Edit config.py for API keys and provider settings
"""

import os
import time
import re
import json
from typing import List, Optional, Dict, Any, Type, TypeVar

import litellm
from pydantic import BaseModel

from config import (
    LLM_PROVIDER,
    API_KEYS,
    LLM_MODELS,
    LITELLM_TIMEOUT,
    LITELLM_MAX_RETRIES,
    LITELLM_DROP_PARAMS,
    TOKEN_COSTS,
    DEFAULT_RETRY_COUNT
)

# Configure LiteLLM
litellm.drop_params = LITELLM_DROP_PARAMS
litellm.set_verbose = False  # Set to True for debugging

T = TypeVar('T', bound=BaseModel)


# ============================================================================
# COST CALCULATION
# ============================================================================

def calculate_cost(
    provider: str,
    input_tokens: int,
    output_tokens: int,
    embedding_tokens: int = 0
) -> Dict[str, float]:
    """
    Calculate cost for token usage.
    
    Args:
        provider: LLM provider name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        embedding_tokens: Number of embedding tokens
        
    Returns:
        Dict with cost breakdown and total
    """
    costs = TOKEN_COSTS.get(provider, {})
    
    input_cost = (input_tokens / 1_000_000) * costs.get("input", 0.0)
    output_cost = (output_tokens / 1_000_000) * costs.get("output", 0.0)
    embedding_cost = (embedding_tokens / 1_000_000) * costs.get("embedding", 0.0)
    
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "embedding_cost": round(embedding_cost, 6),
        "total_cost": round(input_cost + output_cost + embedding_cost, 6),
        "currency": "USD"
    }


# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

def create_client(api_key: str = None):
    """
    Initialize LiteLLM client.
    Sets API key from config if not provided.
    
    Args:
        api_key: Optional API key (uses config if not provided)
        
    Returns:
        None (LiteLLM uses environment variables)
    """
    # Set API key in environment for LiteLLM
    key = api_key or API_KEYS.get(LLM_PROVIDER)
    
    if not key or key.startswith("YOUR_"):
        raise ValueError(f"Please set your {LLM_PROVIDER.upper()} API key in config.py")
    
    # Set environment variable based on provider
    if LLM_PROVIDER == "gemini":
        os.environ["GEMINI_API_KEY"] = key
    elif LLM_PROVIDER == "openai":
        os.environ["OPENAI_API_KEY"] = key
    elif LLM_PROVIDER == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = key
    
    return None  # LiteLLM doesn't need a client object


# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

def get_embeddings(
    client,  # Unused, kept for compatibility
    texts: List[str],
    batch_size: int = 50
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using LiteLLM.
    
    Args:
        client: Unused (kept for backward compatibility)
        texts: List of text strings to embed
        batch_size: Number of texts per API call
        
    Returns:
        List of embedding vectors
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        response = litellm.embedding(
            model=LLM_MODELS["embedding"],
            input=batch,
            timeout=LITELLM_TIMEOUT
        )
        
        all_embeddings.extend([e['embedding'] for e in response.data])
    
    return all_embeddings


def get_single_embedding(client, text: str) -> List[float]:
    """
    Generate embedding for a single text.
    
    Args:
        client: Unused (kept for backward compatibility)
        text: Text string to embed
        
    Returns:
        Embedding vector
    """
    embeddings = get_embeddings(client, [text])
    return embeddings[0] if embeddings else []


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_text(
    client,  # Unused, kept for compatibility
    prompt: str,
    model: str = None,
    use_lite_model: bool = False,
    system_message: str = None
) -> str:
    """
    Generate text response from LLM.
    
    Args:
        client: Unused (kept for backward compatibility)
        prompt: The prompt to send
        model: Specific model to use (overrides use_lite_model)
        use_lite_model: If True, use the lite/fast model
        system_message: Optional system message
        
    Returns:
        Generated text response
    """
    if model is None:
        model = LLM_MODELS["generation_lite"] if use_lite_model else LLM_MODELS["generation"]
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    response = litellm.completion(
        model=model,
        messages=messages,
        timeout=LITELLM_TIMEOUT
    )
    
    return response.choices[0].message.content.strip()


def generate_json(
    client,  # Unused, kept for compatibility
    prompt: str,
    model: str = None,
    use_lite_model: bool = False,
    system_message: str = None
) -> Optional[Dict[str, Any]]:
    """
    Generate and parse JSON response from LLM.
    
    Args:
        client: Unused (kept for backward compatibility)
        prompt: The prompt to send (should request JSON output)
        model: Specific model to use
        use_lite_model: If True, use the lite/fast model
        system_message: Optional system message
        
    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    response_text = generate_text(client, prompt, model, use_lite_model, system_message)
    return parse_json_response(response_text)


def generate_with_pydantic(
    client,  # Unused, kept for compatibility
    prompt: str,
    response_model: Type[T],
    model: str = None,
    use_lite_model: bool = False,
    system_message: str = None
) -> T:
    """
    Generate structured response using Pydantic model.
    
    Args:
        client: Unused (kept for backward compatibility)
        prompt: The prompt to send
        response_model: Pydantic model class for response
        model: Specific model to use
        use_lite_model: If True, use the lite/fast model
        system_message: Optional system message
        
    Returns:
        Validated Pydantic model instance
    """
    if model is None:
        model = LLM_MODELS["generation_lite"] if use_lite_model else LLM_MODELS["generation"]
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Add schema instruction to prompt
    schema_prompt = f"""{prompt}

Return ONLY a valid JSON object matching this schema:
{response_model.model_json_schema()}

JSON only:"""
    
    messages.append({"role": "user", "content": schema_prompt})
    
    response = litellm.completion(
        model=model,
        messages=messages,
        timeout=LITELLM_TIMEOUT,
        response_format={"type": "json_object"}  # Request JSON mode
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Parse and validate with Pydantic
    return response_model.model_validate_json(response_text)


def generate_with_retry(
    client,  # Unused, kept for compatibility
    prompt: str,
    model: str = None,
    use_lite_model: bool = False,
    retry_count: int = DEFAULT_RETRY_COUNT,
    system_message: str = None
) -> str:
    """
    Generate text with automatic retry on errors.
    
    Args:
        client: Unused (kept for backward compatibility)
        prompt: The prompt to send
        model: Specific model to use
        use_lite_model: If True, use the lite/fast model
        retry_count: Number of retries on failure
        system_message: Optional system message
        
    Returns:
        Generated text response
        
    Raises:
        Exception: If all retries fail
    """
    if model is None:
        model = LLM_MODELS["generation_lite"] if use_lite_model else LLM_MODELS["generation"]
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    for attempt in range(retry_count):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                timeout=LITELLM_TIMEOUT,
                num_retries=LITELLM_MAX_RETRIES
            )
            return response.choices[0].message.content.strip()
            
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
    client,  # Unused, kept for compatibility
    prompt: str,
    model: str = None,
    use_lite_model: bool = False,
    retry_count: int = DEFAULT_RETRY_COUNT,
    system_message: str = None
) -> Optional[Dict[str, Any]]:
    """
    Generate and parse JSON with automatic retry.
    
    Args:
        client: Unused (kept for backward compatibility)
        prompt: The prompt to send (should request JSON output)
        model: Specific model to use
        use_lite_model: If True, use the lite/fast model
        retry_count: Number of retries on failure
        system_message: Optional system message
        
    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    response_text = generate_with_retry(client, prompt, model, use_lite_model, retry_count, system_message)
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
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    
    text = text.strip()
    
    try:
        return json.loads(text)
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
    return LLM_MODELS["generation_lite"] if use_lite else LLM_MODELS["generation"]
