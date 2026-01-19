"""
Token Counter Utility
Accurate token counting using tiktoken for LLM and embedding usage tracking
"""

import tiktoken
from typing import List, Dict, Tuple

from config import TIKTOKEN_ENCODING


# Initialize encoder
try:
    _encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
except Exception:
    # Fallback to cl100k_base if config encoding fails
    _encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count tokens in a text string using tiktoken.
    
    Args:
        text: Input text
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    return len(_encoder.encode(text))


def count_tokens_batch(texts: List[str]) -> List[int]:
    """
    Count tokens for multiple texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of token counts
    """
    return [count_tokens(text) for text in texts]


def estimate_embedding_tokens(texts: List[str]) -> int:
    """
    Estimate total tokens for embedding API calls.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        Total embedding tokens
    """
    return sum(count_tokens(text) for text in texts)


def track_llm_usage(prompt: str, response: str) -> Dict[str, int]:
    """
    Track token usage for a single LLM call.
    
    Args:
        prompt: Input prompt
        response: LLM response
        
    Returns:
        Dict with input_tokens, output_tokens, total_tokens
    """
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(response)
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    }


def aggregate_token_usage(usage_list: List[Dict[str, int]]) -> Dict[str, int]:
    """
    Aggregate multiple token usage records.
    
    Args:
        usage_list: List of usage dicts from track_llm_usage
        
    Returns:
        Aggregated usage dict
    """
    total_input = sum(u.get("input_tokens", 0) for u in usage_list)
    total_output = sum(u.get("output_tokens", 0) for u in usage_list)
    
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "num_calls": len(usage_list)
    }


class TokenTracker:
    """
    Track token usage across multiple operations with separation by type.
    """
    
    def __init__(self):
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.embedding_tokens = 0
        self.llm_calls = 0
        self.embedding_calls = 0
    
    def add_llm_usage(self, prompt: str, response: str):
        """Add LLM generation usage."""
        usage = track_llm_usage(prompt, response)
        self.llm_input_tokens += usage["input_tokens"]
        self.llm_output_tokens += usage["output_tokens"]
        self.llm_calls += 1
    
    def add_embedding_usage(self, texts: List[str]):
        """Add embedding usage."""
        tokens = estimate_embedding_tokens(texts)
        self.embedding_tokens += tokens
        self.embedding_calls += 1
    
    def get_summary(self) -> Dict[str, int]:
        """Get complete usage summary."""
        return {
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_total_tokens": self.llm_input_tokens + self.llm_output_tokens,
            "llm_calls": self.llm_calls,
            "embedding_tokens": self.embedding_tokens,
            "embedding_calls": self.embedding_calls,
            "total_tokens": self.llm_input_tokens + self.llm_output_tokens + self.embedding_tokens,
            "total_calls": self.llm_calls + self.embedding_calls
        }
    
    def reset(self):
        """Reset all counters."""
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.embedding_tokens = 0
        self.llm_calls = 0
        self.embedding_calls = 0
