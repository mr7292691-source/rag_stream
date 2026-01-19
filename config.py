"""
Configuration settings for RAG PDF Field Extractor
ALL LLM SETTINGS ARE HERE - Edit this file only for API keys and provider settings
"""

# ============================================================================
# LLM PROVIDER CONFIGURATION - EDIT HERE ONLY
# ============================================================================

# Provider: "gemini", "openai", "anthropic", "azure", etc.
LLM_PROVIDER = "gemini"

# API Keys - Add your keys here
API_KEYS = {
    "gemini": "YOUR_GEMINI_API_KEY_HERE",
    "openai": "YOUR_OPENAI_API_KEY_HERE",
    "anthropic": "YOUR_ANTHROPIC_API_KEY_HERE",
}

# Model mapping - provider/model-name format for LiteLLM
LLM_MODELS = {
    "generation": "gemini/gemini-2.5-flash",
    "generation_lite": "gemini/gemini-2.5-flash-lite",
    "embedding": "text-embedding-004",  # Gemini embedding model
}

# Alternative models for other providers (uncomment to use)
# LLM_MODELS = {
#     "generation": "gpt-4o-mini",
#     "generation_lite": "gpt-4o-mini",
#     "embedding": "text-embedding-3-small",
# }

# LiteLLM settings
LITELLM_TIMEOUT = 60  # seconds
LITELLM_MAX_RETRIES = 3
LITELLM_DROP_PARAMS = True  # Drop unsupported params for compatibility

# ============================================================================
# COST CALCULATION - Update pricing per 1M tokens
# ============================================================================

# Pricing per 1M tokens (USD) - Update these based on your provider
# Set to 0 if you want to calculate manually
TOKEN_COSTS = {
    "gemini": {
        "input": 0.0,   # $0.075 per 1M for gemini-2.5-flash
        "output": 0.0,  # $0.30 per 1M for gemini-2.5-flash
        "embedding": 0.0,  # Free for Gemini embeddings
    },
    "openai": {
        "input": 0.0,   # $0.15 per 1M for gpt-4o-mini
        "output": 0.0,  # $0.60 per 1M for gpt-4o-mini
        "embedding": 0.0,  # $0.02 per 1M for text-embedding-3-small
    },
    "anthropic": {
        "input": 0.0,   # Update with Claude pricing
        "output": 0.0,
        "embedding": 0.0,
    },
}

# ============================================================================
# MODEL CONFIGURATION (Legacy - for reference)
# ============================================================================

# Embedding model
EMBEDDING_MODEL = "gemini-embedding-001"

# Generation models
GENERATION_MODEL = "gemini-2.5-flash"
LITE_GENERATION_MODEL = "gemini-2.5-flash-lite"

# ============================================================================
# CHUNKING DEFAULTS
# ============================================================================

DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 20
DEFAULT_TOP_K = 5

CHUNKING_ALGORITHMS = ["Sliding Window", "Recursive"]
CHUNKING_MODES = ["token", "sentence", "paragraph"]

# ============================================================================
# API SETTINGS
# ============================================================================

DEFAULT_API_DELAY = 2.0
MAX_API_DELAY = 5.0
MIN_API_DELAY = 0.5

DEFAULT_BENCHMARK_RUNS = 3
MAX_BENCHMARK_RUNS = 10

DEFAULT_RETRY_COUNT = 3

# Token estimation multiplier (for rough token count from word count)
TOKEN_MULTIPLIER = 1.3

# Maximum document length for zero-shot extraction
MAX_DOCUMENT_LENGTH = 15000

# Maximum sample length for document analysis
MAX_ANALYSIS_SAMPLE = 3000

# ============================================================================
# UI CONFIGURATION
# ============================================================================

PAGE_TITLE = "RAG PDF Field Extractor"
PAGE_ICON = "📄"
LAYOUT = "wide"

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 70
MEDIUM_CONFIDENCE_THRESHOLD = 50

# Hallucination thresholds
LOW_HALLUCINATION_THRESHOLD = 20
MEDIUM_HALLUCINATION_THRESHOLD = 40

# PDF size threshold for embedded preview (in MB)
PDF_EMBED_SIZE_THRESHOLD_MB = 5.0

# ============================================================================
# ENCODING
# ============================================================================

TIKTOKEN_ENCODING = "cl100k_base"
