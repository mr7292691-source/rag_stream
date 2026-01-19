"""
UI - Styles Module
CSS styles and HTML templates for the UI
"""

# PDF Preview container style
PDF_PREVIEW_STYLE = """
<div style="
    height:500px;
    overflow-y:scroll;
    border:1px solid #ddd;
    border-radius:8px;
    padding:16px;
    background-color:#f8f9fa;
    color:#000000;
    font-size:13px;
    line-height:1.6;
    font-family: 'Courier New', monospace;
    white-space: pre-wrap;
">
{content}
</div>
"""

# PDF iframe embed style
PDF_IFRAME_STYLE = """
<iframe 
    src="data:application/pdf;base64,{pdf_base64}" 
    width="100%" 
    height="600px" 
    type="application/pdf"
    style="border:1px solid #ddd; border-radius:8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
></iframe>
"""

# Text highlight style
HIGHLIGHT_STYLE = """
<mark style="
    background-color:#ffeb3b;
    color:#000000;
    padding:2px 4px;
    border-radius:4px;
    font-weight:600;
">{text}</mark>
"""


def get_styles() -> str:
    """
    Get custom CSS styles for the application.
    
    Returns:
        CSS string to inject into the page
    """
    return """
    <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Card styling */
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e9ecef;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 4px;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
        }
        
        /* DataFrame styling */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #4CAF50;
        }
        
        /* Success/Error message styling */
        .stSuccess, .stError, .stWarning, .stInfo {
            border-radius: 8px;
        }
    </style>
    """


def get_confidence_color(confidence: float) -> str:
    """
    Get background color based on confidence level.
    
    Args:
        confidence: Confidence percentage (0-100)
        
    Returns:
        CSS color string
    """
    if confidence >= 70:
        return "#d4edda"  # Green
    elif confidence >= 50:
        return "#fff3cd"  # Yellow
    else:
        return "#f8d7da"  # Red


def get_match_icon(match_type: str) -> str:
    """
    Get icon for match type.
    
    Args:
        match_type: 'exact', 'partial', or 'mismatch'
        
    Returns:
        Emoji icon string
    """
    icons = {
        "exact": "✅",
        "partial": "🟡",
        "fuzzy": "🟠",
        "mismatch": "❌",
        "N/A": "⚪"
    }
    return icons.get(match_type, "❓")


def get_metric_color(value: float, thresholds: tuple) -> str:
    """
    Get color indicator based on value and thresholds.
    
    Args:
        value: The metric value
        thresholds: Tuple of (low, high) thresholds
        
    Returns:
        Emoji color indicator
    """
    low, high = thresholds
    if value >= high:
        return "🟢"
    elif value >= low:
        return "🟡"
    else:
        return "🔴"
