"""
Utils - Text Highlighting
Utilities for highlighting extracted text in documents
"""

import re
from html import escape


def highlight_text(full_text: str, extracted_values: list) -> str:
    """
    Highlight extracted values in the full text with HTML markup.
    
    Args:
        full_text: The complete document text
        extracted_values: List of values to highlight
        
    Returns:
        HTML string with highlighted values
    """
    safe = escape(full_text)
    
    for val in extracted_values:
        if not val or not val.strip():
            continue
        
        pattern = re.escape(val.strip())
        safe = re.sub(
            pattern,
            f"""<mark style="
                background-color:#ffeb3b;
                color:#000000;
                padding:2px 4px;
                border-radius:4px;
                font-weight:600;
            ">{val}</mark>""",
            safe,
            flags=re.IGNORECASE
        )
    
    return safe
