"""
Hallucination Scoring Module
Calculate hallucination scores for extracted values
"""


def calculate_hallucination_score(
    extracted_value: str,
    master_value: str,
    context_text: str
) -> int:
    """
    Calculate hallucination score for a single field.
    Lower score = less hallucination (better).
    
    Args:
        extracted_value: The value extracted by the model
        master_value: The ground truth value
        context_text: The source context text
        
    Returns:
        Hallucination score (0-80, lower is better)
    """
    if extracted_value in ["N/A", "ERROR", ""]:
        return 0  # No hallucination if nothing extracted
    
    extracted_lower = extracted_value.lower().strip()
    master_lower = master_value.lower().strip() if master_value else ""
    
    # Check if extracted value matches master
    if extracted_lower == master_lower:
        return 0  # Perfect match, no hallucination
    
    # Check if extracted value is in the context
    if extracted_lower in context_text.lower():
        return 10  # Value exists in document but differs from master
    
    # Partial match check
    extracted_words = set(extracted_lower.split())
    context_words = set(context_text.lower().split())
    overlap = len(extracted_words & context_words) / len(extracted_words) if extracted_words else 0
    
    if overlap > 0.8:
        return 20  # Most words found in context
    elif overlap > 0.5:
        return 40  # Some words found
    elif overlap > 0.2:
        return 60  # Few words found
    else:
        return 80  # Likely hallucinated
