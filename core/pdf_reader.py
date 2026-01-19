"""
PDF Reader Module
Handles PDF file reading and text extraction
"""

from pypdf import PdfReader


def read_pdf(file) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        file: File-like object or path to PDF file
        
    Returns:
        str: Extracted text content from all pages
    """
    reader = PdfReader(file)
    text = ""
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    return text
