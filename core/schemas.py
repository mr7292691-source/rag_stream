"""
Pydantic Schemas for LLM Structured Outputs
All LLM responses use these models for type safety and validation
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class FieldExtractionResponse(BaseModel):
    """Response model for field value extraction"""
    value: str = Field(..., description="Extracted field value or 'N/A' if not found")
    confidence: float = Field(..., ge=0, le=100, description="Confidence score 0-100")
    reason: str = Field(..., description="Brief explanation of why this value and confidence")


class FieldDefinition(BaseModel):
    """Model for a single field definition"""
    field_name: str = Field(..., description="Name of the field")
    query: str = Field(..., description="Query to extract this field")
    description: Optional[str] = Field(None, description="Optional description of the field")


class DocumentAnalysisResponse(BaseModel):
    """Response model for document field identification"""
    fields: List[FieldDefinition] = Field(..., description="List of identified fields")
    document_type: Optional[str] = Field(None, description="Type of document identified")


class BenchmarkResult(BaseModel):
    """Model for a single benchmark result"""
    run_number: int
    value: str
    confidence: float
    time_seconds: float


class BenchmarkResponse(BaseModel):
    """Response model for benchmark results"""
    field_name: str
    results: List[BenchmarkResult]
    average_confidence: float
    consistency_score: float


class ZeroShotFieldExtraction(BaseModel):
    """Model for zero-shot extraction of a single field"""
    value: str
    confidence: float
    reason: str


class ZeroShotExtractionResponse(BaseModel):
    """Response model for zero-shot extraction of multiple fields"""
    # Dynamic fields - field_name: extraction_data
    # Using __root__ for dynamic field names
    class Config:
        extra = "allow"  # Allow additional fields
