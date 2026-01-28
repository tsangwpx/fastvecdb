"""Pydantic request/response schemas."""

from pydantic import BaseModel


class DocumentCreateRequest(BaseModel):
    """Request schema for creating a document."""
    
    description: str | None = None


class DocumentResponse(BaseModel):
    """Response schema for a document."""
    
    id: str
    link: str
    description: str | None


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    
    documents: list[DocumentResponse]


class SearchRequest(BaseModel):
    """Request schema for searching documents."""
    
    limit: int = 10
    distance_function: str = "cosine"


class SearchResponse(BaseModel):
    """Response schema for search results."""
    
    results: list[DocumentResponse]
