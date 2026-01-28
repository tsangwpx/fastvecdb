"""API routes for documents."""

from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .dependencies import get_db, get_embedding_model, get_user_id
from .config import EMBEDDING_DIMENSION
from .models import Document
from .schemas import DocumentListResponse, DocumentResponse, SearchResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=dict[str, str])
async def create_document(
    user_id: Annotated[str, Depends(get_user_id)],
    db: Annotated[AsyncSession, Depends(get_db)],
    embedding_fn: Annotated[object, Depends(get_embedding_model)],
    document: Annotated[UploadFile | None, File()] = None,
    text: Annotated[str | None, Form()] = None,
    description: Annotated[str | None, Form()] = None,
    vector: Annotated[str | None, Form(alias="_vector")] = None,
):
    """Create a new document."""

    # Validate input
    if document and text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either document or text, not both",
        )

    if not document and not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either document or text is required",
        )

    # Process document or text
    if document:
        content_type = document.content_type
        data = await document.read()
        content_str = data.decode("utf-8", errors="ignore")
    else:
        content_type = "text/plain"
        data = (text or "").encode("utf-8")
        content_str = text or ""

    # Validate content type
    allowed_types = {"text/plain", "text/html"}
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Content type must be one of {allowed_types}",
        )

    # Get embedding
    if vector:
        # Use provided vector for testing (format: comma-separated floats)
        try:
            embedding = [float(v.strip()) for v in vector.split(",")]
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="_vector must be a comma-separated list of floats",
            )
    else:
        # Get embedding using embedding function with content_type and data
        # noinspection PyUnresolvedReference
        embedding: list[float] = await embedding_fn(content_type, data)  # type: ignore

    # Validate embedding dimension
    if len(embedding) != EMBEDDING_DIMENSION:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding dimension mismatch: got {len(embedding)}, expected {EMBEDDING_DIMENSION}",
        )

    # Create document
    doc = Document(
        user_id=user_id,
        content_type=content_type,
        data=data,
        embedding=embedding,
        description=description,
    )

    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    return {"id": doc.id}


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    user_id: Annotated[str, Depends(get_user_id)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """List all documents for the current user."""

    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()

    doc_responses = [
        DocumentResponse(
            id=doc.id,
            link=f"/documents/{doc.id}",
            description=doc.description,
        )
        for doc in documents
    ]

    return DocumentListResponse(documents=doc_responses)


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    user_id: Annotated[str, Depends(get_user_id)],
    db: Annotated[AsyncSession, Depends(get_db)],
    embedding_fn: Annotated[object, Depends(get_embedding_model)],
    query: Annotated[str, Form()],
    limit: Annotated[int, Form()] = 10,
    distance_function: Annotated[str, Form()] = "cosine",
    vector: Annotated[str | None, Form(alias="_vector")] = None,
):
    """Search documents by similarity."""

    # Validate limit
    if limit < 1 or limit > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be between 1 and 50",
        )

    # Validate distance function
    valid_distances = {"cosine", "euclidean", "manhattan"}
    if distance_function not in valid_distances:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"distance_function must be one of {valid_distances}",
        )

    # Get embedding for query
    if vector:
        # Use provided vector for testing (format: comma-separated floats)
        try:
            query_embedding = [float(v.strip()) for v in vector.split(",")]
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="_vector must be a comma-separated list of floats",
            )
    else:
        # Get embedding using embedding function with content_type and data
        # noinspection PyUnresolvedReference
        query_embedding: list[float] = await embedding_fn("text/plain", query)  # type: ignore

    # Validate embedding dimension
    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding dimension mismatch: got {len(query_embedding)}, expected {EMBEDDING_DIMENSION}",
        )

    # Build similarity query based on distance function
    if distance_function == "cosine":
        # Cosine similarity: <=> returns distance, smaller is better
        similarity_score = func.cast(
            Document.embedding.cosine_distance(query_embedding),
            type_=sa.Float,
        )
        order_by_clause = similarity_score.asc()
    elif distance_function == "euclidean":
        # L2 distance
        similarity_score = func.cast(
            Document.embedding.l2_distance(query_embedding),
            type_=sa.Float,
        )
        order_by_clause = similarity_score.asc()
    else:  # manhattan
        # L1 distance
        similarity_score = func.cast(
            Document.embedding.l1_distance(query_embedding),
            type_=sa.Float,
        )
        order_by_clause = similarity_score.asc()

    stmt = (
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(order_by_clause)
        .limit(limit)
    )

    # Query for similar documents
    result = await db.execute(stmt)
    documents = result.scalars().all()

    doc_responses = [
        DocumentResponse(
            id=doc.id,
            link=f"/documents/{doc.id}",
            description=doc.description,
        )
        for doc in documents
    ]

    return SearchResponse(results=doc_responses)


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, str | bytes | None]:
    """Get a document by ID."""

    result = await db.execute(
        select(Document).where(
            and_(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return {
        "id": doc.id,
        "data": doc.data,
        "content_type": doc.content_type,
        "description": doc.description,
    }
