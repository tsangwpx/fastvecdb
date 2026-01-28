"""FastAPI dependencies."""

from collections.abc import Callable, Awaitable
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session
from .embedding import get_embedding

get_auth_header = HTTPBearer()


async def get_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(get_auth_header)],
) -> str:
    """
    Extract user_id from Authorization header using HTTPBearer.

    Expected format: Bearer <user-id>
    For now, we trust the ID directly without validation.
    """
    if not credentials or not hasattr(credentials, "credentials"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = credentials.credentials

    if not 4 <= len(user_id) <= 256:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id


async def get_db(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AsyncSession:
    """Dependency to get database session."""
    return session


async def get_embedding_model() -> Callable[
    [str | None, bytes | str],
    Awaitable[list[float]],
]:
    """Dependency to get embedding function."""
    return get_embedding
