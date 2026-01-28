"""Database connection and setup."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from fastvecdb.config import DATABASE_URL

# SQLAlchemy engine
engine = None
async_session_factory = None
Base = declarative_base()


async def init_db():
    """Initialize database engine and session factory."""
    global engine, async_session_factory

    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        future=True,
    )

    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)


async def close_db():
    """Close database connection."""
    global engine
    if engine:
        await engine.dispose()
        engine = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    if async_session_factory is None:
        raise RuntimeError("Database not initialized")
    async with async_session_factory() as session:
        yield session
