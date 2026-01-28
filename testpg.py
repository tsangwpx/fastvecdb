import asyncio
import time
import uuid
import random
import numpy as np
from uuid import uuid4
from typing import List

import sqlalchemy as sa
from sqlalchemy import Index
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

# ==========================================
# CONFIGURATION
# ==========================================
DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"

VECTOR_COUNT = 100_000  # Total vectors to insert
USER_COUNT = 1_000  # Total users
DIMENSION = 768  # Vector size
BATCH_SIZE = 1_000  # UPDATED: Batch size set to 1000

if VECTOR_COUNT < USER_COUNT:
    raise ValueError("vector_count must be >= user_count")

# ==========================================
# DATABASE SETUP & MODEL
# ==========================================


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Document model with HNSW Index."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        sa.String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    # Standard B-Tree index for exact matching on user_id
    user_id: Mapped[str] = mapped_column(sa.String(255), nullable=False, index=True)

    content_type: Mapped[str | None] = mapped_column(sa.String(255))
    data: Mapped[bytes | None] = mapped_column(sa.LargeBinary)
    embedding: Mapped[List[float]] = mapped_column(Vector(DIMENSION), nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True),
        default=sa.func.now(),
    )

    # UPDATED: HNSW Index Definition
    __table_args__ = (
        Index(
            "ix_documents_embedding_hnsw",  # Index Name
            "embedding",  # Column
            postgresql_using="hnsw",  # Index Method
            postgresql_with={"m": 16, "ef_construction": 64},  # HNSW params
            postgresql_ops={
                "embedding": "vector_l2_ops"
            },  # Operator (must match query type)
        ),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} user_id={self.user_id}>"


# ==========================================
# DATA GENERATION
# ==========================================


def generate_batch(batch_size: int, dimension: int, start_idx: int, total_users: int):
    """Generates a batch of data using Numpy."""
    embeddings = np.random.rand(batch_size, dimension).astype(np.float32)
    batch_data = []

    for i in range(batch_size):
        global_index = start_idx + i
        user_index = global_index % total_users

        batch_data.append(
            {
                "id": str(uuid.uuid4()),
                "user_id": f"user_{user_index}",
                "content_type": "text/plain",
                "data": b"dummy_data",
                "embedding": embeddings[i].tolist(),
                "description": f"Generated document {global_index}",
            }
        )
    return batch_data


# ==========================================
# OPERATIONS
# ==========================================


async def setup_database(engine):
    """Drops and creates tables (including Indexes)."""
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        # This will create the table AND the HNSW index immediately
        await conn.run_sync(Base.metadata.create_all)
        print("└── Database schema and HNSW index initialized.")


async def insert_vectors(session_maker):
    """Inserts vectors in batches."""
    print(
        f"\n[Insertion] Starting insert of {VECTOR_COUNT} vectors (Batch: {BATCH_SIZE})..."
    )
    print("            Note: Insertion will be slower due to live HNSW indexing.")
    start_time = time.perf_counter()

    total_inserted = 0

    for i in range(0, VECTOR_COUNT, BATCH_SIZE):
        current_batch_size = min(BATCH_SIZE, VECTOR_COUNT - i)
        batch_data = generate_batch(current_batch_size, DIMENSION, i, USER_COUNT)

        async with session_maker() as session:
            stmt = sa.insert(Document).values(batch_data)
            await session.execute(stmt)
            await session.commit()

        total_inserted += current_batch_size
        print(f"   Processed {total_inserted}/{VECTOR_COUNT}...", end="\r")

    duration = time.perf_counter() - start_time
    print(
        f"\n[Insertion] Completed in {duration:.2f}s ({VECTOR_COUNT / duration:.0f} vec/s)"
    )


async def query_vectors(session_maker):
    """Performs sample similarity searches."""
    print(f"\n[Query] Running sample queries...")

    # 1. Generate query vector
    query_vec = np.random.rand(DIMENSION).astype(np.float32).tolist()
    # 2. Pick random user
    target_user = f"user_{random.randint(0, USER_COUNT - 1)}"

    async with session_maker() as session:
        start_time = time.perf_counter()

        # PostgreSQL Optimizer should now use:
        # 1. B-Tree scan on user_id
        # 2. HNSW scan on embedding
        stmt = (
            sa.select(Document)
            .filter(Document.user_id == target_user)
            .order_by(Document.embedding.l2_distance(query_vec))
            .limit(5)
        )
        print(stmt)

        result = await session.execute(stmt)
        docs = result.scalars().all()

        duration = time.perf_counter() - start_time

        print(f"└── Query for {target_user} took {duration:.4f}s")
        print(f"└── Found {len(docs)} results.")

        # Verify execution plan (Optional, useful for debugging indexes)
        # explain_stmt = sa.text(f"EXPLAIN ANALYZE SELECT * FROM documents WHERE user_id = '{target_user}' ORDER BY embedding <-> '{query_vec}' LIMIT 5")
        # explain = await session.execute(explain_stmt)
        # print("\n[Explain Analyze Snippet]:", explain.scalars().first())


async def main():
    engine = create_async_engine(DB_URL, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    try:
        await setup_database(engine)
        await insert_vectors(async_session)
        await query_vectors(async_session)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
