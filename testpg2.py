import asyncio
import time
import uuid
import random
import numpy as np
from uuid import uuid4
from typing import List

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

# ==========================================
# CONFIGURATION
# ==========================================
DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"

USER_COUNT = 100  # Total unique users
ITEM_COUNT = 10_000  # Total items (documents)
EMBEDDINGS_PER_ITEM = 3  # Chunks per item (Total vectors = ITEM_COUNT * 3)
DIMENSION = 768
BATCH_SIZE = 2_000

# ==========================================
# DATABASE MODELS
# ==========================================


class Base(DeclarativeBase):
    pass


class Item(Base):
    """
    The main document/item table.
    """

    __tablename__ = "items"

    id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    user_id: Mapped[str] = mapped_column(sa.String(255), nullable=False, index=True)
    content: Mapped[str] = mapped_column(sa.Text, default="Some content...")
    created_at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True), default=sa.func.now()
    )


class Embedding(Base):
    """
    The vector table.
    One Item -> Many Embeddings.
    """

    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # Foreign Key to Item
    item_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("items.id"), nullable=False, index=True
    )

    # We include user_id here to avoid JOINs during vector search filtering
    user_id: Mapped[str] = mapped_column(sa.String(255), nullable=False, index=True)

    # The Vector
    embedding: Mapped[List[float]] = mapped_column(Vector(DIMENSION), nullable=False)

    # Optional: HNSW Index (Commented out to show baseline performance first)
    # __table_args__ = (
    #     sa.Index("ix_embeddings_hnsw", "embedding", postgresql_using="hnsw",
    #              postgresql_with={"m": 16, "ef_construction": 64},
    #              postgresql_ops={"embedding": "vector_l2_ops"}),
    # )


# ==========================================
# DATA GENERATION
# ==========================================


async def generate_and_insert_data(session_maker):
    """
    Generates items first, then generates multiple embeddings per item.
    """
    print(
        f"--- Generating Data: {ITEM_COUNT} Items, {ITEM_COUNT * EMBEDDINGS_PER_ITEM} Vectors ---"
    )

    total_items_inserted = 0
    total_vectors_inserted = 0

    # 1. Generate Items
    item_ids_map = []  # Keep track of IDs to link embeddings later

    for i in range(0, ITEM_COUNT, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, ITEM_COUNT - i)
        item_batch = []

        for k in range(batch_size):
            u_id = f"user_{random.randint(0, USER_COUNT - 1)}"
            i_id = uuid.uuid4()
            item_batch.append(
                {"id": i_id, "user_id": u_id, "content": f"Item {i + k} content"}
            )
            item_ids_map.append((i_id, u_id))

        async with session_maker() as session:
            await session.execute(sa.insert(Item), item_batch)
            await session.commit()

        total_items_inserted += batch_size
        print(f"Items Inserted: {total_items_inserted}/{ITEM_COUNT}", end="\r")

    print("\nItems done. Generating Embeddings...")

    # 2. Generate Embeddings (linked to Items)
    # We loop through our cached item IDs
    current_batch = []

    for idx, (i_id, u_id) in enumerate(item_ids_map):
        # Create N embeddings for this single item
        vectors = np.random.rand(EMBEDDINGS_PER_ITEM, DIMENSION).astype(np.float32)

        for v in range(EMBEDDINGS_PER_ITEM):
            current_batch.append(
                {
                    "id": uuid.uuid4(),
                    "item_id": i_id,
                    "user_id": u_id,  # Copy user_id to embedding table
                    "embedding": vectors[v].tolist(),
                }
            )

        # Flush batch if full
        if len(current_batch) >= BATCH_SIZE:
            async with session_maker() as session:
                await session.execute(sa.insert(Embedding), current_batch)
                await session.commit()
                total_vectors_inserted += len(current_batch)
                print(
                    f"Vectors Inserted: {total_vectors_inserted}/{ITEM_COUNT * EMBEDDINGS_PER_ITEM}",
                    end="\r",
                )
            current_batch = []

    # Flush remaining
    if current_batch:
        async with session_maker() as session:
            await session.execute(sa.insert(Embedding), current_batch)
            await session.commit()

    print(f"\nData Generation Complete.")


# ==========================================
# EXPLAIN ANALYZE TEST
# ==========================================


async def analyze_queries(session_maker):
    print("\n==========================================")
    print("RUNNING EXPLAIN ANALYZE")
    print("==========================================")

    target_user = "user_1"
    query_vec = np.random.rand(DIMENSION).astype(np.float32).tolist()

    async with session_maker() as session:
        # -------------------------------------------------------
        # QUERY 1: The Optimized Query (Filter Embeddings directly)
        # -------------------------------------------------------
        print(f"\n[Test 1] Filter by User ID on EMBEDDINGS table -> Sort by Distance")
        sql_1 = sa.text(
            f"""
            EXPLAIN ANALYZE
            SELECT id, item_id, embedding <-> '[:vec]' as dist
            FROM embeddings
            WHERE user_id = :uid
            ORDER BY dist
            LIMIT 5;
        """.replace("[:vec]", str(query_vec))
        )  # Injecting vector literal for cleaner explain output

        result = await session.execute(sql_1, {"uid": target_user})
        print("PLAN:")
        for row in result.scalars():
            print("  " + row)

        # -------------------------------------------------------
        # QUERY 2: The Expensive Join (Filter Item -> Join Embedding)
        # -------------------------------------------------------
        print(f"\n[Test 2] Filter ITEMS table -> Join Embeddings -> Sort")
        print(
            "         (This simulates if you didn't have user_id on embeddings table)"
        )
        sql_2 = sa.text(
            f"""
            EXPLAIN ANALYZE
            SELECT e.id, e.item_id, e.embedding <-> '[:vec]' as dist
            FROM items i
            JOIN embeddings e ON i.id = e.item_id
            WHERE i.user_id = :uid
            ORDER BY dist
            LIMIT 5;
        """.replace("[:vec]", str(query_vec))
        )

        result = await session.execute(sql_2, {"uid": target_user})
        print("PLAN:")
        for row in result.scalars():
            print("  " + row)


# ==========================================
# MAIN
# ==========================================


async def main():
    engine = create_async_engine(DB_URL, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    await generate_and_insert_data(async_session)
    await analyze_queries(async_session)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
