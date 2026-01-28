import os

# Database connection string
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost/postgres",
)

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

assert EMBEDDING_DIMENSION >= 1, EMBEDDING_DIMENSION
