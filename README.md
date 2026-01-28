# FastVecDB

An experimental, async document database API for research and prototyping with vector similarity search built on FastAPI, PostgreSQL, and pgvector.

WARNING: This is an experimental project. The embedding model integration is incomplete — the repository currently uses a placeholder embedding function for testing and development. Do not rely on this for production use.

## Overview

FastVecDB provides a RESTful API for storing documents with embeddings and performing similarity searches. It's designed for high concurrency with async/await throughout, user-isolated data, and multiple distance functions for similarity computation.

### Key Features

- **Async-first design**: Built with FastAPI and SQLAlchemy async for high concurrency
- **Vector embeddings**: Store and search documents using 768-dimensional vectors
- **Multiple distance functions**: Cosine, Euclidean, and Manhattan distance support
- **User isolation**: Automatic query filtering ensures users only access their own documents
- **Bearer token authentication**: Uses FastAPI's HTTPBearer security scheme
- **Flexible content storage**: Support for plain text and HTML documents, plus file uploads
- **Deterministic embeddings**: Placeholder model provides consistent, reproducible embeddings for testing
- **Testing infrastructure**: Comprehensive test suite with aiohttp against live uvicorn server

## Architecture

### Stack

- **FastAPI** (v0.128.0+): Web framework with async support
- **PostgreSQL + asyncpg**: Async database driver with pgvector extension
- **SQLAlchemy 2.0+**: Async ORM with declarative models
- **Python 3.14**: Modern type annotations and language features

### Project Structure

```
fastvecdb/
├── src/fastvecdb/
│   ├── config.py          # Configuration (DATABASE_URL, EMBEDDING_DIMENSION)
│   ├── db.py              # Database initialization and session management
│   ├── models.py          # SQLAlchemy ORM models
│   ├── schemas.py         # Pydantic request/response models
│   ├── embedding.py       # Embedding model placeholder
│   ├── dependencies.py    # FastAPI dependency injectors
│   ├── routes.py          # API endpoints
│   ├── main.py            # FastAPI application entry point
│   └── __init__.py
├── tests/
│   ├── conftest.py        # Pytest fixtures and configuration
│   ├── test_api.py        # Main API endpoint tests
│   └── test_embeddings.py # Embedding search tests
├── pg/                    # PostgreSQL Docker configuration
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.14+
- PostgreSQL with pgvector extension
- Docker (optional, for PostgreSQL container)

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd fastvecdb
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -e ".[dev]"
# or with uv:
uv sync
```

4. **Set up PostgreSQL**

Using Docker:
```bash
cd pg
./run.sh  # Creates and starts postgres container with pgvector
```

Or use existing PostgreSQL with pgvector extension installed.

5. **Configure environment** (optional)

```bash
export DATABASE_URL="postgresql+asyncpg://user:password@localhost/dbname"
export EMBEDDING_DIMENSION="768"
```

Default values:
- `DATABASE_URL`: `postgresql+asyncpg://postgres:postgres@localhost/postgres`
- `EMBEDDING_DIMENSION`: `768`

## Running

### Start the API server

```bash
uvicorn src.fastvecdb.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### Authentication

All endpoints require Bearer token authentication:

```bash
curl -H "Authorization: Bearer <user-id>" http://localhost:8000/documents/
```

The `<user-id>` is any string that identifies the user. It's used to isolate data between users.

### 1. Create Document

**POST** `/documents/`

Create a new document with embedding.

**Parameters:**
- `text` (form, optional): Plain text content
- `document` (file, optional): File upload (text/plain or text/html)
- `description` (form, optional): Human-readable description
- `_vector` (form, optional): Custom embedding vector (comma-separated floats, testing only)

**Note**: Either `text` or `document` is required, but not both.

**Example:**

```bash
# From text
curl -X POST http://localhost:8000/documents/ \
  -H "Authorization: Bearer user1" \
  -F "text=The quick brown fox" \
  -F "description=Quick fox story"

# From file
curl -X POST http://localhost:8000/documents/ \
  -H "Authorization: Bearer user1" \
  -F "document=@file.txt" \
  -F "description=File upload"

# With custom vector (testing)
curl -X POST http://localhost:8000/documents/ \
  -H "Authorization: Bearer user1" \
  -F "text=test" \
  -F "_vector=0.1,0.2,0.3,..."
```

**Response:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 2. List Documents

**GET** `/documents/`

List all documents for the current user.

**Response:**

```json
{
  "documents": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "link": "/documents/550e8400-e29b-41d4-a716-446655440000",
      "description": "Quick fox story"
    }
  ]
}
```

### 3. Search Documents

**POST** `/documents/search`

Search for similar documents using vector similarity.

**Parameters:**
- `query` (form): Text to embed and search for
- `limit` (form, default=10): Max results (1-50)
- `distance_function` (form, default="cosine"): One of `cosine`, `euclidean`, `manhattan`
- `_vector` (form, optional): Custom search vector (testing only)

**Example:**

```bash
curl -X POST http://localhost:8000/documents/search \
  -H "Authorization: Bearer user1" \
  -F "query=brown fox" \
  -F "limit=10" \
  -F "distance_function=cosine"
```

**Response:**

```json
{
  "results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "link": "/documents/550e8400-e29b-41d4-a716-446655440000",
      "description": "Quick fox story"
    }
  ]
}
```

### 4. Get Document

**GET** `/documents/{document_id}`

Retrieve a specific document (only owner can access).

**Response:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "data": "<base64-encoded-content>",
  "content_type": "text/plain",
  "description": "Quick fox story"
}
```

## Database Schema

### Document Model

```python
class Document(Base):
    __tablename__ = "documents"
    
    id: str                          # UUIDv7
    user_id: str                     # For user isolation
    content_type: str                # MIME type (text/plain, text/html)
    data: bytes                      # Document content
    embedding: list[float]           # 768-dimensional vector
    description: str | None          # Optional description
    created_at: datetime             # Creation timestamp
```

**Indexes:**
- `user_id, created_at` for efficient listing
- `user_id, embedding` for similarity search

## Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run specific test file

```bash
pytest tests/test_api.py -v
pytest tests/test_embeddings.py -v
```

### Run specific test

```bash
pytest tests/test_embeddings.py::test_search_with_fixed_embeddings_ordered_by_distance -v
```

### Test Coverage

The test suite includes:

1. **API Endpoint Tests** (`test_api.py`):
   - Document creation with text and file uploads
   - Document listing with pagination
   - Similarity search with different distance functions
   - Document retrieval with ownership validation
   - User isolation enforcement
   - Authentication and error handling
   - Edge cases and validation

2. **Embedding Tests** (`test_embeddings.py`):
   - Custom vector parameter (`_vector`) for testing
   - Vector dimension validation
   - Invalid vector format handling
   - Simple embedding ordering tests
   - Large-scale search with 10k documents
   - ~90% accuracy validation for neighbor finding
   - User isolation with custom vectors

### Running Tests Against Live Server

Tests use aiohttp to connect to a live uvicorn server:

```bash
# Terminal 1: Start server
uvicorn src.fastvecdb.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Run tests
pytest tests/ -v
```

## Configuration

All configuration is in `src/fastvecdb/config.py`:

```python
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost/postgres",
)

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
```

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string (default: local postgres)
- `EMBEDDING_DIMENSION`: Vector dimension (default: 768)

## Embedding Model (Work In Progress)

Embedding integration is not yet complete. Currently the project uses a deterministic placeholder embedding function (based on SHA256 hashing) to generate reproducible vectors for development and testing only. This placeholder is intentionally simple and is not a substitute for a real embedding model.

Planned next steps:
- Integrate a real embedding model (e.g., OpenAI, Hugging Face) with configurable providers
- Add async batching, rate limits, and model selection

### Custom Vectors for Testing

While the real embedding model is pending, you can supply custom embeddings via the `_vector` form parameter to exercise and validate search behaviors:

```bash
curl -X POST http://localhost:8000/documents/ \
  -H "Authorization: Bearer user1" \
  -F "text=test" \
  -F "_vector=0.1,0.2,0.3,..."
```

This is intended for development and testing only — it allows:
- Exercising search logic without a live model
- Validating similarity ranking with controlled vectors
- Reproducing test scenarios deterministically

## Security

### Authentication

- Uses FastAPI's `HTTPBearer` security scheme
- Bearer tokens identify users (no validation)
- All requests must include `Authorization: Bearer <user-id>` header
- Missing authentication returns `401 Unauthorized`

### User Isolation

- All database queries automatically filtered by `user_id`
- Users cannot access, modify, or delete other users' documents
- Search results only include user's own documents

### Content Type Validation

- Only `text/plain` and `text/html` are accepted
- Invalid content types return `400 Bad Request`

## Performance Considerations

- **Async throughout**: All I/O is non-blocking
- **Connection pooling**: SQLAlchemy manages async session pool
- **Vector indexing**: pgvector provides efficient similarity search
- **User filtering**: Indexed on (user_id, embedding) for fast searches
- **Pagination**: List endpoint orders by creation time for consistent pagination

## Development

### Code Style

Uses Ruff for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Type Annotations

Modern Python 3.14+ type hints throughout:

```python
# Union types
value: str | None

# Type aliases
Vector: TypeAlias = list[float]
```

### Adding New Features

1. Define schema in `schemas.py`
2. Add database model to `models.py`
3. Implement route in `routes.py`
4. Add tests in `tests/`

## Troubleshooting

### PostgreSQL Connection Issues

```
psycopg2.OperationalError: could not connect to server
```

Check:
- PostgreSQL is running
- DATABASE_URL is correct
- pgvector extension is installed: `CREATE EXTENSION IF NOT EXISTS vector;`

### Vector Dimension Mismatch

```
500 Internal Server Error: Embedding dimension mismatch
```

Ensure:
- EMBEDDING_DIMENSION matches vector sizes
- Custom `_vector` parameters have correct dimensions
- Embedding model returns vectors of correct size

### Authentication Failures

```
401 Unauthorized
```

Check:
- Include `Authorization: Bearer <user-id>` header
- User ID is any non-empty string

## Future Enhancements

- Integrate real embedding models (OpenAI, Hugging Face, etc.)
- Add metadata indexing and filtering
- Implement document versioning
- Add bulk operations (batch create, delete)
- Rate limiting and quota management
- Webhook support for async operations
- GraphQL interface
- Multi-tenancy with API keys

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Check existing tests for usage examples
- Review API documentation at `/docs`
