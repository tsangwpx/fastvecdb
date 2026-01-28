"""Guide for running the tests"""

# Testing with aiohttp against live uvicorn server

## Setup

1. Install test dependencies (pytest and aiohttp should be in your project):
   ```bash
   pip install pytest pytest-asyncio aiohttp
   ```

## Running Tests

You need to run the tests against a live uvicorn server. In one terminal/window:

```bash
# Terminal 1: Start the uvicorn server
uvicorn src.fastvecdb.main:app --host 0.0.0.0 --port 8000
```

Then in another terminal:

```bash
# Terminal 2: Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run specific test class
pytest tests/test_api.py::TestDocumentCreation -v

# Run specific test
pytest tests/test_api.py::TestDocumentCreation::test_create_document_with_text -v

# Run with coverage
pytest tests/ --cov=src/fastvecdb -v
```

## Important Notes

- The tests connect to `http://localhost:8000` (configurable in conftest.py)
- Each test uses Bearer token authentication with different user IDs
- Tests are isolated by user ID (user-123, user-456, etc.)
- Database must be running and accessible for the server
- The test "test_identical_documents_same_embedding" verifies deterministic embeddings
