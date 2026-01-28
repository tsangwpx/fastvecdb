"""Tests for FastAPI document database API using aiohttp."""

import pytest
import aiohttp


# Note: Fixtures are defined in conftest.py
# Run tests with a uvicorn server: uvicorn src.fastvecdb.main:app --host 0.0.0.0 --port 8000


class TestDocumentCreation:
    """Tests for document creation endpoint."""

    @pytest.mark.asyncio
    async def test_create_document_with_text(self, http_client, base_url, auth_headers):
        """Test creating a document with plain text."""
        data = aiohttp.FormData()
        data.add_field("text", "Hello World")
        data.add_field("description", "Test document")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert "id" in body
            assert isinstance(body["id"], str)

    @pytest.mark.asyncio
    async def test_create_document_with_file(self, http_client, base_url, auth_headers):
        """Test creating a document with file upload."""
        data = aiohttp.FormData()
        data.add_field("document", b"File content", filename="test.txt", content_type="text/plain")
        data.add_field("description", "File upload test")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert "id" in body

    @pytest.mark.asyncio
    async def test_create_document_missing_text_and_file(self, http_client, base_url, auth_headers):
        """Test creating document without text or file."""
        data = aiohttp.FormData()
        data.add_field("description", "Test")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 400
            body = await resp.text()
            assert "Either document or text is required" in body

    @pytest.mark.asyncio
    async def test_create_document_both_text_and_file(self, http_client, base_url, auth_headers):
        """Test creating document with both text and file."""
        data = aiohttp.FormData()
        data.add_field("text", "Text content")
        data.add_field("document", b"File content", filename="test.txt", content_type="text/plain")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 400
            body = await resp.text()
            assert "Provide either document or text, not both" in body

    @pytest.mark.asyncio
    async def test_create_document_unsupported_content_type(self, http_client, base_url, auth_headers):
        """Test creating document with unsupported content type."""
        data = aiohttp.FormData()
        data.add_field("document", b"PDF content", filename="test.pdf", content_type="application/pdf")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 400
            body = await resp.text()
            assert "Content type must be one of" in body

    @pytest.mark.asyncio
    async def test_create_document_missing_auth(self, http_client, base_url):
        """Test creating document without authorization."""
        data = aiohttp.FormData()
        data.add_field("text", "Test")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data,
        ) as resp:
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_identical_documents_same_embedding(self, http_client, base_url, auth_headers):
        """Test that identical documents produce the same embedding."""
        text_content = "The quick brown fox jumps over the lazy dog"

        # Create first document
        data1 = aiohttp.FormData()
        data1.add_field("text", text_content)
        data1.add_field("description", "Document 1")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data1,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            doc_id1 = (await resp.json())["id"]

        # Create second document with same content
        data2 = aiohttp.FormData()
        data2.add_field("text", text_content)
        data2.add_field("description", "Document 2")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data2,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            doc_id2 = (await resp.json())["id"]

        # Different IDs but same content
        assert doc_id1 != doc_id2

        # Search for the text - both documents should be returned with same score
        search_data = aiohttp.FormData()
        search_data.add_field("query", text_content)
        search_data.add_field("limit", "10")

        async with http_client.post(
            f"{base_url}/documents/search",
            data=search_data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            results = body["results"]

            # Should find both documents
            assert len(results) == 2
            result_ids = {r["id"] for r in results}
            assert doc_id1 in result_ids
            assert doc_id2 in result_ids


class TestDocumentListing:
    """Tests for document listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, http_client, base_url, auth_headers):
        """Test listing documents when none exist."""
        async with http_client.get(
            f"{base_url}/documents/",
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert body["documents"] == []

    @pytest.mark.asyncio
    async def test_list_documents_with_data(self, http_client, base_url, auth_headers):
        """Test listing documents after creating some."""
        # Create documents
        for i in range(3):
            data = aiohttp.FormData()
            data.add_field("text", f"Document {i}")
            data.add_field("description", f"Desc {i}")

            async with http_client.post(
                f"{base_url}/documents/",
                data=data,
                headers=auth_headers,
            ) as resp:
                assert resp.status == 200

        async with http_client.get(
            f"{base_url}/documents/",
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert len(body["documents"]) == 3

            # Check structure of response
            for doc in body["documents"]:
                assert "id" in doc
                assert "link" in doc
                assert "description" in doc
                assert doc["link"].startswith("/documents/")

    @pytest.mark.asyncio
    async def test_user_isolation(self, http_client, base_url, auth_headers, auth_headers_user_2):
        """Test that users can only see their own documents."""
        # Create document as user 1
        data1 = aiohttp.FormData()
        data1.add_field("text", "User 1 document")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data1,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            doc_id1 = (await resp.json())["id"]

        # Create document as user 2
        data2 = aiohttp.FormData()
        data2.add_field("text", "User 2 document")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data2,
            headers=auth_headers_user_2,
        ) as resp:
            assert resp.status == 200
            doc_id2 = (await resp.json())["id"]

        # User 2 should see only their document
        async with http_client.get(
            f"{base_url}/documents/",
            headers=auth_headers_user_2,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            docs = body["documents"]
            assert len(docs) == 1
            assert docs[0]["id"] == doc_id2

    @pytest.mark.asyncio
    async def test_list_documents_missing_auth(self, http_client, base_url):
        """Test listing documents without authorization."""
        async with http_client.get(f"{base_url}/documents/") as resp:
            assert resp.status == 401


class TestDocumentSearch:
    """Tests for document search endpoint."""

    @pytest.mark.asyncio
    async def test_search_empty_database(self, http_client, base_url, auth_headers):
        """Test search in empty database."""
        data = aiohttp.FormData()
        data.add_field("query", "test")

        async with http_client.post(
            f"{base_url}/documents/search",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert body["results"] == []

    @pytest.mark.asyncio
    async def test_search_with_results(self, http_client, base_url, auth_headers):
        """Test search that finds documents."""
        # Create documents
        docs = [
            "The quick brown fox",
            "A lazy dog",
            "Quick fox jumps",
        ]
        doc_ids = []
        for doc in docs:
            data = aiohttp.FormData()
            data.add_field("text", doc)

            async with http_client.post(
                f"{base_url}/documents/",
                data=data,
                headers=auth_headers,
            ) as resp:
                assert resp.status == 200
                doc_ids.append((await resp.json())["id"])

        # Search for "quick fox"
        search_data = aiohttp.FormData()
        search_data.add_field("query", "quick fox")
        search_data.add_field("limit", "10")

        async with http_client.post(
            f"{base_url}/documents/search",
            data=search_data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert len(body["results"]) == 3

    @pytest.mark.asyncio
    async def test_search_limit(self, http_client, base_url, auth_headers):
        """Test search limit parameter."""
        # Create documents
        for i in range(5):
            data = aiohttp.FormData()
            data.add_field("text", "test document")

            async with http_client.post(
                f"{base_url}/documents/",
                data=data,
                headers=auth_headers,
            ) as resp:
                assert resp.status == 200

        search_data = aiohttp.FormData()
        search_data.add_field("query", "test")
        search_data.add_field("limit", "2")

        async with http_client.post(
            f"{base_url}/documents/search",
            data=search_data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert len(body["results"]) == 2

    @pytest.mark.asyncio
    async def test_search_limit_exceeds_max(self, http_client, base_url, auth_headers):
        """Test that limit cannot exceed 50."""
        data = aiohttp.FormData()
        data.add_field("query", "test")
        data.add_field("limit", "100")

        async with http_client.post(
            f"{base_url}/documents/search",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 400
            body = await resp.text()
            assert "limit must be between 1 and 50" in body

    @pytest.mark.asyncio
    async def test_search_invalid_distance_function(self, http_client, base_url, auth_headers):
        """Test search with invalid distance function."""
        data = aiohttp.FormData()
        data.add_field("query", "test")
        data.add_field("distance_function", "invalid")

        async with http_client.post(
            f"{base_url}/documents/search",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 400
            body = await resp.text()
            assert "distance_function must be one of" in body

    @pytest.mark.asyncio
    async def test_search_different_distance_functions(self, http_client, base_url, auth_headers):
        """Test search with different distance functions."""
        # Create a document
        data = aiohttp.FormData()
        data.add_field("text", "test content")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200

        # Test each distance function
        for dist_fn in ["cosine", "euclidean", "manhattan"]:
            search_data = aiohttp.FormData()
            search_data.add_field("query", "test")
            search_data.add_field("distance_function", dist_fn)

            async with http_client.post(
                f"{base_url}/documents/search",
                data=search_data,
                headers=auth_headers,
            ) as resp:
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_search_user_isolation(self, http_client, base_url, auth_headers, auth_headers_user_2):
        """Test that search only returns user's documents."""
        # Create document as user 1
        data1 = aiohttp.FormData()
        data1.add_field("text", "Test document")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data1,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200

        # Search as user 2 - should not find user 1's document
        search_data = aiohttp.FormData()
        search_data.add_field("query", "test")

        async with http_client.post(
            f"{base_url}/documents/search",
            data=search_data,
            headers=auth_headers_user_2,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert len(body["results"]) == 0

    @pytest.mark.asyncio
    async def test_search_missing_auth(self, http_client, base_url):
        """Test search without authorization."""
        data = aiohttp.FormData()
        data.add_field("query", "test")

        async with http_client.post(
            f"{base_url}/documents/search",
            data=data,
        ) as resp:
            assert resp.status == 401


class TestDocumentRetrieval:
    """Tests for document retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_get_document(self, http_client, base_url, auth_headers):
        """Test retrieving a document by ID."""
        # Create a document
        data = aiohttp.FormData()
        data.add_field("text", "Test content")
        data.add_field("description", "Test doc")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            doc_id = (await resp.json())["id"]

        # Retrieve it
        async with http_client.get(
            f"{base_url}/documents/{doc_id}",
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert body["id"] == doc_id
            assert body["description"] == "Test doc"

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, http_client, base_url, auth_headers):
        """Test retrieving a non-existent document."""
        async with http_client.get(
            f"{base_url}/documents/nonexistent-id",
            headers=auth_headers,
        ) as resp:
            assert resp.status == 404
            body = await resp.text()
            assert "Document not found" in body

    @pytest.mark.asyncio
    async def test_get_other_users_document(self, http_client, base_url, auth_headers, auth_headers_user_2):
        """Test that user cannot retrieve another user's document."""
        # Create document as user 1
        data = aiohttp.FormData()
        data.add_field("text", "Secret document")

        async with http_client.post(
            f"{base_url}/documents/",
            data=data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200
            doc_id = (await resp.json())["id"]

        # Try to retrieve as user 2
        async with http_client.get(
            f"{base_url}/documents/{doc_id}",
            headers=auth_headers_user_2,
        ) as resp:
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_get_document_missing_auth(self, http_client, base_url):
        """Test retrieving document without authorization."""
        async with http_client.get(f"{base_url}/documents/some-id") as resp:
            assert resp.status == 401


class TestHealth:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, http_client, base_url):
        """Test health check endpoint."""
        async with http_client.get(f"{base_url}/health") as resp:
            assert resp.status == 200
            body = await resp.json()
            assert body["status"] == "ok"
