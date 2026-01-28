"""Tests for embedding functionality using _vector parameter."""

import random

import pytest
from aiohttp import FormData

from fastvecdb.config import EMBEDDING_DIMENSION


def create_test_vector(values: list[float]) -> str:
    """Helper to create a vector string with padding to EMBEDDING_DIMENSION."""
    # Pad the provided values with zeros to match EMBEDDING_DIMENSION
    padded = values + [0.0] * (EMBEDDING_DIMENSION - len(values))
    return ",".join(str(v) for v in padded)


@pytest.mark.asyncio
async def test_search_with_fixed_embeddings_ordered_by_distance(
    http_client, base_url, auth_headers
):
    """Test that search returns results ordered by distance.
    
    Given vectors:
    - [0, 1, 0, 0, ...] 
    - [0, 2, 0, 0, ...]
    
    Query [0, 3, 0, 0, ...] should return them in order:
    1. [0, 2, 0, 0, ...] (distance 1)
    2. [0, 1, 0, 0, ...] (distance 2)
    """
    # Create document 1 with vector [0, 1, 0, ...]
    vector_1 = create_test_vector([0.0, 1.0, 0.0])
    form_data_1 = FormData()
    form_data_1.add_field("text", "document one")
    form_data_1.add_field("description", "doc_1")
    form_data_1.add_field("_vector", vector_1)

    async with http_client.post(
        f"{base_url}/documents/",
        data=form_data_1,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200
        doc_id_1 = (await resp.json())["id"]

    # Create document 2 with vector [0, 2, 0, ...]
    vector_2 = create_test_vector([0.0, 2.0, 0.0])
    form_data_2 = FormData()
    form_data_2.add_field("text", "document two")
    form_data_2.add_field("description", "doc_2")
    form_data_2.add_field("_vector", vector_2)

    async with http_client.post(
        f"{base_url}/documents/",
        data=form_data_2,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200
        doc_id_2 = (await resp.json())["id"]

    # Search with query vector [0, 3, 0, ...]
    query_vector = create_test_vector([0.0, 3.0, 0.0])
    search_form = FormData()
    search_form.add_field("query", "search query")
    search_form.add_field("_vector", query_vector)
    search_form.add_field("limit", "10")
    search_form.add_field("distance_function", "euclidean")

    async with http_client.post(
        f"{base_url}/documents/search",
        data=search_form,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        results = data["results"]

        # Should have both documents
        assert len(results) == 2

        # First result should be doc_2 (vector [0, 2, 0]) - distance 1 from query [0, 3, 0]
        # Second result should be doc_1 (vector [0, 1, 0]) - distance 2 from query [0, 3, 0]
        assert results[0]["id"] == doc_id_2
        assert results[1]["id"] == doc_id_1


@pytest.mark.asyncio
async def test_search_with_zero_vectors(http_client, base_url, auth_headers):
    """Test search with zero vectors."""
    # Create document with zero vector
    vector = create_test_vector([0.0, 0.0, 0.0])
    form_data = FormData()
    form_data.add_field("text", "document")
    form_data.add_field("_vector", vector)

    async with http_client.post(
        f"{base_url}/documents/",
        data=form_data,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200
        doc_id = (await resp.json())["id"]

    # Search with zero vector
    search_form = FormData()
    search_form.add_field("query", "search")
    search_form.add_field("_vector", vector)
    search_form.add_field("limit", "10")

    async with http_client.post(
        f"{base_url}/documents/search",
        data=search_form,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == doc_id


@pytest.mark.asyncio
async def test_create_document_with_custom_vector(http_client, base_url, auth_headers):
    """Test creating a document with custom embedding vector."""
    # Create a test vector with correct dimension
    test_vector = [float(i) / EMBEDDING_DIMENSION for i in range(EMBEDDING_DIMENSION)]
    vector_str = ",".join(str(v) for v in test_vector)

    form_data = FormData()
    form_data.add_field("text", "test document")
    form_data.add_field("description", "test description")
    form_data.add_field("_vector", vector_str)

    async with http_client.post(
        f"{base_url}/documents/",
        data=form_data,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert "id" in data
        doc_id = data["id"]

    # Verify the document was created with the custom vector
    async with http_client.get(
        f"{base_url}/documents/{doc_id}",
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200
        doc = await resp.json()
        assert doc["description"] == "test description"


@pytest.mark.asyncio
async def test_vector_dimension_validation(http_client, base_url, auth_headers):
    """Test that incorrect vector dimension is rejected."""
    # Create a vector with wrong dimension
    wrong_dimension = 100
    test_vector = [float(i) / wrong_dimension for i in range(wrong_dimension)]
    vector_str = ",".join(str(v) for v in test_vector)

    form_data = FormData()
    form_data.add_field("text", "test document")
    form_data.add_field("_vector", vector_str)

    async with http_client.post(
        f"{base_url}/documents/",
        data=form_data,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 500
        data = await resp.json()
        assert "Embedding dimension mismatch" in data["detail"]


@pytest.mark.asyncio
async def test_invalid_vector_format(http_client, base_url, auth_headers):
    """Test that invalid vector format is rejected."""
    form_data = FormData()
    form_data.add_field("text", "test document")
    form_data.add_field("_vector", "not,a,valid,vector,format")

    async with http_client.post(
        f"{base_url}/documents/",
        data=form_data,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 400
        data = await resp.json()
        assert "_vector must be a comma-separated list of floats" in data["detail"]


@pytest.mark.asyncio
async def test_search_invalid_vector_format(http_client, base_url, auth_headers):
    """Test that invalid vector format in search is rejected."""
    search_form = FormData()
    search_form.add_field("query", "search")
    search_form.add_field("_vector", "not,valid,numbers")

    async with http_client.post(
        f"{base_url}/documents/search",
        data=search_form,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 400
        data = await resp.json()
        assert "_vector must be a comma-separated list of floats" in data["detail"]


@pytest.mark.asyncio
async def test_vector_dimension_validation_in_search(http_client, base_url, auth_headers):
    """Test that incorrect vector dimension in search is rejected."""
    # Create a vector with wrong dimension
    wrong_dimension = 100
    test_vector = [float(i) / wrong_dimension for i in range(wrong_dimension)]
    vector_str = ",".join(str(v) for v in test_vector)

    search_form = FormData()
    search_form.add_field("query", "search")
    search_form.add_field("_vector", vector_str)

    async with http_client.post(
        f"{base_url}/documents/search",
        data=search_form,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 500
        data = await resp.json()
        assert "Embedding dimension mismatch" in data["detail"]


@pytest.mark.asyncio
async def test_embedding_dimension_constant(http_client, base_url, auth_headers):
    """Test that EMBEDDING_DIMENSION constant is used correctly."""
    # Create a vector with exactly EMBEDDING_DIMENSION elements
    test_vector = list(range(EMBEDDING_DIMENSION))
    vector_str = ",".join(str(float(v)) for v in test_vector)

    form_data = FormData()
    form_data.add_field("text", "test")
    form_data.add_field("_vector", vector_str)

    async with http_client.post(
        f"{base_url}/documents/",
        data=form_data,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200
        assert EMBEDDING_DIMENSION == 768  # Verify constant value


@pytest.mark.asyncio
async def test_vector_with_negative_values(http_client, base_url, auth_headers):
    """Test that vectors with negative values are accepted."""
    # Create a vector with negative values
    test_vector = [float(i - EMBEDDING_DIMENSION / 2) for i in range(EMBEDDING_DIMENSION)]
    vector_str = ",".join(str(v) for v in test_vector)

    form_data = FormData()
    form_data.add_field("text", "test document")
    form_data.add_field("_vector", vector_str)

    async with http_client.post(
        f"{base_url}/documents/",
        data=form_data,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert "id" in data


@pytest.mark.asyncio
async def test_user_isolation_with_custom_vectors(
    http_client, base_url, auth_headers, auth_headers_user_2
):
    """Test that user isolation works correctly with custom vectors."""
    # User 1 creates a document
    test_vector = [float(i) / EMBEDDING_DIMENSION for i in range(EMBEDDING_DIMENSION)]
    vector_str = ",".join(str(v) for v in test_vector)

    form_data = FormData()
    form_data.add_field("text", "user 1 document")
    form_data.add_field("_vector", vector_str)

    async with http_client.post(
        f"{base_url}/documents/",
        data=form_data,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200

    # User 2 searches - should find nothing
    query_vector = create_test_vector([0.0, 1.0, 0.0])
    search_form = FormData()
    search_form.add_field("query", "user 1 document")
    search_form.add_field("_vector", query_vector)
    search_form.add_field("limit", "10")

    async with http_client.post(
        f"{base_url}/documents/search",
        data=search_form,
        headers=auth_headers_user_2,
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert len(data["results"]) == 0


@pytest.mark.asyncio
async def test_large_scale_embedding_search(http_client, base_url, auth_headers):
    """Test similarity search on large scale with 10k documents.
    
    Creates 10k documents with vectors [0, k, 0, ...] where k ranges from 0 to 10000.
    Inserts them in random order. Queries with [0, 5000, ...].
    Expects ~90% of results to be neighbors of 5000 (within +/-5 range).
    
    The vector ID is stored in the description to track which document is which,
    independent of insertion order.
    """
    NUM_DOCUMENTS = 10000
    QUERY_VALUE = 5000
    NEIGHBOR_RANGE = 5  # +/- range from query value
    EXPECTED_ACCURACY = 0.9  # 90% of results should be neighbors
    RESULT_LIMIT = 10

    # Create list of vector indices and shuffle for random insertion
    vector_indices = list(range(NUM_DOCUMENTS))
    random.shuffle(vector_indices)

    # Create documents in random order
    for idx, vector_idx in enumerate(vector_indices):
        # Create vector [0, vector_idx, 0, ...]
        vector = create_test_vector([0.0, float(vector_idx), 0.0])

        form_data = FormData()
        form_data.add_field("text", f"document_{vector_idx}")
        form_data.add_field("description", str(vector_idx))  # Store vector ID in description
        form_data.add_field("_vector", vector)

        async with http_client.post(
            f"{base_url}/documents/",
            data=form_data,
            headers=auth_headers,
        ) as resp:
            assert resp.status == 200

        # Log progress every 1000 documents
        if (idx + 1) % 1000 == 0:
            print(f"Created {idx + 1} documents")

    # Query with [0, 5000, ...]
    query_vector = create_test_vector([0.0, float(QUERY_VALUE), 0.0])
    search_form = FormData()
    search_form.add_field("query", "search query")
    search_form.add_field("_vector", query_vector)
    search_form.add_field("limit", str(RESULT_LIMIT))
    search_form.add_field("distance_function", "euclidean")

    async with http_client.post(
        f"{base_url}/documents/search",
        data=search_form,
        headers=auth_headers,
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        results = data["results"]

        # Should have RESULT_LIMIT results
        assert len(results) == RESULT_LIMIT

        # Extract vector IDs from descriptions
        vector_ids = []
        for result in results:
            # Get the description which contains the vector ID
            doc_id = result["id"]

            # Retrieve full document to get description
            async with http_client.get(
                f"{base_url}/documents/{doc_id}",
                headers=auth_headers,
            ) as get_resp:
                assert get_resp.status == 200
                doc = await get_resp.json()
                vector_id = int(doc["description"])
                vector_ids.append(vector_id)

        # Check that ~90% of results are neighbors
        neighbors = [
            vid
            for vid in vector_ids
            if abs(vid - QUERY_VALUE) <= NEIGHBOR_RANGE
        ]
        accuracy = len(neighbors) / RESULT_LIMIT

        print(f"Query value: {QUERY_VALUE}")
        print(f"Results: {vector_ids}")
        print(f"Neighbors: {neighbors}")
        print(f"Accuracy: {accuracy:.1%}")

        # At least 90% should be neighbors
        assert accuracy >= EXPECTED_ACCURACY, (
            f"Expected at least {EXPECTED_ACCURACY:.0%} neighbors, "
            f"got {accuracy:.1%} ({len(neighbors)}/{RESULT_LIMIT})"
        )
