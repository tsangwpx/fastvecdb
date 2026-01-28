"""Test configuration and fixtures."""

import pytest
import pytest_asyncio
import aiohttp
from sqlalchemy.ext.asyncio import create_async_engine

from fastvecdb.db import Base
from fastvecdb.config import DATABASE_URL
from fastvecdb.models import Document  # noqa # required to init metadata


# Base URL for the running uvicorn server
BASE_URL = "http://localhost:8000"


@pytest_asyncio.fixture(loop_scope="function", scope="function")
async def http_client():
    """Create an aiohttp client session for testing and clean database before tests."""
    # Drop all tables before each test
    engine = create_async_engine(DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all, checkfirst=True)
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)
    await engine.dispose()

    # Yield the client for the test
    async with aiohttp.ClientSession() as session:
        yield session

    # No cleanup - leave tables as-is for debugging if tests fail


@pytest.fixture()
def base_url():
    """Return the base URL for API requests."""
    return BASE_URL


@pytest.fixture
def auth_headers():
    """Return authorization headers with test user ID."""
    return {"Authorization": "Bearer test-user-123"}


@pytest.fixture
def auth_headers_user_2():
    """Return authorization headers for a second test user."""
    return {"Authorization": "Bearer test-user-456"}
