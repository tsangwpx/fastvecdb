"""FastAPI main application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from .db import init_db, close_db
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup and shutdown."""
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()


app = FastAPI(
    title="FastVecDB",
    description="Document database with vector similarity search",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routes
app.include_router(router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
