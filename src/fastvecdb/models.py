"""SQLAlchemy ORM models."""

from uuid import UUID, uuid4

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from .config import EMBEDDING_DIMENSION
from .db import Base


class Document(Base):
    """Document model."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        sa.String(36),
        primary_key=True,
        server_default=func.uuidv7(),
    )
    user_id: Mapped[str] = mapped_column(
        sa.String(255),
        nullable=False,
        index=True,
    )
    content_type: Mapped[str | None] = mapped_column(sa.String(255))
    data: Mapped[bytes | None] = mapped_column(sa.LargeBinary)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True),
        default=sa.func.now(),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} user_id={self.user_id}>"
