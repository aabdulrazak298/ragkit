"""Storage layer — SQLite (default) or PostgreSQL via SQLAlchemy."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship

Base = declarative_base()

DEFAULT_DB_PATH = os.path.expanduser("~/.rag-kit/rag.db")

# ── Models ──────────────────────────────────────────────────────────────


class RAGFile(Base):
    """Stores file metadata."""

    __tablename__ = "rag_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(Text, nullable=True)
    file_path = Column(Text, nullable=True)
    filename = Column(Text, nullable=True)
    chunk_size = Column(Integer, nullable=False, default=2500)
    overlap = Column(Integer, nullable=False, default=200)
    total_chunks = Column(Integer, nullable=False, default=0)
    toc = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_accessed = Column(DateTime, nullable=False, default=datetime.utcnow)

    chunks = relationship(
        "RAGChunk", back_populates="file", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_rag_files_accessed", last_accessed),
        Index("idx_rag_files_name", filename),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.id,
            "url": self.url,
            "file_path": self.file_path,
            "filename": self.filename,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "total_chunks": self.total_chunks,
            "toc_present": bool(self.toc),
            "toc_length": len(self.toc) if self.toc else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
        }


class RAGChunk(Base):
    """Stores individual text chunks."""

    __tablename__ = "rag_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(
        Integer, ForeignKey("rag_files.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    keywords = Column(Text, nullable=True)
    preview = Column(Text, nullable=True)

    file = relationship("RAGFile", back_populates="chunks")

    __table_args__ = (
        Index("idx_rag_chunks_file", file_id),
        Index("idx_rag_chunks_file_idx", file_id, chunk_index, unique=True),
    )


# ── Engine helpers ─────────────────────────────────────────────────────


def _create_engine(db_path: str):
    """Create SQLAlchemy engine from path (SQLite or PostgreSQL)."""
    if db_path.startswith("postgresql"):
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise ImportError(
                "Install rag-kit[postgres] for PostgreSQL support: "
                "pip install 'rag-kit[postgres]'"
            )
    else:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db_path = f"sqlite:///{db_path}"

    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)
    return engine


# ── Public API ──────────────────────────────────────────────────────────


class Storage:
    """Persistence layer wrapping SQLAlchemy sessions."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self._engine = _create_engine(db_path)

    def session(self) -> Session:
        return Session(self._engine)

    # ── CRUD ─────────────────────────────────────────────────────────

    def create_file(
        self,
        url: str | None,
        file_path: str | None,
        filename: str,
        chunk_size: int,
        overlap: int,
        total_chunks: int,
        chunks: list[dict],
    ) -> RAGFile:
        """Insert a new file with chunks in a single transaction."""
        with self.session() as db:
            rag_file = RAGFile(
                url=url,
                file_path=file_path,
                filename=filename,
                chunk_size=chunk_size,
                overlap=overlap,
                total_chunks=total_chunks,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
            )
            db.add(rag_file)
            db.flush()

            for i, c in enumerate(chunks):
                chunk = RAGChunk(
                    file_id=rag_file.id,
                    chunk_index=i,
                    chunk_text=c["text"],
                    keywords=c.get("keywords", ""),
                    preview=c.get("preview", ""),
                )
                db.add(chunk)

            db.commit()
            file_id = rag_file.id
        return file_id

    def get_file(self, file_id: int) -> dict | None:
        with self.session() as db:
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if not rec:
                return None
            rec.last_accessed = datetime.utcnow()
            db.commit()
            return rec.to_dict()

    def get_chunk(self, file_id: int, index: int) -> dict | None:
        with self.session() as db:
            chunk = (
                db.query(RAGChunk)
                .filter(
                    RAGChunk.file_id == file_id, RAGChunk.chunk_index == index
                )
                .first()
            )
            if not chunk:
                return None
            # Touch file access time
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if rec:
                rec.last_accessed = datetime.utcnow()
                db.commit()
            return {
                "index": chunk.chunk_index,
                "text": chunk.chunk_text,
                "keywords": (
                    chunk.keywords.split(", ") if chunk.keywords else []
                ),
            }

    def get_all_chunks(self, file_id: int) -> list[dict]:
        with self.session() as db:
            chunks = (
                db.query(RAGChunk)
                .filter(RAGChunk.file_id == file_id)
                .order_by(RAGChunk.chunk_index)
                .all()
            )
            return [
                {
                    "index": c.chunk_index,
                    "text": c.chunk_text,
                    "keywords": (
                        c.keywords.split(", ") if c.keywords else []
                    ),
                    "preview": c.preview or "",
                }
                for c in chunks
            ]

    def search(self, file_id: int, query: str, threshold: float = 0.6) -> list[dict]:
        """Fetch all chunks for a file and filter by fuzzy match score."""
        chunks = self.get_all_chunks(file_id)
        # Use _search module for scoring — called externally
        return chunks  # raw list for the search module to score

    def set_toc(self, file_id: int, toc_text: str) -> bool:
        with self.session() as db:
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if not rec:
                return False
            rec.toc = toc_text
            rec.last_accessed = datetime.utcnow()
            db.commit()
            return True

    def get_toc(self, file_id: int) -> str | None:
        with self.session() as db:
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if not rec:
                return None
            rec.last_accessed = datetime.utcnow()
            db.commit()
            return rec.toc or ""

    def delete_file(self, file_id: int) -> bool:
        with self.session() as db:
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if not rec:
                return False
            db.delete(rec)
            db.commit()
            return True

    def list_files(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "last_accessed",
        descending: bool = True,
    ) -> list[dict]:
        with self.session() as db:
            from sqlalchemy import desc

            col = getattr(RAGFile, order_by, RAGFile.last_accessed)
            q = db.query(RAGFile)
            if descending:
                q = q.order_by(desc(col))
            else:
                q = q.order_by(col)
            return [r.to_dict() for r in q.offset(skip).limit(limit).all()]

    def stats(self) -> dict:
        with self.session() as db:
            total_files = db.query(RAGFile).count()
            total_chunks = db.query(RAGChunk).count()
            return {
                "total_files": total_files,
                "total_chunks": total_chunks,
            }
