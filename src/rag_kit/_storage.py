"""Storage layer — SQLite with FTS5 full-text search (or PostgreSQL).

Uses SQLAlchemy ORM for CRUD. FTS5 is managed via raw SQL through the SQLite
connection, providing BM25 relevance scoring. Falls back to PostgreSQL for
production use (without FTS5).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

# Python 3.11+ has datetime.UTC, but older need timezone.utc
_UTC = timezone.utc
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, relationship

Base = declarative_base()

DEFAULT_DB_PATH = os.path.expanduser("~/.rag-kit/rag.db")

# ── Models ──────────────────────────────────────────────────────────────


class RAGFile(Base):
    """Stores file metadata."""

    __tablename__ = "rag_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(Text, nullable=False, default="default", server_default="default")
    url = Column(Text, nullable=True)
    file_path = Column(Text, nullable=True)
    filename = Column(Text, nullable=True)
    source_type = Column(Text, nullable=True)
    content_hash = Column(Text, nullable=True)
    chunk_size = Column(Integer, nullable=False, default=2500)
    overlap = Column(Integer, nullable=False, default=200)
    total_chunks = Column(Integer, nullable=False, default=0)
    toc = Column(Text, nullable=True)
    section_mappings = Column(Text, nullable=True)  # JSON array of section mappings
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(_UTC))
    last_accessed = Column(DateTime, nullable=False, default=lambda: datetime.now(_UTC))

    chunks = relationship(
        "RAGChunk", back_populates="file", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_rag_files_namespace", namespace),
        Index("idx_rag_files_content_hash", content_hash),
        Index("idx_rag_files_accessed", last_accessed),
        Index("idx_rag_files_name", filename),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.id,
            "namespace": self.namespace,
            "url": self.url,
            "file_path": self.file_path,
            "filename": self.filename,
            "source_type": self.source_type,
            "content_hash": self.content_hash,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "total_chunks": self.total_chunks,
            "toc_present": bool(self.toc),
            "toc_length": len(self.toc) if self.toc else 0,
            "section_mappings_present": bool(self.section_mappings),
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
    keywords_json = Column(Text, nullable=True)
    preview = Column(Text, nullable=True)
    chunk_offset = Column(Integer, nullable=True)

    file = relationship("RAGFile", back_populates="chunks")

    __table_args__ = (
        Index("idx_rag_chunks_file", file_id),
        Index("idx_rag_chunks_file_idx", file_id, chunk_index, unique=True),
    )


# ── FTS5 setup (SQLite only) ──────────────────────────────────────────


def _setup_fts5(engine: Engine):
    """Create FTS5 virtual table and triggers for SQLite."""
    if engine.name != "sqlite":
        return  # FTS5 is SQLite-only

    def _create(conn):
        from sqlalchemy import text

        # Create FTS5 virtual table
        conn.execute(
            text(
                """\
CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
    chunk_text,
    content='rag_chunks',
    content_rowid='id',
    tokenize='porter unicode61'
)"""
            )
        )
        # Triggers to keep FTS index in sync
        conn.execute(
            text(
                """\
CREATE TRIGGER IF NOT EXISTS rag_chunks_ai AFTER INSERT ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rowid, chunk_text)
    VALUES (new.id, new.chunk_text);
END"""
            )
        )
        conn.execute(
            text(
                """\
CREATE TRIGGER IF NOT EXISTS rag_chunks_ad AFTER DELETE ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rag_chunks_fts, rowid, chunk_text)
    VALUES ('delete', old.id, old.chunk_text);
END"""
            )
        )
        conn.execute(
            text(
                """\
CREATE TRIGGER IF NOT EXISTS rag_chunks_au AFTER UPDATE ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rag_chunks_fts, rowid, chunk_text)
    VALUES ('delete', old.id, old.chunk_text);
    INSERT INTO rag_chunks_fts(rowid, chunk_text)
    VALUES (new.id, new.chunk_text);
END"""
            )
        )

    with engine.begin() as conn:
        _create(conn)


# ── Schema Migration ───────────────────────────────────────────────────


def _migrate_schema(engine: Engine):
    """Add new columns to existing databases without data loss.

    Currently migrates: rag_files.section_mappings
    """
    if engine.name != "sqlite":
        return

    def _migrate(conn):
        from sqlalchemy import text

        # Check if section_mappings column exists in rag_files
        result = conn.execute(
            text("PRAGMA table_info(rag_files)")
        ).fetchall()
        columns = {row[1] for row in result}

        if "section_mappings" not in columns:
            conn.execute(
                text(
                    "ALTER TABLE rag_files "
                    "ADD COLUMN section_mappings TEXT DEFAULT NULL"
                )
            )

    with engine.begin() as conn:
        _migrate(conn)


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

    # Migrate existing database: add section_mappings column if missing
    _migrate_schema(engine)

    # Set up FTS5
    _setup_fts5(engine)

    return engine


# ── Public API ──────────────────────────────────────────────────────────


class Storage:
    """Persistence layer wrapping SQLAlchemy sessions + FTS5."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self._engine = _create_engine(db_path)
        self._is_sqlite = self._engine.name == "sqlite"

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
        namespace: str = "default",
        source_type: str | None = None,
        content_hash: str | None = None,
    ) -> int:
        """Insert a new file with chunks in a single transaction.

        Returns the new file_id.
        """
        with self.session() as db:
            rag_file = RAGFile(
                namespace=namespace,
                url=url,
                file_path=file_path,
                filename=filename,
                source_type=source_type,
                content_hash=content_hash,
                chunk_size=chunk_size,
                overlap=overlap,
                total_chunks=total_chunks,
                created_at=datetime.now(_UTC),
                last_accessed=datetime.now(_UTC),
            )
            db.add(rag_file)
            db.flush()

            for i, c in enumerate(chunks):
                kw_json = json.dumps(c.get("keywords_list", []))
                chunk = RAGChunk(
                    file_id=rag_file.id,
                    chunk_index=i,
                    chunk_text=c["text"],
                    keywords=c.get("keywords", ""),
                    keywords_json=kw_json,
                    preview=c.get("preview", ""),
                    chunk_offset=c.get("offset"),
                )
                db.add(chunk)

            db.commit()
            file_id = rag_file.id
        return file_id

    def find_by_hash(self, content_hash: str, namespace: str = "default") -> int | None:
        """Return existing file_id if a file with this hash + namespace exists."""
        with self.session() as db:
            rec = (
                db.query(RAGFile)
                .filter(
                    RAGFile.content_hash == content_hash,
                    RAGFile.namespace == namespace,
                )
                .first()
            )
            return rec.id if rec else None

    def get_file(self, file_id: int) -> dict | None:
        with self.session() as db:
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if not rec:
                return None
            rec.last_accessed = datetime.now(_UTC)
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
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if rec:
                rec.last_accessed = datetime.now(_UTC)
                db.commit()
            kw_list: list[str] = []
            if chunk.keywords_json:
                try:
                    kw_list = json.loads(chunk.keywords_json)
                except (json.JSONDecodeError, TypeError):
                    kw_list = []
            return {
                "index": chunk.chunk_index,
                "text": chunk.chunk_text,
                "keywords": kw_list,
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
                    "id": c.id,
                    "index": c.chunk_index,
                    "text": c.chunk_text,
                    "keywords": (
                        json.loads(c.keywords_json) if c.keywords_json else []
                    ),
                    "preview": c.preview or "",
                }
                for c in chunks
            ]

    def fts5_search(
        self,
        query: str,
        file_id: int | None = None,
        namespace: str | None = None,
        top_k: int = 20,
        chunk_start: int | None = None,
        chunk_end: int | None = None,
    ) -> list[dict]:
        """FTS5 BM25 full-text search.

        Optionally scoped to a chunk index range via chunk_start/chunk_end.

        Only available for SQLite backend. Falls back to linear scan otherwise.
        """
        if not self._is_sqlite:
            if file_id:
                return self.get_all_chunks(file_id)
            return []

        fts_query = _fts5_query_string(query)
        if not fts_query:
            return []

        with self.session() as db:
            from sqlalchemy import text

            sql = """
                SELECT c.file_id, c.chunk_index, c.chunk_text,
                       c.preview, c.id as chunk_id,
                       bm25(rag_chunks_fts) AS score
                FROM rag_chunks_fts
                JOIN rag_chunks c ON c.id = rag_chunks_fts.rowid
                JOIN rag_files f ON f.id = c.file_id
                WHERE rag_chunks_fts MATCH :query
            """
            params: dict[str, Any] = {"query": fts_query, "top_k": top_k}

            if file_id is not None:
                sql += " AND c.file_id = :file_id"
                params["file_id"] = file_id
            elif namespace is not None:
                sql += " AND f.namespace = :namespace"
                params["namespace"] = namespace

            if chunk_start is not None:
                sql += " AND c.chunk_index >= :chunk_start"
                params["chunk_start"] = chunk_start
            if chunk_end is not None:
                sql += " AND c.chunk_index <= :chunk_end"
                params["chunk_end"] = chunk_end

            sql += " ORDER BY score DESC LIMIT :top_k"

            rows = db.execute(text(sql), params).fetchall()
            return [
                {
                    "file_id": row[0],
                    "chunk_index": row[1],
                    "text": row[2],
                    "preview": row[3] or "",
                    "chunk_id": row[4],
                    "score": float(row[5]),
                }
                for row in rows
            ]

    def set_toc(self, file_id: int, toc_text: str) -> bool:
        with self.session() as db:
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if not rec:
                return False
            rec.toc = toc_text
            rec.last_accessed = datetime.now(_UTC)
            db.commit()
            return True

    def get_toc(self, file_id: int) -> str | None:
        with self.session() as db:
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if not rec:
                return None
            rec.last_accessed = datetime.now(_UTC)
            db.commit()
            return rec.toc or ""

    # ── Section Mappings ────────────────────────────────────────────

    def set_section_mappings(self, file_id: int, mappings: list[dict]) -> bool:
        """Store section mappings (heading → chunk range) for a file."""
        with self.session() as db:
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if not rec:
                return False
            rec.section_mappings = json.dumps(mappings, ensure_ascii=False)
            rec.last_accessed = datetime.now(_UTC)
            db.commit()
            return True

    def get_section_mappings(self, file_id: int) -> list[dict]:
        """Retrieve section mappings for a file. Returns [] if none."""
        with self.session() as db:
            rec = db.query(RAGFile).filter(RAGFile.id == file_id).first()
            if not rec:
                return []
            rec.last_accessed = datetime.now(_UTC)
            db.commit()
            if not rec.section_mappings:
                return []
            try:
                return json.loads(rec.section_mappings)
            except (json.JSONDecodeError, TypeError):
                return []

    def get_chunks_by_range(
        self, file_id: int, chunk_start: int, chunk_end: int
    ) -> list[dict]:
        """Get all chunks for a file within a given chunk index range."""
        with self.session() as db:
            chunks = (
                db.query(RAGChunk)
                .filter(
                    RAGChunk.file_id == file_id,
                    RAGChunk.chunk_index >= chunk_start,
                    RAGChunk.chunk_index <= chunk_end,
                )
                .order_by(RAGChunk.chunk_index)
                .all()
            )
            return [
                {
                    "id": c.id,
                    "index": c.chunk_index,
                    "text": c.chunk_text,
                    "keywords": (
                        json.loads(c.keywords_json) if c.keywords_json else []
                    ),
                    "preview": c.preview or "",
                    "offset": c.chunk_offset,
                }
                for c in chunks
            ]

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
        namespace: str | None = None,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "last_accessed",
        descending: bool = True,
    ) -> list[dict]:
        with self.session() as db:
            from sqlalchemy import desc

            col = getattr(RAGFile, order_by, RAGFile.last_accessed)
            q = db.query(RAGFile)
            if namespace is not None:
                q = q.filter(RAGFile.namespace == namespace)
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


# ── Helpers ────────────────────────────────────────────────────────────


def _fts5_query_string(query: str) -> str:
    """Convert a plain text query into an FTS5-safe query string.

    Removes common stop words and uses OR between top meaningful terms.
    BM25 ranking still prioritizes chunks with more matching terms.
    """
    import re

    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "by", "with", "from", "what", "when", "where", "how", "why",
        "who", "which", "this", "that", "these", "those", "it", "its",
        "do", "does", "did", "will", "would", "can", "could", "may",
        "might", "shall", "should", "has", "have", "had", "not", "no",
        "if", "about", "into", "than", "then", "also", "very", "just",
        "each", "all", "any", "both", "some", "such", "only", "need",
        "needed", "before", "after", "during", "other", "more", "most",
        "like", "make", "made", "use", "used", "using", "way", "ways",
        "without", "within", "much", "many", "even", "well", "back",
        "here", "there", "over", "under", "still", "yet", "already",
    }

    # Remove special characters — hyphens confuse FTS5 tokenizer (e.g. "p300-p312")
    cleaned = re.sub(r'[^\w\s]', " ", query)
    terms = [t.strip().lower() for t in cleaned.split() if t.strip()]
    # Remove stop words and single chars
    terms = [t for t in terms if t not in STOP_WORDS and len(t) > 1]

    if not terms:
        return ""

    # Use top 6 terms with OR for broader matching
    # BM25 ranks documents with more matching terms higher
    return " OR ".join(terms[:6])


def compute_content_hash(text: str) -> str:
    """Compute blake3 (or SHA-256 fallback) hash of content."""
    try:
        import blake3

        return blake3.blake3(text.encode("utf-8")).hexdigest()
    except ImportError:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
