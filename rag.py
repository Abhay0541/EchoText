"""Hardened Notes Search (RAG) module with PDF-to-text, chunking, scoped notes, and Okapi BM25.

Features:
- Upload hardening: allows ``.txt`` and ``.pdf`` only, enforces ``MAX_UPLOAD_BYTES`` size cap,
  strips path traversal (``../``), and sanitizes filenames strictly to ``[A-Za-z0-9_.-]``.
- PDF extraction: converts uploaded PDFs into clean UTF-8 text via ``pypdf``.
- Scoped notes: notes are stored under ``rag_memory/trusted/`` (trusted contacts only),
  ``rag_memory/public/`` (even unknown contacts), ``rag_memory/tiers/<tier>/``, or
  ``rag_memory/contacts/<canonical_name>/`` so private notes are never leaked to strangers.
- Chunking + hand-written Okapi BM25: splits documents into overlapping chunks and ranks the
  top-k (default 3) chunks for each incoming message, returning chunk IDs and snippets for
  display on each Approval Inbox card.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import math
from pathlib import Path
import re
from typing import Any

from config import ALLOWED_UPLOAD_EXTENSIONS, MAX_UPLOAD_BYTES, RAG_DIR, setup_logging
from database import normalize_handle

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore[misc,assignment]

logger = setup_logging("echotext.rag")

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0900-\u097F]{2,}")

STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "is",
        "at",
        "which",
        "on",
        "and",
        "or",
        "to",
        "in",
        "for",
        "of",
        "with",
        "as",
        "by",
        "an",
        "be",
        "this",
        "that",
        "it",
        "from",
        "are",
        "was",
        "were",
    }
)


@dataclass(frozen=True)
class NoteChunk:
    """Represents a single retrievable chunk from a scoped notes file."""

    chunk_id: str
    source_file: str
    scope: str
    text: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_upload_filename(raw_filename: str) -> str:
    """Sanitize a client-supplied filename and enforce the ``.txt`` / ``.pdf`` whitelist."""
    if not raw_filename or not raw_filename.strip():
        raise ValueError("Uploaded filename cannot be empty.")

    if "\x00" in raw_filename:
        raise ValueError("Null byte in filename is not allowed.")

    base = Path(raw_filename.replace("\\", "/")).name.strip()
    if not base or base in (".", "..") or ".." in base:
        raise ValueError(f"Unsafe filename rejected: '{raw_filename}'.")

    suffix = Path(base).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Only .txt and .pdf files are allowed."
        )

    stem = Path(base).stem
    clean_stem = _SAFE_FILENAME_RE.sub("_", stem).strip("._-")
    if not clean_stem:
        clean_stem = "note"
    return f"{clean_stem[:80]}{suffix}"


def extract_text_from_pdf_bytes(raw_bytes: bytes) -> str:
    """Extract UTF-8 text from PDF bytes using ``pypdf``."""
    if PdfReader is None:
        raise RuntimeError("pypdf is required to extract text from PDF uploads.")
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages_text: list[str] = []
    for idx, page in enumerate(reader.pages):
        extracted = (page.extract_text() or "").strip()
        if extracted:
            pages_text.append(f"[Page {idx + 1}]\n{extracted}")
    return "\n\n".join(pages_text).strip()


def resolve_scope_directory(
    base_rag_dir: Path,
    scope_type: str = "trusted",
    scope_target: str = "",
) -> Path:
    """Return the subdirectory inside ``rag_memory`` for a given note scope."""
    stype = scope_type.strip().lower()
    if stype == "contact":
        norm_target = normalize_handle(scope_target)
        return base_rag_dir / "contacts" / norm_target
    if stype == "tier":
        clean_tier = _SAFE_FILENAME_RE.sub("_", scope_target.strip().lower()) or "friend"
        return base_rag_dir / "tiers" / clean_tier
    if stype == "public":
        return base_rag_dir / "public"
    return base_rag_dir / "trusted"


def save_uploaded_note(
    filename: str,
    raw_bytes: bytes,
    *,
    scope_type: str = "trusted",
    scope_target: str = "",
    rag_dir: Path | None = None,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Path:
    """Validate, convert (if PDF), and save an uploaded note inside its scoped directory."""
    if len(raw_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(raw_bytes) > max_bytes:
        raise ValueError(
            f"Uploaded file size ({len(raw_bytes)} bytes) exceeds limit ({max_bytes} bytes)."
        )

    safe_name = sanitize_upload_filename(filename)
    base_dir = (RAG_DIR if rag_dir is None else rag_dir).resolve()
    target_dir = resolve_scope_directory(
        base_dir, scope_type=scope_type, scope_target=scope_target
    ).resolve()

    if base_dir not in target_dir.parents and target_dir != base_dir:
        raise ValueError("Resolved scope directory escaped RAG base directory.")

    target_dir.mkdir(parents=True, exist_ok=True)

    if safe_name.lower().endswith(".pdf"):
        extracted_text = extract_text_from_pdf_bytes(raw_bytes)
        if not extracted_text:
            raise ValueError("Uploaded PDF contained no extractable text.")
        dest_path = (target_dir / f"{safe_name}.txt").resolve()
        dest_path.write_text(extracted_text, encoding="utf-8")
        return dest_path

    try:
        text_content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Uploaded .txt file must be valid UTF-8 text.") from exc

    if not text_content.strip():
        raise ValueError("Uploaded .txt file contains only whitespace.")

    dest_path = (target_dir / safe_name).resolve()
    dest_path.write_text(text_content.strip(), encoding="utf-8")
    return dest_path


def split_text_into_chunks(
    text: str,
    source_file: str,
    scope: str,
    chunk_words: int = 90,
    overlap_words: int = 20,
) -> list[NoteChunk]:
    """Split a note's text into small, overlapping chunks for BM25 retrieval."""
    cleaned = text.strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p.strip()]
    chunks: list[NoteChunk] = []
    chunk_idx = 0

    for para in paragraphs:
        words = para.split()
        if len(words) <= chunk_words:
            chunks.append(
                NoteChunk(
                    chunk_id=f"{source_file}#chunk_{chunk_idx}",
                    source_file=source_file,
                    scope=scope,
                    text=para,
                )
            )
            chunk_idx += 1
        else:
            step = max(1, chunk_words - overlap_words)
            for start in range(0, len(words), step):
                slice_words = words[start : start + chunk_words]
                if not slice_words:
                    break
                chunks.append(
                    NoteChunk(
                        chunk_id=f"{source_file}#chunk_{chunk_idx}",
                        source_file=source_file,
                        scope=scope,
                        text=" ".join(slice_words),
                    )
                )
                chunk_idx += 1
                if start + chunk_words >= len(words):
                    break

    return chunks


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize English, Hinglish, and Devanagari text into lowercase terms."""
    tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
    return [t for t in tokens if t not in STOPWORDS]


def rank_chunks_bm25(
    query: str,
    chunks: list[NoteChunk],
    top_k: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[NoteChunk]:
    """Hand-written Okapi BM25 ranking over ``chunks`` returning the top ``top_k`` matches."""
    if not chunks or top_k <= 0:
        return []

    query_terms = tokenize_for_bm25(query)
    if not query_terms:
        return chunks[:top_k]

    doc_tokens_list: list[list[str]] = [tokenize_for_bm25(c.text) for c in chunks]
    n_docs = len(chunks)
    avgdl = sum(len(dt) for dt in doc_tokens_list) / max(1, n_docs)
    if avgdl == 0:
        return []

    df: dict[str, int] = {}
    for term in set(query_terms):
        df[term] = sum(1 for dt in doc_tokens_list if term in dt)

    scored: list[NoteChunk] = []
    for chunk, doc_tokens in zip(chunks, doc_tokens_list, strict=False):
        if not doc_tokens:
            continue
        dl = len(doc_tokens)
        tf_map: dict[str, int] = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for q_term in query_terms:
            tf = tf_map.get(q_term, 0)
            if tf == 0:
                continue
            n_q = df.get(q_term, 0)
            idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * ((tf * (k1 + 1.0)) / denom)

        if score > 0 and chunk.scope.startswith("contact:"):
            score *= 1.15

        if score > 0:
            scored.append(
                NoteChunk(
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    scope=chunk.scope,
                    text=chunk.text,
                    score=round(score, 4),
                )
            )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


def collect_scoped_chunks(
    contact_handle: str,
    relationship_tier: str = "unknown",
    rag_dir: Path | None = None,
) -> list[NoteChunk]:
    """Load note files eligible for ``(contact_handle, relationship_tier)`` and chunk them."""
    base_dir = (RAG_DIR if rag_dir is None else rag_dir).resolve()
    if not base_dir.exists():
        return []

    norm_handle = normalize_handle(contact_handle)
    tier = relationship_tier.strip().lower()

    eligible_dirs: list[tuple[Path, str]] = [
        (base_dir / "contacts" / norm_handle, f"contact:{norm_handle}"),
        (base_dir / "public", "public"),
    ]
    if tier != "unknown":
        eligible_dirs.append((base_dir / "tiers" / tier, f"tier:{tier}"))
        eligible_dirs.append((base_dir / "trusted", "trusted"))

    all_chunks: list[NoteChunk] = []
    for folder, scope_label in eligible_dirs:
        if not folder.exists() or not folder.is_dir():
            continue
        for file_path in sorted(folder.glob("*.txt")):
            try:
                content = file_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.debug("Skipping unreadable note file %s: %s", file_path, exc)
                continue
            if content:
                all_chunks.extend(
                    split_text_into_chunks(
                        content,
                        source_file=file_path.name,
                        scope=scope_label,
                    )
                )

    if tier != "unknown":
        for file_path in sorted(base_dir.glob("*.txt")):
            try:
                content = file_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.debug("Skipping unreadable root note file %s: %s", file_path, exc)
                continue
            if content:
                all_chunks.extend(
                    split_text_into_chunks(
                        content,
                        source_file=file_path.name,
                        scope="trusted_root",
                    )
                )

    return all_chunks


def retrieve_top_chunks(
    query: str,
    contact_handle: str,
    relationship_tier: str = "unknown",
    top_k: int = 3,
    rag_dir: Path | None = None,
) -> list[NoteChunk]:
    """Retrieve the top ``top_k`` BM25-ranked note chunks scoped to the sender."""
    candidates = collect_scoped_chunks(
        contact_handle=contact_handle,
        relationship_tier=relationship_tier,
        rag_dir=rag_dir,
    )
    return rank_chunks_bm25(query=query, chunks=candidates, top_k=top_k)


def list_all_note_files(rag_dir: Path | None = None) -> list[dict[str, Any]]:
    """List all stored note files across all scope directories for the dashboard."""
    base_dir = (RAG_DIR if rag_dir is None else rag_dir).resolve()
    if not base_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for file_path in sorted(base_dir.rglob("*.txt")):
        try:
            stat = file_path.stat()
            rel = file_path.relative_to(base_dir)
            items.append(
                {
                    "name": file_path.name,
                    "relative_path": str(rel),
                    "scope": str(rel.parent) if str(rel.parent) != "." else "trusted_root",
                    "size_bytes": stat.st_size,
                    "full_path": str(file_path),
                }
            )
        except OSError as exc:
            logger.debug("Skipping note stat for %s: %s", file_path, exc)
    return items
