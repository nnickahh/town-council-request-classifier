"""Reads data/docs/*.md|txt, runs Two-Stage Semantic Chunking, and saves the hybrid RAG index.

Re-run manually (`python ingest.py`) after bulk document edits, or call `add_document(...)`
from the staff dashboard — the running bot also auto-detects file updates and hot-reindexes
with zero downtime.

Document format:
    # Title
    Updated: 2026-03
    ...body...
"""

from __future__ import annotations

from pathlib import Path

import config
import rag

_BASE_DIR = Path(__file__).resolve().parent


def _docs_dir() -> Path:
    p = Path(config.DOCS_DIR)
    return p if p.is_absolute() else _BASE_DIR / p


def add_document(title: str, updated: str, body: str, stem: str) -> int:
    """Writes a new knowledge-base document to disk and hot-reindexes the Semantic RAG store."""
    docs_dir = _docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem).strip("_") or "staff_answer"
    out_path = docs_dir / f"{safe_stem}.md"
    out_path.write_text(f"# {title}\nUpdated: {updated}\n\n{body.strip()}\n", encoding="utf-8")
    return rag.get_engine().reindex()


def add_pdf(pdf_path: Path | str, engine: str = "auto") -> tuple[Path, int]:
    """Converts a PDF handbook into Markdown inside data/docs/ and hot-reindexes RAG."""
    import pdf_converter

    src = Path(pdf_path)
    out_md = _docs_dir() / f"{src.stem.lower().replace(' ', '_')}.md"
    pdf_converter.convert_pdf_to_markdown(src, engine=engine, output_path=out_md)
    chunks = rag.get_engine().reindex()
    return out_md, chunks


def convert_pending_pdfs() -> int:
    """Converts any .pdf files placed inside data/docs/ into .md files before indexing."""
    import pdf_converter

    converted = 0
    for pdf_file in sorted(_docs_dir().glob("*.pdf")):
        target_md = pdf_file.with_suffix(".md")
        if not target_md.exists() or pdf_file.stat().st_mtime > target_md.stat().st_mtime:
            res = pdf_converter.convert_pdf_to_markdown(pdf_file, output_path=target_md)
            print(
                f"Converted PDF '{pdf_file.name}' -> '{target_md.name}' "
                f"({res.page_count}p, {res.word_count} words, engine={res.engine_used})"
            )
            converted += 1
    return converted


def main() -> None:
    convert_pending_pdfs()
    engine = rag.get_engine()
    chunk_count = engine.reindex()
    docs = sorted(p for p in _docs_dir().glob("*") if p.suffix.lower() in {".md", ".txt"} and p.is_file())
    if not docs or chunk_count == 0:
        raise SystemExit(f"No documents found in {_docs_dir()}")
    print(
        f"Indexed {chunk_count} semantic chunks from {len(docs)} documents "
        f"(backend: {engine.embedding_provider.backend_name})."
    )
    print(f"Saved hybrid index cache to: {engine.index_path}")


if __name__ == "__main__":
    main()
