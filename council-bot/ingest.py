"""Reads data/docs/*.md|txt, chunks them, embeds them into Chroma. Re-run after changing documents.

Document format:
    # Title
    Updated: 2026-03
    ...body...
"""
from pathlib import Path

import config
import rag


def chunk(text: str, size: int = 900, overlap: int = 150):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + p  # small overlap keeps context across chunks
        else:
            current = (current + "\n\n" + p) if current else p
    if current:
        chunks.append(current)
    return chunks


def parse(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    title, updated, body_start = path.stem, "unknown", 0
    if lines and lines[0].startswith("# "):
        title, body_start = lines[0][2:].strip(), 1
    if len(lines) > body_start and lines[body_start].lower().startswith("updated:"):
        updated = lines[body_start].split(":", 1)[1].strip()
        body_start += 1
    return title, updated, "\n".join(lines[body_start:])


def add_document(title: str, updated: str, body: str, stem: str):
    """Add one document without rebuilding. Also saved to disk so a full re-ingest keeps it."""
    Path(config.DOCS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.DOCS_DIR, f"{stem}.md").write_text(f"# {title}\nUpdated: {updated}\n\n{body}\n", encoding="utf-8")
    col = rag.get_collection()
    for i, c in enumerate(chunk(body)):
        col.upsert(
            ids=[f"{stem}-{i}"],
            documents=[c],
            metadatas=[{"title": title, "updated": updated, "file": f"{stem}.md"}],
        )


def main():
    try:
        rag._chroma.delete_collection("council_docs")  # rebuild from scratch
    except Exception:
        pass
    col = rag.get_collection()
    ids, docs, metas = [], [], []
    for path in sorted(Path(config.DOCS_DIR).glob("*")):
        if path.suffix not in {".md", ".txt"}:
            continue
        title, updated, body = parse(path)
        for i, c in enumerate(chunk(body)):
            ids.append(f"{path.stem}-{i}")
            docs.append(c)
            metas.append({"title": title, "updated": updated, "file": path.name})
    if not ids:
        raise SystemExit(f"No documents found in {config.DOCS_DIR}")
    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Ingested {len(ids)} chunks from {len(set(m['file'] for m in metas))} documents.")


if __name__ == "__main__":
    main()
