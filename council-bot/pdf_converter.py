"""Lossless PDF-to-Markdown Ingestion Pipeline for Singapore Town Council Handbooks.

Converts Town Council PDF circulars, by-laws, and handbooks into clean
GitHub-Flavored Markdown (.md) files ready for Semantic Hybrid RAG indexing:
1. Fast local parsing via PyMuPDF4LLM (if installed) or lightweight pure-Python `pypdf`.
2. Multimodal Gemini Vision OCR fallback for scanned / image-heavy PDFs.
3. Layout noise cleanup (running page numbers, repeated headers, soft hyphens)
   and sentence-boundary healing across page breaks while preserving tables.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("council-bot.pdf_converter")


@dataclass
class PDFConversionResult:
    source_path: str
    markdown_text: str
    page_count: int
    table_count: int
    word_count: int
    headings: list[dict[str, Any]] = field(default_factory=list)
    pages: list[dict[str, Any]] = field(default_factory=list)
    engine_used: str = "local"


def heal_page_break_splits(text: str) -> str:
    """Reunites sentences artificially severed across PDF page breaks."""
    if not text:
        return text

    paragraphs = re.split(r"(\n\s*\n+)", text)
    healed: list[str] = []
    i = 0
    while i < len(paragraphs):
        seg = paragraphs[i]
        if i % 2 == 0 and i + 2 < len(paragraphs):
            p1 = seg.rstrip()
            p2 = paragraphs[i + 2].lstrip()
            if (
                p1
                and p2
                and not p1.endswith(("|", "#", "---", ">", ":", ".", "!", "?"))
                and not p2.startswith(
                    ("#", "|", "---", ">", "-", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")
                )
                and p2[0].islower()
            ):
                healed.append(p1 + " " + p2)
                i += 3
                continue
        healed.append(seg)
        i += 1
    return "".join(healed)


def clean_markdown_artifacts(text: str) -> str:
    """Removes PDF layout noise while keeping 100% of text, numbers, headers, and tables."""
    text = text.replace("\xad", "-").replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"<!--\s*(?:Start|End) of picture text\s*-->", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\*\*(?:CONTENTS|CONTENTS\s+BACK)\*\*\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    page_num_pattern = re.compile(
        r"^\s*(?:page\s+\d+(?:\s*(?:of|\/)\s*\d+)?|\d+\s*(?:of|\/)\s*\d+|[-–—]\s*\d+\s*[-–—]|\d+)\s*$",
        re.IGNORECASE,
    )

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if "|" not in line and page_num_pattern.match(line.strip()):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = heal_page_break_splits(cleaned)
    return cleaned.strip()


def extract_headings(markdown_text: str) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for line in markdown_text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            title = re.sub(r"<[^>]+>", "", m.group(2).strip())
            title = re.sub(r"[\*_~]", "", title).replace("\xad", "-").strip()
            headings.append({"level": len(m.group(1)), "title": title})
    return headings


def count_markdown_tables(markdown_text: str) -> int:
    return len(re.findall(r"^\|(?:\s*:?-+:?\s*\|)+\s*$", markdown_text, re.MULTILINE))


def convert_local_pymupdf(pdf_path: Path) -> PDFConversionResult:
    """High-fidelity structural conversion using PyMuPDF4LLM (tables + headings)."""
    import pymupdf as fitz
    import pymupdf4llm

    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    page_data = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    repeated_headers: set[str] = set()
    if len(page_data) >= 3:
        first_lines = [
            ln.strip()
            for item in page_data
            for ln in [next((x for x in item.get("text", "").splitlines() if x.strip()), "")]
            if ln
        ]
        repeated_headers = {line for line, cnt in Counter(first_lines).items() if cnt >= 3 and len(line) > 5}

    pages: list[dict[str, Any]] = []
    parts: list[str] = []
    for idx, item in enumerate(page_data):
        p_num = item.get("metadata", {}).get("page", idx) + 1
        raw = item.get("text", "")
        if idx > 0 and repeated_headers:
            raw = "\n".join(ln for ln in raw.splitlines() if ln.strip() not in repeated_headers)
        cleaned = clean_markdown_artifacts(raw)
        pages.append({"page_number": p_num, "text": cleaned})
        parts.append(cleaned)

    full_md = clean_markdown_artifacts("\n\n".join(parts))
    return PDFConversionResult(
        source_path=str(pdf_path),
        markdown_text=full_md,
        page_count=page_count,
        table_count=count_markdown_tables(full_md),
        word_count=len(full_md.split()),
        headings=extract_headings(full_md),
        pages=pages,
        engine_used="local_pymupdf",
    )


def convert_local_pypdf(pdf_path: Path) -> PDFConversionResult:
    """Pure-Python lightweight PDF text extractor using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    raw_pages = [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]

    repeated_headers: set[str] = set()
    if page_count >= 3:
        first_lines = [
            next((ln.strip() for ln in txt.splitlines() if ln.strip()), "")
            for _, txt in raw_pages
        ]
        repeated_headers = {
            line for line, cnt in Counter(ln for ln in first_lines if ln).items()
            if cnt >= 3 and len(line) > 5
        }

    pages: list[dict[str, Any]] = []
    parts: list[str] = []
    for idx, (p_num, raw) in enumerate(raw_pages):
        if idx > 0 and repeated_headers:
            raw = "\n".join(ln for ln in raw.splitlines() if ln.strip() not in repeated_headers)
        cleaned = clean_markdown_artifacts(raw)
        pages.append({"page_number": p_num, "text": cleaned})
        parts.append(cleaned)

    full_md = clean_markdown_artifacts("\n\n".join(parts))
    return PDFConversionResult(
        source_path=str(pdf_path),
        markdown_text=full_md,
        page_count=page_count,
        table_count=count_markdown_tables(full_md),
        word_count=len(full_md.split()),
        headings=extract_headings(full_md),
        pages=pages,
        engine_used="local_pypdf",
    )


def convert_gemini_vision(pdf_path: Path) -> PDFConversionResult:
    """Multimodal Gemini Vision OCR for scanned or table-heavy PDFs."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for Gemini Vision PDF conversion.")

    page_count = 1
    try:
        from pypdf import PdfReader
        page_count = len(PdfReader(str(pdf_path)).pages)
    except Exception:
        pass

    client = genai.Client(api_key=api_key)
    pdf_bytes = pdf_path.read_bytes()

    prompt = (
        "Convert the attached Singapore Town Council PDF document into high-fidelity "
        "GitHub-Flavored Markdown (GFM).\n"
        "RULES:\n"
        "1. Preserve all text, rules, hours, fees, phone numbers, and units verbatim.\n"
        "2. Convert all visual tables into clean GFM Markdown tables (| Col 1 | Col 2 |).\n"
        "3. Use # for the main document title, ## for sections, ### for subsections.\n"
        "4. Omit only running page numbers and repeated page headers/footers.\n"
    )

    preferred = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    models_to_try = list(dict.fromkeys([preferred, "gemini-2.5-flash", "gemini-2.0-flash"]))
    res_text = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    prompt,
                ],
            )
            if response.text and len(response.text.strip()) > 30:
                res_text = response.text.strip()
                break
        except Exception as exc:
            logger.warning("Gemini model %s failed during PDF conversion: %s", model_name, exc)

    if not res_text:
        raise RuntimeError("Gemini Vision could not convert the PDF.")

    if res_text.startswith("```markdown"):
        res_text = res_text[11:]
    elif res_text.startswith("```"):
        res_text = res_text[3:]
    if res_text.endswith("```"):
        res_text = res_text[:-3]

    cleaned = clean_markdown_artifacts(res_text.strip())
    return PDFConversionResult(
        source_path=str(pdf_path),
        markdown_text=cleaned,
        page_count=page_count,
        table_count=count_markdown_tables(cleaned),
        word_count=len(cleaned.split()),
        headings=extract_headings(cleaned),
        pages=[{"page_number": 1, "text": cleaned}],
        engine_used="gemini_vision",
    )


def convert_pdf_to_markdown(
    pdf_path: str | Path,
    engine: str = "auto",
    output_path: str | Path | None = None,
    title_override: str | None = None,
) -> PDFConversionResult:
    """Converts a PDF into structured Markdown ready for Town Council Semantic RAG."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if engine == "gemini":
        result = convert_gemini_vision(pdf_file)
    elif engine == "pymupdf":
        result = convert_local_pymupdf(pdf_file)
    elif engine == "pypdf":
        result = convert_local_pypdf(pdf_file)
    else:  # "auto"
        result = None
        for fn in (convert_local_pymupdf, convert_local_pypdf):
            try:
                result = fn(pdf_file)
                break
            except Exception as exc:
                logger.info("%s unavailable/failed (%s); trying next engine.", fn.__name__, exc)

        if result is None:
            result = convert_gemini_vision(pdf_file)
        else:
            avg_words = result.word_count / max(1, result.page_count)
            if (avg_words < 10 or result.word_count == 0) and os.getenv("GEMINI_API_KEY", "").strip():
                logger.info("Low text density (%.1f words/page); upgrading to Gemini Vision OCR.", avg_words)
                try:
                    gem = convert_gemini_vision(pdf_file)
                    if gem.word_count > result.word_count:
                        result = gem
                except Exception as exc:
                    logger.warning("Gemini Vision fallback failed; keeping local result: %s", exc)

    # Ensure Town Council Markdown header format (`# Title` + `Updated: YYYY-MM`)
    md = result.markdown_text.strip()
    if not md.startswith("# "):
        doc_title = title_override or pdf_file.stem.replace("_", " ").replace("-", " ").title()
        md = f"# {doc_title}\nUpdated: {date.today().strftime('%Y-%m')}\n\n{md}"
    else:
        lines = md.splitlines()
        if len(lines) < 2 or not lines[1].strip().lower().startswith("updated:"):
            lines.insert(1, f"Updated: {date.today().strftime('%Y-%m')}")
            md = "\n".join(lines)

    result.markdown_text = md

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")

    return result


def ingest_pdf_bytes_to_kb(
    pdf_bytes: bytes,
    filename: str,
    docs_dir: Path | str,
    engine: str = "auto",
) -> tuple[Path, PDFConversionResult]:
    """Saves uploaded PDF bytes temporarily, converts to Markdown in docs_dir, and returns output path."""
    import tempfile

    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem).strip("_").lower() or "uploaded_handbook"
    out_md = docs_path / f"{safe_stem}.md"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        res = convert_pdf_to_markdown(
            tmp_path,
            engine=engine,
            output_path=out_md,
            title_override=Path(filename).stem.replace("_", " ").replace("-", " ").title(),
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return out_md, res


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Town Council PDF handbook to Markdown for RAG.")
    parser.add_argument("pdf_path", help="Path to input PDF file")
    parser.add_argument("-o", "--output", help="Output .md path (defaults to data/docs/<name>.md)")
    parser.add_argument("--engine", choices=["auto", "pymupdf", "pypdf", "gemini"], default="auto")
    args = parser.parse_args()

    out = Path(args.output) if args.output else Path(__file__).resolve().parent / "data" / "docs" / f"{Path(args.pdf_path).stem}.md"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = convert_pdf_to_markdown(args.pdf_path, engine=args.engine, output_path=out)
    print(
        f"Converted {res.page_count} page(s), {res.table_count} table(s), {res.word_count} words "
        f"-> {out} (engine: {res.engine_used})"
    )


if __name__ == "__main__":
    main()
