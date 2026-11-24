"""Town Council Handbook Policy Copilot (Human-in-the-Loop).

Empowers Singapore Town Council officers to amend policy documents and converted
PDF handbooks in `data/docs/` using natural language without editing raw Markdown:
1. Structure-Aware Section Parser: Supports both Markdown `# / ## / ###` headings
   and Town Council `Section Heading:` lines with exact character offsets.
2. Section Localization Agent: Pinpoints the exact policy clause to amend.
3. Surgical Constrained Rewriting: Preserves untouched rules and Markdown tables,
   reconciling spurious whitespace/formatting diffs automatically.
4. Visual Diff Preview: Generates GitHub-style color-coded HTML diffs for staff sign-off.
5. Atomic Backup, Immutable Audit Log & 1-Click Rollback: Saves timestamped snapshots
   in `data/backups/` and logs every amendment in `amendment_audit_log.json`.
6. Zero-Downtime Hot Re-Index: Immediately updates the Semantic Hybrid RAG engine.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("council-bot.policy_editor")

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"
BACKUPS_DIR = BASE_DIR / "data" / "backups"
AUDIT_LOG_FILE = BACKUPS_DIR / "amendment_audit_log.json"

COLON_HEADING_RE = re.compile(r"^([A-Z][A-Za-z0-9 &/()\-,'&]{2,70}):\s*$")


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PolicySection:
    index: int
    level: int
    heading_line: str
    title: str
    parent_heading: str
    breadcrumb: str
    start_char: int
    end_char: int
    line_no: int
    raw_block: str
    body: str

    @property
    def section_id(self) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", self.breadcrumb.lower()).strip("_")
        return f"sec_{self.index}_{clean[:32]}"


@dataclass
class PolicyAmendmentProposal:
    section: PolicySection
    intent: str
    explanation: str
    original_raw_block: str
    proposed_raw_block: str
    proposed_body: str
    summary_bullets: list[str]
    diff_lines: list[tuple[str, str]]
    unified_diff: str
    additions_count: int
    deletions_count: int
    model_used: str


@dataclass
class AmendmentAuditRecord:
    log_id: str
    timestamp: str
    author: str
    document_name: str
    intent: str
    target_section: str
    backup_file: str
    summary: str
    additions_count: int
    deletions_count: int
    diff_snippet: str
    status: str = "APPLIED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp,
            "author": self.author,
            "document_name": self.document_name,
            "intent": self.intent,
            "target_section": self.target_section,
            "backup_file": self.backup_file,
            "summary": self.summary,
            "additions_count": self.additions_count,
            "deletions_count": self.deletions_count,
            "diff_snippet": self.diff_snippet,
            "status": self.status,
        }


# ============================================================================
# Section Parsing & Exact Offsets
# ============================================================================

def list_policy_documents(docs_dir: Path | str | None = None) -> list[Path]:
    root = Path(docs_dir) if docs_dir else DOCS_DIR
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*.md") if p.is_file())


def parse_handbook_sections(markdown_text: str) -> list[PolicySection]:
    """Parses a Markdown policy doc into surgically replaceable sections with exact offsets."""
    lines = markdown_text.splitlines(keepends=True)
    md_heading_re = re.compile(r"^(#{1,4})\s+(.+)$")

    raw_headings: list[dict[str, Any]] = []
    char_offset = 0
    in_code_block = False

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            char_offset += len(line)
            continue

        if not in_code_block:
            line_no_nl = line.rstrip("\r\n")
            m_md = md_heading_re.match(line_no_nl)
            m_col = (
                COLON_HEADING_RE.match(stripped)
                if not stripped.startswith(("-", "*", "|", "#"))
                else None
            )
            if m_md:
                raw_headings.append({
                    "line_no": line_idx,
                    "start_char": char_offset,
                    "level": len(m_md.group(1)),
                    "heading_line": line_no_nl,
                    "title": m_md.group(2).strip(),
                })
            elif m_col:
                raw_headings.append({
                    "line_no": line_idx,
                    "start_char": char_offset,
                    "level": 2,
                    "heading_line": line_no_nl,
                    "title": m_col.group(1).strip(),
                })
        char_offset += len(line)

    if not raw_headings:
        return [
            PolicySection(
                index=0,
                level=1,
                heading_line="",
                title="Full Document",
                parent_heading="",
                breadcrumb="Full Document",
                start_char=0,
                end_char=len(markdown_text),
                line_no=0,
                raw_block=markdown_text,
                body=markdown_text,
            )
        ]

    for i in range(len(raw_headings)):
        raw_headings[i]["end_char"] = (
            raw_headings[i + 1]["start_char"] if i < len(raw_headings) - 1 else len(markdown_text)
        )

    sections: list[PolicySection] = []
    if raw_headings[0]["start_char"] > 0:
        preamble = markdown_text[: raw_headings[0]["start_char"]]
        sections.append(
            PolicySection(
                index=0,
                level=1,
                heading_line="",
                title="Frontmatter & Metadata",
                parent_heading="",
                breadcrumb="Frontmatter & Metadata",
                start_char=0,
                end_char=raw_headings[0]["start_char"],
                line_no=0,
                raw_block=preamble,
                body=preamble,
            )
        )

    current_h1 = ""
    current_h2 = ""
    for h in raw_headings:
        idx = len(sections)
        lvl = h["level"]
        title = h["title"]
        raw_block = markdown_text[h["start_char"]: h["end_char"]]
        parts = raw_block.split("\n", 1)
        body = parts[1] if len(parts) > 1 else ""

        if lvl == 1:
            current_h1 = title
            current_h2 = ""
            parent_h = ""
            breadcrumb = title
        elif lvl == 2:
            current_h2 = title
            parent_h = current_h1
            breadcrumb = f"{current_h1} > {title}" if current_h1 else title
        else:
            parent_h = current_h2 or current_h1
            breadcrumb = f"{parent_h} > {title}" if parent_h else title

        sections.append(
            PolicySection(
                index=idx,
                level=lvl,
                heading_line=h["heading_line"],
                title=title,
                parent_heading=parent_h,
                breadcrumb=breadcrumb,
                start_char=h["start_char"],
                end_char=h["end_char"],
                line_no=h["line_no"],
                raw_block=raw_block,
                body=body,
            )
        )

    return sections


# ============================================================================
# Section Localization Agent
# ============================================================================

def locate_target_section(
    sections: list[PolicySection],
    user_intent: str,
    use_llm: bool = True,
) -> tuple[PolicySection, float, str]:
    """Locates the exact policy section to amend via keyword scoring + optional Gemini verification."""
    if not sections:
        raise ValueError("No sections found in document.")
    if not user_intent or not user_intent.strip():
        return sections[0], 0.0, "No amendment instruction provided."

    candidates = [s for s in sections if s.level >= 2] or sections
    stopwords = {
        "the", "a", "an", "and", "or", "to", "for", "in", "of", "on", "from",
        "with", "is", "be", "change", "update", "set", "can", "please", "make",
    }
    intent_clean = user_intent.lower()
    query_terms = [t for t in re.findall(r"\w+", intent_clean) if t not in stopwords and len(t) > 1]

    scored: list[tuple[float, PolicySection, list[str]]] = []
    for s in candidates:
        score = 0.0
        matches: list[str] = []
        title_l = s.breadcrumb.lower()
        body_l = s.raw_block.lower()
        for term in query_terms:
            if term in title_l:
                score += 5.0
                matches.append(term)
            elif term in body_l:
                score += 2.0
                matches.append(term)
        if s.level >= 3 and score > 0:
            score += 0.5
        scored.append((score, s, list(dict.fromkeys(matches))))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_sec, matched_words = scored[0]

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if use_llm and api_key and len(scored) > 1:
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            top_cands = scored[:3]
            cand_text = "\n".join(
                f"[{i + 1}] {c[1].breadcrumb}\nSnippet: {c[1].body[:220]}..."
                for i, c in enumerate(top_cands)
            )
            prompt = (
                "You are a Singapore Town Council Policy Navigation Agent.\n"
                f'OFFICER INTENT: "{user_intent}"\n\n'
                f"CANDIDATE SECTIONS:\n{cand_text}\n\n"
                "Select the single best candidate number (1, 2, or 3) and provide a 1-sentence rationale.\n"
                'Return JSON: {"selected_choice": 1, "rationale": "..."}'
            )
            res = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.1},
            )
            data = json.loads(res.text)
            sel = int(data.get("selected_choice", 1)) - 1
            if 0 <= sel < len(top_cands):
                chosen = top_cands[sel][1]
                rationale = data.get("rationale") or f"Selected '{chosen.breadcrumb}'."
                return chosen, top_cands[sel][0], rationale
        except Exception as exc:
            logger.debug("LLM section localization skipped (%s); using token overlap.", exc)

    if matched_words:
        rationale = f"Matched key terms [{', '.join(matched_words)}] in '{best_sec.breadcrumb}'."
    else:
        rationale = f"Selected '{best_sec.breadcrumb}' as the closest policy section."
    return best_sec, best_score, rationale


# ============================================================================
# Surgical Amendment Proposal & Diff Engine
# ============================================================================

def reconcile_surgical_amendment(orig_block: str, prop_block: str, user_intent: str = "") -> str:
    """Eliminates accidental whitespace or formatting micro-diffs on untouched lines."""
    orig_trailing_ws = orig_block[len(orig_block.rstrip("\r\n")):]
    orig_lines = [l.rstrip("\r\n") for l in orig_block.splitlines()]
    prop_lines = [l.rstrip("\r\n") for l in prop_block.splitlines()]

    is_deletion = any(
        w in user_intent.lower()
        for w in ["delete", "remove", "omit", "drop", "strike", "cancel", "rescind", "scrap", "eliminate"]
    )

    def _same(a: str, b: str) -> bool:
        return re.sub(r"[^a-zA-Z0-9]+", "", a).lower() == re.sub(r"[^a-zA-Z0-9]+", "", b).lower()

    sm = difflib.SequenceMatcher(None, orig_lines, prop_lines)
    reconciled: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            reconciled.extend(orig_lines[i1:i2])
        elif tag == "replace":
            o_sub, p_sub = orig_lines[i1:i2], prop_lines[j1:j2]
            if len(o_sub) == len(p_sub):
                for ol, pl in zip(o_sub, p_sub):
                    reconciled.append(ol if _same(ol, pl) else pl)
            else:
                reconciled.extend(p_sub)
        elif tag == "delete":
            if not is_deletion or all(not l.strip() for l in orig_lines[i1:i2]):
                reconciled.extend(orig_lines[i1:i2])
        elif tag == "insert":
            reconciled.extend(prop_lines[j1:j2])

    out = "\n".join(reconciled)
    if orig_trailing_ws:
        out = out.rstrip("\r\n") + orig_trailing_ws
    elif not out.endswith("\n"):
        out += "\n"
    return out


def _compute_annotated_diff(
    orig_lines: list[str],
    prop_lines: list[str],
) -> tuple[list[tuple[str, str]], int, int]:
    matcher = difflib.SequenceMatcher(None, orig_lines, prop_lines)
    diff_lines: list[tuple[str, str]] = []
    adds = 0
    dels = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for ln in orig_lines[i1:i2]:
                diff_lines.append(("unchanged", ln.rstrip("\r\n")))
        elif tag == "replace":
            for ln in orig_lines[i1:i2]:
                diff_lines.append(("delete", ln.rstrip("\r\n")))
                dels += 1
            for ln in prop_lines[j1:j2]:
                diff_lines.append(("add", ln.rstrip("\r\n")))
                adds += 1
        elif tag == "delete":
            for ln in orig_lines[i1:i2]:
                diff_lines.append(("delete", ln.rstrip("\r\n")))
                dels += 1
        elif tag == "insert":
            for ln in prop_lines[j1:j2]:
                diff_lines.append(("add", ln.rstrip("\r\n")))
                adds += 1
    return diff_lines, adds, dels


def _deterministic_amendment_fallback(
    section: PolicySection,
    user_intent: str,
) -> tuple[str, list[str], str]:
    """Deterministic surgical modifier when Gemini API is unavailable."""
    updated = section.raw_block
    summary: list[str] = []

    # 1. Dollar fee / fine / deposit adjustments (e.g., "from $200 to $300" or "set deposit to $300")
    dollars = re.findall(r"\$(\d+(?:,\d+)*(?:\.\d{2})?)", user_intent)
    if len(dollars) >= 2:
        old_val, new_val = dollars[0], dollars[1]
        pat = re.compile(rf"(?<!\d)S?\$\s*{re.escape(old_val)}(?:\.00)?(?!\d)", re.IGNORECASE)
        if pat.search(updated):
            updated = pat.sub(f"${new_val}", updated)
            summary.append(f"Updated monetary figure from ${old_val} to ${new_val}")
    elif len(dollars) == 1:
        new_val = dollars[0]
        lines = updated.splitlines()
        stopwords = {"the", "a", "an", "and", "or", "to", "for", "in", "of", "on", "from", "set", "change", "update", "increase", "decrease"}
        hints = [t for t in re.findall(r"\b\w+\b", user_intent.lower()) if t not in stopwords and len(t) > 2]
        replaced = False
        for i, ln in enumerate(lines):
            if any(h in ln.lower() for h in hints):
                new_ln = re.sub(r"(?<!\d)S?\$(\d+(?:,\d+)*(?:\.\d{2})?)(?!\d)", f"${new_val}", ln, count=1)
                if new_ln != ln:
                    lines[i] = new_ln
                    replaced = True
                    break
        updated = "\n".join(lines) if replaced else re.sub(
            r"(?<!\d)S?\$(\d+(?:,\d+)*(?:\.\d{2})?)(?!\d)", f"${new_val}", updated, count=1
        )
        if updated != section.raw_block:
            summary.append(f"Updated fee/deposit amount to ${new_val}")

    # 2. Time adjustments (e.g., "from 10:30 PM to 11:00 PM")
    if updated == section.raw_block:
        times = re.findall(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", user_intent, re.IGNORECASE)
        if len(times) >= 2:
            pat_t = re.compile(re.escape(times[0]), re.IGNORECASE)
            if pat_t.search(updated):
                updated = pat_t.sub(times[1].upper(), updated)
                summary.append(f"Adjusted timing from {times[0]} to {times[1].upper()}")

    # 3. Fallback: append an explicit Town Council policy addendum bullet
    if updated == section.raw_block:
        addendum = f"- **Policy Addendum:** {user_intent.strip()}"
        updated = updated.rstrip() + f"\n{addendum}\n"
        summary.append(f"Added policy addendum: {user_intent.strip()}")

    return updated, summary, f"Drafted amendment to '{section.title}'."


def generate_policy_amendment(
    section: PolicySection,
    user_intent: str,
    model_name: str | None = None,
) -> PolicyAmendmentProposal:
    """Drafts a surgical policy amendment with visual diff for human-in-the-loop review."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    proposed_raw_block = ""
    summary_bullets: list[str] = []
    explanation = ""
    model_used = "deterministic_rule_engine"

    if api_key:
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            sys_prompt = (
                "You are a Singapore Town Council Policy Editor.\n"
                "Apply SURGICAL, MINIMAL amendments to Town Council policy documents and Markdown tables.\n"
                "RULES:\n"
                "1. Change ONLY the specific number, time, rule, or clause requested.\n"
                "2. Keep every untouched line, heading, bullet, and GFM table row 100% verbatim.\n"
                "3. Return ONLY a valid JSON object matching the schema."
            )
            prompt = (
                f'OFFICER INSTRUCTION:\n"{user_intent}"\n\n'
                f"CURRENT SECTION:\n{section.raw_block}\n\n"
                "Return JSON:\n"
                "{\n"
                '  "explanation": "1-sentence summary of the policy change",\n'
                '  "summary_bullets": ["Exact change 1"],\n'
                '  "replacements": [{"original_phrase": "verbatim substring", "updated_phrase": "replacement"}],\n'
                '  "updated_markdown": "Full updated section markdown"\n'
                "}"
            )
            res = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config={
                    "system_instruction": sys_prompt,
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            parsed = json.loads(res.text)
            summary_bullets = parsed.get("summary_bullets", [])
            explanation = parsed.get("explanation", f"Amended '{section.title}'.")
            model_used = f"gemini:{gemini_model}"

            reps = parsed.get("replacements", [])
            if isinstance(reps, list) and reps:
                cand = section.raw_block
                ok = True
                for r in reps:
                    op, up = r.get("original_phrase", ""), r.get("updated_phrase", "")
                    if op and op in cand:
                        cand = cand.replace(op, up, 1)
                    else:
                        ok = False
                        break
                if ok and cand != section.raw_block:
                    proposed_raw_block = cand

            if not proposed_raw_block:
                proposed_raw_block = (parsed.get("updated_markdown") or "").strip()
        except Exception as exc:
            logger.warning("Gemini policy amendment failed (%s); using deterministic fallback.", exc)

    if not proposed_raw_block:
        proposed_raw_block, summary_bullets, explanation = _deterministic_amendment_fallback(
            section, user_intent
        )

    proposed_raw_block = reconcile_surgical_amendment(
        section.raw_block, proposed_raw_block, user_intent=user_intent
    )
    if not proposed_raw_block.endswith("\n"):
        proposed_raw_block += "\n"

    parts = proposed_raw_block.split("\n", 1)
    proposed_body = parts[1] if len(parts) > 1 else ""

    orig_lines = section.raw_block.splitlines(keepends=True)
    prop_lines = proposed_raw_block.splitlines(keepends=True)
    unified = "".join(
        difflib.unified_diff(
            orig_lines,
            prop_lines,
            fromfile=f"Current: {section.title}",
            tofile=f"Proposed: {section.title}",
            n=3,
        )
    )
    diff_lines, adds, dels = _compute_annotated_diff(orig_lines, prop_lines)

    return PolicyAmendmentProposal(
        section=section,
        intent=user_intent,
        explanation=explanation,
        original_raw_block=section.raw_block,
        proposed_raw_block=proposed_raw_block,
        proposed_body=proposed_body,
        summary_bullets=summary_bullets,
        diff_lines=diff_lines,
        unified_diff=unified,
        additions_count=adds,
        deletions_count=dels,
        model_used=model_used,
    )


# ============================================================================
# Atomic Apply, Backup, Audit Log, Rollback & Live Hot Re-Index
# ============================================================================

def _audit_log_path(backups_dir: Path) -> Path:
    return backups_dir / "amendment_audit_log.json"


def _append_audit_log(record: AmendmentAuditRecord, backups_dir: Path) -> None:
    backups_dir.mkdir(parents=True, exist_ok=True)
    log_file = _audit_log_path(backups_dir)
    logs: list[dict[str, Any]] = []
    if log_file.exists():
        try:
            logs = json.loads(log_file.read_text(encoding="utf-8"))
        except Exception:
            logs = []
    logs.insert(0, record.to_dict())
    log_file.write_text(json.dumps(logs, indent=2), encoding="utf-8")


def load_audit_history(limit: int = 50, backups_dir: Path | str | None = None) -> list[dict[str, Any]]:
    bdir = Path(backups_dir) if backups_dir else BACKUPS_DIR
    log_file = _audit_log_path(bdir)
    if not log_file.exists():
        return []
    try:
        return json.loads(log_file.read_text(encoding="utf-8"))[:limit]
    except Exception:
        return []


def _reindex_semantic_rag() -> str:
    try:
        import rag

        chunks = rag.get_engine().reindex()
        return f"Semantic RAG hot-reindexed ({chunks} chunks)."
    except Exception as exc:
        logger.warning("RAG hot re-index warning: %s", exc)
        return f"RAG re-index note: {exc}"


def apply_amendment_to_handbook(
    handbook_path: Path | str,
    section: PolicySection,
    new_raw_block: str,
    user_intent: str,
    summary: str,
    author: str = "Council Officer",
    backups_dir: Path | str | None = None,
) -> tuple[bool, AmendmentAuditRecord, str]:
    """Applies a verified policy amendment with backup, audit trail, and live RAG hot re-index."""
    doc_path = Path(handbook_path).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"Policy document not found: {doc_path}")

    bdir = Path(backups_dir).resolve() if backups_dir else BACKUPS_DIR.resolve()
    bdir.mkdir(parents=True, exist_ok=True)

    original_text = doc_path.read_text(encoding="utf-8")
    if original_text[section.start_char: section.end_char] != section.raw_block:
        reparsed = parse_handbook_sections(original_text)
        matches = [s for s in reparsed if s.heading_line == section.heading_line and s.title == section.title]
        if not matches:
            raise ValueError(f"Document changed concurrently; could not relocate '{section.title}'.")
        section = matches[0]

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    backup_filename = f"{doc_path.stem}_{ts}_{short_id}.md"
    backup_path = bdir / backup_filename
    backup_path.write_bytes(doc_path.read_bytes())

    clean_block = new_raw_block if new_raw_block.endswith("\n") else new_raw_block + "\n"
    updated_text = original_text[: section.start_char] + clean_block + original_text[section.end_char:]
    doc_path.write_text(updated_text, encoding="utf-8", newline="\n")

    orig_lines = section.raw_block.splitlines()
    new_lines = clean_block.splitlines()
    _, adds, dels = _compute_annotated_diff(orig_lines, new_lines)
    diff_snippet = "\n".join(difflib.unified_diff(orig_lines, new_lines, lineterm="", n=2))

    record = AmendmentAuditRecord(
        log_id=f"amend_{short_id}",
        timestamp=now.isoformat(),
        author=author,
        document_name=doc_path.name,
        intent=user_intent,
        target_section=section.breadcrumb,
        backup_file=backup_filename,
        summary=summary,
        additions_count=adds,
        deletions_count=dels,
        diff_snippet=diff_snippet[:2000],
        status="APPLIED",
    )
    _append_audit_log(record, bdir)
    reindex_msg = _reindex_semantic_rag()

    return (
        True,
        record,
        f"Applied amendment to '{section.breadcrumb}' in {doc_path.name}. Backup: {backup_filename}. {reindex_msg}",
    )


def rollback_amendment(
    handbook_path: Path | str,
    backup_filename: str,
    author: str = "Council Officer (Rollback)",
    backups_dir: Path | str | None = None,
) -> tuple[bool, str]:
    """Restores a policy document from a previous snapshot and hot-reindexes Semantic RAG."""
    doc_path = Path(handbook_path).resolve()
    bdir = Path(backups_dir).resolve() if backups_dir else BACKUPS_DIR.resolve()
    target_backup = (bdir / Path(backup_filename).name).resolve()

    if not target_backup.is_relative_to(bdir):
        return False, "Access denied: backup must reside inside the backups directory."
    if not target_backup.exists():
        return False, f"Backup snapshot not found: {target_backup.name}"

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    safety_name = f"{doc_path.stem}_prerollback_{ts}_{uuid.uuid4().hex[:6]}.md"
    if doc_path.exists():
        (bdir / safety_name).write_bytes(doc_path.read_bytes())

    doc_path.write_bytes(target_backup.read_bytes())

    record = AmendmentAuditRecord(
        log_id=f"rollback_{uuid.uuid4().hex[:8]}",
        timestamp=now.isoformat(),
        author=author,
        document_name=doc_path.name,
        intent=f"Rollback {doc_path.name} to {target_backup.name}",
        target_section="Entire Document (Restoration)",
        backup_file=safety_name,
        summary=f"Restored {doc_path.name} from backup {target_backup.name}",
        additions_count=0,
        deletions_count=0,
        diff_snippet=f"Restored from {target_backup.name}",
        status="ROLLED_BACK",
    )
    _append_audit_log(record, bdir)
    reindex_msg = _reindex_semantic_rag()
    return True, f"Restored {doc_path.name} from '{target_backup.name}'. {reindex_msg}"


# ============================================================================
# Visual HTML Diff Renderer (for Streamlit)
# ============================================================================

def render_diff_html(diff_lines: list[tuple[str, str]]) -> str:
    """Renders GitHub-style color-coded diff HTML."""
    out = [
        '<div style="font-family: Consolas, monospace; font-size: 13px; line-height: 1.5; '
        'background-color: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; '
        'padding: 12px; overflow-x: auto; max-height: 380px;">'
    ]
    for tag, line in diff_lines:
        safe = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ", "&nbsp;")
        )
        if tag == "add":
            out.append(
                f'<div style="background-color: #e6ffed; color: #1a7f37; border-left: 3px solid #2da44e; '
                f'padding: 2px 6px; margin: 1px 0;">+ {safe}</div>'
            )
        elif tag == "delete":
            out.append(
                f'<div style="background-color: #ffebe9; color: #cf222e; border-left: 3px solid #cf222e; '
                f'text-decoration: line-through; padding: 2px 6px; margin: 1px 0;">- {safe}</div>'
            )
        else:
            out.append(f'<div style="color: #24292f; padding: 2px 6px; margin: 1px 0;">&nbsp;&nbsp;{safe}</div>')
    out.append("</div>")
    return "".join(out)
