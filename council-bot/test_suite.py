"""Automated test suite for EstateKaki Town Council Bot (Semantic RAG, Routing, DB & Trello)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

import config
import council_lookup
import db
import ingest
import rag
from semantic_rag import SemanticChunker, TownCouncilSemanticRAG
from trello_client import TrelloWorkOrders


# ============================================================================
# Semantic Hybrid RAG Tests
# ============================================================================

def test_semantic_chunker_preserves_markdown_tables_and_breadcrumbs():
    sample_md = """# Sample Town Council Fee Schedule
Updated: 2026-04

## Void Deck Fees
Booking fees apply to all HDB estates.

| Facility | Daily Rental | Security Deposit |
| --- | --- | --- |
| Void Deck | $50 | $200 |
| Multi-Purpose Hall | $150 | $500 |

## Operating Hours
Counters open Monday to Friday 8:00 AM to 5:00 PM.
"""
    chunker = SemanticChunker(max_chunk_chars=200)
    chunks = chunker.chunk_markdown_text(sample_md, source_file="fees.md")

    table_chunks = [c for c in chunks if c.has_table]
    assert len(table_chunks) == 1
    assert "| Void Deck | $50 | $200 |" in table_chunks[0].content
    assert "| Multi-Purpose Hall | $150 | $500 |" in table_chunks[0].content
    assert "Sample Town Council Fee Schedule" in table_chunks[0].header_breadcrumb
    assert "Void Deck Fees" in table_chunks[0].header_breadcrumb
    assert "[Document: Sample Town Council Fee Schedule | Section: Void Deck Fees]" in table_chunks[0].embeddable_text


def test_rag_answers_town_council_queries_and_refuses_out_of_scope(tmp_path):
    docs_dir = Path(__file__).resolve().parent / "data" / "docs"
    engine = TownCouncilSemanticRAG(
        docs_dir=docs_dir,
        index_path=tmp_path / "idx.json",
    )
    chunk_count = engine.initialize()
    assert chunk_count >= 20

    # In-scope Singapore Town Council query (colloquial phrasing)
    res_bulky = engine.query("Can I throw away my old sofa for free?")
    assert res_bulky.grounded is True
    assert res_bulky.answer is not None
    assert "sofa" in res_bulky.answer.lower() or "three" in res_bulky.answer.lower()

    res_corridor = engine.query("What is the 1.2m corridor fire safety rule?")
    assert res_corridor.grounded is True
    assert res_corridor.answer is not None
    assert "1.2" in res_corridor.answer

    # Out-of-scope query must be refused so bot escalates to staff
    res_oos = engine.query("What is the best pizza restaurant in Tokyo?")
    assert res_oos.answer is None
    assert res_oos.grounded is False


def test_rag_hot_reindexes_on_new_document(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "hours.md").write_text(
        "# Office Hours\nUpdated: 2026-03\n\nOpen Mon-Fri 8am to 5pm.\n",
        encoding="utf-8",
    )

    engine = TownCouncilSemanticRAG(docs_dir=docs_dir, index_path=tmp_path / "idx.json")
    engine.initialize()
    assert engine.query("Where is the solar panel charging kiosk?").answer is None

    # Simulate staff approving a new answer into data/docs without restarting the bot
    (docs_dir / "staff_answer_99.md").write_text(
        "# Solar Panel Charging Kiosk\nUpdated: 2026-04\n\n"
        "The solar panel charging kiosk is located at Blk 123 Void Deck next to the lift lobby.\n",
        encoding="utf-8",
    )
    res = engine.query("Where is the solar panel charging kiosk?")
    assert res.answer is not None
    assert "Blk 123" in res.answer


# ============================================================================
# Intent Classification, Emergency Guard & Town Council Routing Tests
# ============================================================================

def test_rule_based_classifier_distinguishes_questions_from_fault_reports():
    # Policy questions must NOT trigger the fault-report flow
    assert rag._rule_based_classify("What are the allowed renovation hours?")["intent"] == "question"
    assert rag._rule_based_classify("How much are S&CC charges?")["intent"] == "question"
    assert rag._rule_based_classify("What is the fine for feeding pigeons?")["intent"] == "question"

    # Physical fault reports MUST trigger report flow
    assert rag._rule_based_classify("The corridor light is flickering badly")["intent"] == "report"
    assert rag._rule_based_classify("Broken streetlight outside Blk 402")["category"] == "lighting"
    assert rag._rule_based_classify("Water pipe leaking at void deck")["category"] == "water"

    # Pre-LLM emergency guard
    assert config.EMERGENCY_RE.search("There is a fire at level 5 staircase!")
    assert rag._rule_based_classify("Someone collapsed and is unconscious")["intent"] == "emergency"


def test_statutory_agency_deflection_and_postal_mapping():
    assert "LTA" in (council_lookup.check_statutory_agency("Traffic light Spoilt at bus stop") or "")
    assert "NParks" in (council_lookup.check_statutory_agency("There is a wild boar near the park") or "")
    assert "SFA" in (council_lookup.check_statutory_agency("Dirty hawker centre food stall") or "")
    assert "HDB" in (council_lookup.check_statutory_agency("Ceiling leak inside my flat bedroom floor") or "")
    assert council_lookup.check_statutory_agency("Broken corridor light at Blk 102") is None

    assert council_lookup.POSTAL_SECTOR_MAP["56"] == "Ang Mo Kio Town Council"
    assert council_lookup.POSTAL_SECTOR_MAP["52"] == "Tampines Town Council"


def test_db_ticket_lifecycle_and_duplicate_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test_council.db"))
    db.init()

    tid = db.create_ticket(
        user_id=1001,
        category="lighting",
        description="Corridor light out",
        location="Blk 123 Ang Mo Kio Ave 3, Singapore 560123",
        photo_id=None,
        town_council="Ang Mo Kio Town Council",
        lat=1.3691,
        lng=103.8454,
        severity="Medium",
    )
    db.set_trello_card(tid, "card_abc", "https://trello.com/c/abc")

    dup = db.find_duplicate_ticket("lighting", "560123")
    assert dup is not None
    assert dup["id"] == tid
    assert dup["trello_card_id"] == "card_abc"

    db.subscribe_to_ticket(tid, 1002)
    t = db.get_ticket(tid)
    assert json.loads(t["subscribers"]) == [1001, 1002]


# ============================================================================
# Trello Work-Order Dispatch Tests (Mocked HTTP Transport)
# ============================================================================

def test_trello_from_env_and_work_order_lifecycle(monkeypatch):
    requests_seen: list[tuple[str, str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        requests_seen.append((request.method, request.url.path, body))

        if request.method == "GET" and request.url.path.endswith("/boards/board123/lists"):
            return httpx.Response(200, json=[
                {"id": "list_todo", "name": "To Do"},
                {"id": "list_urgent", "name": "Urgent"},
                {"id": "list_prog", "name": "In Progress"},
                {"id": "list_done", "name": "Done"},
            ])
        if request.method == "POST" and request.url.path.endswith("/cards"):
            return httpx.Response(200, json={"id": "card_999", "shortUrl": "https://trello.com/c/999"})
        if request.method == "PUT" and "/cards/card_999" in request.url.path:
            return httpx.Response(200, json={"id": "card_999"})
        if request.method == "POST" and "/actions/comments" in request.url.path:
            return httpx.Response(200, json={"id": "cmt_1"})
        return httpx.Response(404)

    monkeypatch.setenv("TRELLO_BOARD_ID", "board123")
    monkeypatch.setenv("TRELLO_API_KEY", "key_test")
    monkeypatch.setenv("TRELLO_TOKEN", "tok_test")

    sync_transport = httpx.MockTransport(handler)
    async_transport = httpx.MockTransport(handler)

    client = TrelloWorkOrders.from_env(transport=sync_transport, async_transport=async_transport)
    assert client is not None
    client.now = lambda: datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Critical ticket routes to Urgent list with 1-hour SLA due date
    card = asyncio.run(
        client.submit_work_order(
            ticket_id=42,
            category="water",
            description="Burst water pipe flooding lift lobby",
            location="Blk 501 Tampines Central 1, Singapore 520501",
            town_council="Tampines Town Council",
            severity="Critical",
            ai_diagnosis="Burst pipe (Assigned: Plumbing Team)",
            reporter="@resident1",
            lat=1.3533,
            lng=103.9452,
        )
    )
    assert card.card_id == "card_999"
    assert card.list_name == "Urgent"
    assert card.due_iso == "2026-04-01T13:00:00.000Z"

    post_req = [r for r in requests_seen if r[0] == "POST" and r[1].endswith("/cards")][0]
    assert post_req[2]["idList"] == "list_urgent"
    assert "Tampines Town Council" in post_req[2]["name"]
    assert "Burst water pipe" in post_req[2]["desc"]

    # 2. Routine ticket routes to To Do list with no urgent tag
    routine = asyncio.run(
        client.submit_work_order(
            ticket_id=43,
            category="lighting",
            description="Flickering corridor bulb",
            location="Blk 123 Ang Mo Kio, Singapore 560123",
            town_council="Ang Mo Kio Town Council",
            severity="Low",
        )
    )
    assert routine.list_name == "To Do"
    assert routine.due_iso is None

    # 3. Status sync moves card to Done and posts comment
    asyncio.run(client.update_status("card_999", "resolved", "Pipe replaced by EMSU crew"))
    put_reqs = [r for r in requests_seen if r[0] == "PUT" and "/cards/card_999" in r[1]]
    assert put_reqs and put_reqs[-1][2]["idList"] == "list_done"

    # 4. Sync dashboard status update works cleanly
    assert client.update_status_sync("card_999", "in progress", "Crew en route") is True


# ============================================================================
# PDF-to-Markdown Ingestion Pipeline Tests
# ============================================================================

def test_pdf_converter_heals_page_breaks_and_cleans_artifacts():
    import pdf_converter

    raw = (
        "<!-- Start of picture text -->\n"
        "Page 1 of 12\n\n"
        "# Town Council Handbook\n\n"
        "Residents must keep corridors clean and\n\n"
        "unobstructed at all times.\n\n"
        "| Service | Fee |\n"
        "| --- | --- |\n"
        "| Void Deck | $50 |\n\n"
        "Page 2 of 12\n"
    )
    cleaned = pdf_converter.clean_markdown_artifacts(raw)
    assert "Page 1 of 12" not in cleaned
    assert "Start of picture text" not in cleaned
    assert "clean and unobstructed at all times." in cleaned
    assert pdf_converter.count_markdown_tables(cleaned) == 1
    assert pdf_converter.extract_headings(cleaned)[0]["title"] == "Town Council Handbook"


# ============================================================================
# Handbook Policy Copilot (HITL) Tests
# ============================================================================

def test_policy_copilot_amend_audit_and_rollback(tmp_path):
    import policy_editor

    doc_path = tmp_path / "void_deck_test.md"
    original = (
        "# Void Deck Bookings\n"
        "Updated: 2026-03\n\n"
        "Fees, Deposits, and Utilities:\n"
        "- Daily space rental fees range from $50 to $150 per day.\n"
        "- Refundable security deposit is $200 per event.\n\n"
        "Regulations:\n"
        "- Amplified music must cease by 10:30 PM.\n"
    )
    doc_path.write_text(original, encoding="utf-8")
    backups_dir = tmp_path / "backups"

    sections = policy_editor.parse_handbook_sections(original)
    assert len(sections) >= 3

    target, score, rationale = policy_editor.locate_target_section(
        sections, "Increase security deposit from $200 to $300", use_llm=False
    )
    assert "Fees" in target.title
    assert score > 0

    proposal = policy_editor.generate_policy_amendment(
        target, "Increase security deposit from $200 to $300"
    )
    assert "$300" in proposal.proposed_raw_block
    assert proposal.additions_count >= 1
    assert "background-color: #e6ffed" in policy_editor.render_diff_html(proposal.diff_lines)

    ok, record, msg = policy_editor.apply_amendment_to_handbook(
        doc_path,
        target,
        proposal.proposed_raw_block,
        proposal.intent,
        proposal.explanation,
        author="Officer Tan",
        backups_dir=backups_dir,
    )
    assert ok is True
    assert "$300" in doc_path.read_text(encoding="utf-8")
    history = policy_editor.load_audit_history(backups_dir=backups_dir)
    assert len(history) == 1
    assert history[0]["status"] == "APPLIED"

    # 1-Click Rollback restores original $200 text
    rb_ok, rb_msg = policy_editor.rollback_amendment(
        doc_path, record.backup_file, backups_dir=backups_dir
    )
    assert rb_ok is True
    assert "$200" in doc_path.read_text(encoding="utf-8")
    assert "$300" not in doc_path.read_text(encoding="utf-8")


# ============================================================================
# FAQ Benchmark Dataset Test (from Dylan's condo_faq.csv + faq_eval.csv)
# ============================================================================

def test_semantic_rag_against_benchmark_faq_dataset(tmp_path):
    import csv

    bench_dir = Path(__file__).resolve().parent / "data" / "benchmarks"
    faq_csv = bench_dir / "condo_faq.csv"
    eval_csv = bench_dir / "faq_eval.csv"
    assert faq_csv.exists() and eval_csv.exists()

    # Convert FAQ CSV rows into markdown docs inside tmp_path to verify Semantic RAG FAQ accuracy
    faq_docs_dir = tmp_path / "faq_docs"
    faq_docs_dir.mkdir()
    with faq_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            md = f"# {row['question']}\nUpdated: 2026-04\n\nAnswer:\n{row['answer']}\n\nKeywords:\n{row['keywords']}\n"
            (faq_docs_dir / f"{row['id']}.md").write_text(md, encoding="utf-8")

    engine = TownCouncilSemanticRAG(docs_dir=faq_docs_dir, index_path=tmp_path / "faq_idx.json")
    assert engine.initialize() >= 30

    hits = 0
    total_faq = 0
    with eval_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expected_id = (row.get("faq_id") or "").strip()
            if not expected_id:
                continue
            total_faq += 1
            res = engine.query(row["text"], top_k=3)
            top_stems = [Path(c.source_file).stem for c in res.retrieved_chunks[:3]]
            if expected_id in top_stems:
                hits += 1

    assert total_faq >= 30
    assert hits / total_faq >= 0.85

