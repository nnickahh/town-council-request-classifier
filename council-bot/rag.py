"""Town Council RAG pipeline + intent classifier.

Backed by the upgraded Two-Stage Semantic Hybrid RAG engine (`semantic_rag.py`)
with Contextual Breadcrumbs, Small-to-Big parent context, Singlish/HDB query
expansion, zero-downtime hot re-indexing, and strict staff-escalation refusal.
Supports Google Gemini, Anthropic Claude, and deterministic offline fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import config
from semantic_rag import TownCouncilSemanticRAG

_BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else _BASE_DIR / p


_ENGINE: TownCouncilSemanticRAG | None = None


def get_engine() -> TownCouncilSemanticRAG:
    """Returns the singleton Town Council Semantic RAG engine (auto-hot-reloads on doc edits)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TownCouncilSemanticRAG(
            docs_dir=_resolve_path(config.DOCS_DIR),
            index_path=_resolve_path(config.RAG_INDEX_PATH),
            council_name=config.COUNCIL_NAME,
            gemini_model=config.GEMINI_MODEL,
            anthropic_model=config.ANSWER_MODEL,
            min_relevance_score=config.MIN_RELEVANCE_SCORE,
        )
    return _ENGINE


def reset_engine() -> None:
    """Forces immediate re-indexing on the next query (useful after bulk ingestion or tests)."""
    global _ENGINE
    if _ENGINE is not None:
        _ENGINE.reindex()
    else:
        get_engine().reindex()


def retrieve(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieves top ranked semantic chunks for a resident question."""
    return get_engine().retrieve(question, top_k=top_k or config.TOP_K)


async def answer(question: str) -> dict[str, Any]:
    """Answers a resident inquiry using Semantic Hybrid RAG with strict refusal cutoff."""
    result = await asyncio.to_thread(get_engine().query, question, config.TOP_K, True)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "best_distance": result.best_distance,
        "best_score": result.best_score,
        "retrieved_chunks": [
            {
                "title": c.document_title,
                "section": c.section,
                "breadcrumb": c.header_breadcrumb,
                "source_file": c.source_file,
            }
            for c in result.retrieved_chunks
        ],
        "model_used": result.model_used,
        "expanded_query": result.expanded_query,
        "latency_seconds": result.latency_seconds,
    }


# ============================================================================
# Intent & Category Classifier (Gemini -> Claude -> Improved Rule Fallback)
# ============================================================================

_gemini_client = None
_anthropic_client = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

if GEMINI_API_KEY:
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        pass

if ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("sk-ant-..."):
    try:
        from anthropic import AsyncAnthropic
        _anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:
        pass

ROUTER_SYSTEM = """Classify a message sent by a Singapore resident to a town council chatbot.
Return ONLY a JSON object: {"intent": "...", "category": "..."}
intent:
- "question": asks for information (hours, rules, fees, quotas, fines, how to apply, booking procedures, dates...)
- "report": reports a physical problem or requests maintenance action (broken light, pothole, missed bin, active loud drilling complaint, corridor clutter, lift fault, dirty void deck, stagnant water...)
- "emergency": immediate danger to life or safety
- "other": greeting, thanks, chit-chat, unrelated
category (only matters for reports): roads, lighting, waste, water, noise, parks, other"""

INTENTS = {"question", "report", "emergency", "other"}
CATEGORIES = {"roads", "lighting", "waste", "water", "noise", "parks", "other"}

QUESTION_LEAD_RE = re.compile(
    r"^(?:what|when|where|how|who|can i|am i|is it|are there|do i|does|how much|how many|how long)\b",
    re.IGNORECASE,
)
POLICY_QUESTION_RE = re.compile(
    r"\b(hours|open|opening|close|closed|fee|fees|cost|price|quota|free|book|booking|permit|rule|rules|guideline|fine|penalty|allowed|permitted|rebate|giro|paynow|s&cc|how to)\b",
    re.IGNORECASE,
)
ACTIVE_FAULT_RE = re.compile(
    r"\b(broken|spoil|spoilt|faulty|not working|flicker|flickering|burnt out|pothole|crack|choked?|overflowing|stinky|dirty|litter|dumped|leaking|burst|flooded|wet floor|slippery|puddle|spilled|stuck|trapped|stagnant water|mosquito breeding|rats?|cockroach|fallen tree|collapsed)\b",
    re.IGNORECASE,
)


def _rule_based_classify(message: str) -> dict[str, str]:
    """Deterministic fallback classifier when no LLM key is configured."""
    m = message.strip().lower()
    if config.EMERGENCY_RE.search(m):
        return {"intent": "emergency", "category": "other"}

    if re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening|thanks|thank you)\b", m) and len(m) < 28:
        return {"intent": "other", "category": "other"}

    # Distinguish policy/FAQ questions from active fault reports
    if (QUESTION_LEAD_RE.search(m) or m.endswith("?")) and not ACTIVE_FAULT_RE.search(m):
        return {"intent": "question", "category": "other"}
    if POLICY_QUESTION_RE.search(m) and not ACTIVE_FAULT_RE.search(m):
        return {"intent": "question", "category": "other"}

    # Safety hazards / wet floor / cleaning
    if re.search(r"\b(wet floor|slippery|puddle|spill(ed|age)?|oil on (floor|ground)|slime)\b", m):
        return {"intent": "report", "category": "waste"}

    # Keyword patterns for reports
    if re.search(r"\b(light|lamp|dark|bulb|flicker|lighting|streetlight)\b", m):
        return {"intent": "report", "category": "lighting"}
    if re.search(r"\b(pothole|road|crack|curb|kerb|pavement|footpath|hump)\b", m):
        return {"intent": "report", "category": "roads"}
    if re.search(r"\b(rubbish|bin|trash|waste|chute|litter|dirty|smell|dumped|clutter)\b", m):
        return {"intent": "report", "category": "waste"}
    if re.search(r"\b(noise|loud|shouting|drilling|renovation|music|hammering)\b", m):
        return {"intent": "report", "category": "noise"}
    if re.search(r"\b(park|tree|branch|grass|playground|swing|bench|mosquito|pest|rat|cockroach)\b", m):
        return {"intent": "report", "category": "parks"}
    if re.search(r"\b(water|pipe|leak|leaking|drain|flood|choked?|stagnant)\b", m):
        return {"intent": "report", "category": "water"}
    if re.search(r"\b(broken|spoil|spoilt|faulty|stuck|damage|hazard|lift)\b", m):
        return {"intent": "report", "category": "other"}

    return {"intent": "question", "category": "other"}


async def classify(message: str) -> dict[str, str]:
    fallback = _rule_based_classify(message)
    if _gemini_client:
        try:
            resp = await asyncio.to_thread(
                _gemini_client.models.generate_content,
                model=config.GEMINI_MODEL,
                contents=message,
                config={
                    "system_instruction": ROUTER_SYSTEM,
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                },
            )
            data = json.loads(resp.text)
            intent = data.get("intent") if data.get("intent") in INTENTS else fallback["intent"]
            category = data.get("category") if data.get("category") in CATEGORIES else fallback["category"]
            return {"intent": intent, "category": category}
        except Exception:
            return fallback

    if _anthropic_client:
        try:
            resp = await _anthropic_client.messages.create(
                model=config.ROUTER_MODEL,
                max_tokens=60,
                system=ROUTER_SYSTEM,
                messages=[{"role": "user", "content": message}],
            )
            raw = resp.content[0].text
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            intent = data.get("intent") if data.get("intent") in INTENTS else fallback["intent"]
            category = data.get("category") if data.get("category") in CATEGORIES else fallback["category"]
            return {"intent": intent, "category": category}
        except Exception:
            return fallback

    return fallback
