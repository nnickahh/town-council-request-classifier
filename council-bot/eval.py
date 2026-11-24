"""Evaluates the Town Council Semantic Hybrid RAG pipeline (retrieval, accuracy, and refusal).

Run:
    python eval.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import rag


async def main() -> None:
    tests_path = Path(__file__).resolve().parent / "tests.json"
    tests = json.loads(tests_path.read_text(encoding="utf-8"))

    engine = rag.get_engine()
    chunk_count = engine.initialize()
    print(f"RAG Backend: {engine.embedding_provider.backend_name} | Indexed Chunks: {chunk_count}\n")

    correct = 0
    in_scope_total = 0
    in_scope_pass = 0
    refusal_total = 0
    refusal_pass = 0
    hit_at_3 = 0
    mrr_sum = 0.0

    for t in tests:
        res = await rag.answer(t["q"])
        answered = res["answer"] is not None
        score = res.get("best_score", 0.0)
        dist = f"{res['best_distance']:.2f}" if res["best_distance"] is not None else "n/a"

        if t.get("expect_no_answer"):
            refusal_total += 1
            ok = not answered
            refusal_pass += int(ok)
        else:
            in_scope_total += 1
            ok = answered and all(k.lower() in res["answer"].lower() for k in t["must_contain"])
            in_scope_pass += int(ok)

            expected_doc = t.get("expected_doc")
            if expected_doc:
                top_files = [c["source_file"] for c in res.get("retrieved_chunks", [])[:3]]
                if expected_doc in top_files:
                    hit_at_3 += 1
                    mrr_sum += 1.0 / (top_files.index(expected_doc) + 1)

        correct += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status}  score={score:.2f} dist={dist}  {t['q']}")

    print("\n--- Evaluation Summary ---")
    if in_scope_total:
        print(f"Retrieval Hit Rate @ 3 : {hit_at_3}/{in_scope_total} = {hit_at_3 / in_scope_total:.0%}")
        print(f"Mean Reciprocal Rank   : {mrr_sum / in_scope_total:.3f}")
        print(f"In-Scope Answer Acc.   : {in_scope_pass}/{in_scope_total} = {in_scope_pass / in_scope_total:.0%}")
    if refusal_total:
        print(f"Out-of-Scope Refusal   : {refusal_pass}/{refusal_total} = {refusal_pass / refusal_total:.0%}")
    print(f"Overall Score          : {correct}/{len(tests)} = {correct / len(tests):.0%}")


if __name__ == "__main__":
    asyncio.run(main())
