"""Measures how well the RAG pipeline answers (and refuses). Run: python eval.py
Add 30-50 questions to tests.json for your report."""
import asyncio
import json

import rag


async def main():
    tests = json.load(open("tests.json", encoding="utf-8"))
    correct = 0
    for t in tests:
        res = await rag.answer(t["q"])
        answered = res["answer"] is not None
        if t.get("expect_no_answer"):
            ok = not answered
        else:
            ok = answered and all(k.lower() in res["answer"].lower() for k in t["must_contain"])
        correct += ok
        dist = f"{res['best_distance']:.2f}" if res["best_distance"] is not None else "n/a"
        print(f"{'PASS' if ok else 'FAIL'}  dist={dist}  {t['q']}")
    print(f"\nScore: {correct}/{len(tests)} = {correct / len(tests):.0%}")


asyncio.run(main())
