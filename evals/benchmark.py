"""Run the golden dataset through the full analyze pipeline and print a report."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

CASES_PATH = Path(__file__).parent / "test_cases.json"
API_URL = "http://localhost:8000/analyze"


async def run() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    async with httpx.AsyncClient(timeout=120) as client:
        for case in cases:
            payload = {
                "pipeline": case["pipeline"],
                "stage": case["stage"],
                "log_excerpt": case["log"],
            }
            resp = await client.post(API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            ok = data["error_type"].lower() == case["expected_error_type"].lower()
            print(
                f"{case['id']:20} | expected={case['expected_error_type']:24} "
                f"got={data['error_type']:24} {'OK' if ok else 'MISS'} "
                f"({data['llm']['provider']}, {data['llm']['latency_ms']}ms)"
            )


if __name__ == "__main__":
    asyncio.run(run())
