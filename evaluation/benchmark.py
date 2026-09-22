"""Run baseline vs optimized benchmark modes."""

import argparse
import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.gateway.pipeline import GatewayPipeline, PipelineConfig
from backend.schemas import ChatRequest
from database.models import clear_cache
from evaluation.splits import ensure_splits, load_dataset, load_splits


MODES = {
    "baseline": PipelineConfig(False, False, False),
    "cache_only": PipelineConfig(True, False, False),
    "rag_only": PipelineConfig(False, True, False),
    "routing_only": PipelineConfig(False, False, True),
    "combined": PipelineConfig(True, True, True),
}

# Rough USD per 1M tokens (illustrative for Gemini flash)
COST_PER_M_INPUT = 0.10
COST_PER_M_OUTPUT = 0.40


@dataclass
class RunSummary:
    mode: str
    split: str
    total_questions: int
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float
    llm_calls: int
    cache_hits: int
    estimated_cost_usd: float
    results: list[dict]


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000 * COST_PER_M_INPUT) + (
        output_tokens / 1_000_000 * COST_PER_M_OUTPUT
    )


def run_benchmark(
    mode: str,
    split: str = "dev",
    limit: int | None = None,
    clear_cache_first: bool = False,
) -> RunSummary:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")

    ensure_splits()
    dataset = load_dataset()
    splits = load_splits()
    allowed = set(splits["dev_ids"] if split == "dev" else splits["test_ids"])
    items = [item for item in dataset if item["id"] in allowed]
    if limit:
        items = items[:limit]

    if clear_cache_first and mode in ("cache_only", "combined"):
        try:
            clear_cache()
        except Exception:
            pass

    pipeline = GatewayPipeline()
    cfg = MODES[mode]
    results = []
    total_in = total_out = 0
    total_latency = 0.0
    llm_calls = cache_hits = 0

    for item in items:
        request = ChatRequest(
            message=item["question"],
            context_chunks=item.get("context_chunks", []),
            relevant_chunk_ids=item.get("relevant_chunk_ids") or None,
        )
        start = time.perf_counter()
        response = pipeline.run(request, cfg)
        elapsed = (time.perf_counter() - start) * 1000

        total_in += response.input_tokens
        total_out += response.output_tokens
        total_latency += elapsed
        if not response.cache_hit:
            llm_calls += 1
        if response.cache_hit:
            cache_hits += 1

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": response.answer,
                "reference_answer": item["reference_answer"],
                "category": item["category"],
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": elapsed,
                "cache_hit": response.cache_hit,
                "llm_called": not response.cache_hit,
                "model": response.model,
            }
        )
        time.sleep(0.1)

    summary = RunSummary(
        mode=mode,
        split=split,
        total_questions=len(items),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_latency_ms=total_latency,
        llm_calls=llm_calls,
        cache_hits=cache_hits,
        estimated_cost_usd=estimate_cost(total_in, total_out),
        results=results,
    )

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    out_path = out_dir / f"{run_id}_{mode}_{split}.json"
    payload = asdict(summary)
    payload["run_id"] = run_id
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    latest = out_dir / "latest.json"
    with latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {out_path}")
    print(
        f"{mode} ({split}): questions={summary.total_questions}, "
        f"tokens={total_in + total_out}, llm_calls={llm_calls}, "
        f"cache_hits={cache_hits}, cost=${summary.estimated_cost_usd:.4f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TokenFlow benchmark")
    parser.add_argument("--mode", choices=list(MODES.keys()), default="baseline")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()
    run_benchmark(args.mode, args.split, args.limit, args.clear_cache)


if __name__ == "__main__":
    main()
