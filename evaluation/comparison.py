"""Build baseline vs optimized comparison for dashboard and reports."""

import json
from pathlib import Path


def _run_key(payload: dict) -> str:
    return payload.get("timestamp") or payload.get("run_id") or ""


def latest_by_mode(payloads: list[dict]) -> dict[str, dict]:
    by_mode: dict[str, dict] = {}
    for p in sorted(payloads, key=_run_key):
        mode = p.get("mode")
        if mode:
            by_mode[mode] = p
    return by_mode


def load_test_results(results_dir: Path) -> list[dict]:
    payloads = []
    for path in sorted(results_dir.glob("*_test.json")):
        with path.open(encoding="utf-8") as f:
            payloads.append(json.load(f))
    return payloads


def _summary_row(payload: dict) -> dict:
    tokens = payload["total_input_tokens"] + payload["total_output_tokens"]
    latency = payload["total_latency_ms"] / max(payload["total_questions"], 1)
    return {
        "mode": payload["mode"],
        "split": payload.get("split", "test"),
        "total_questions": payload["total_questions"],
        "total_tokens": tokens,
        "llm_calls": payload["llm_calls"],
        "cache_hits": payload.get("cache_hits", 0),
        "avg_latency_ms": round(latency, 1),
        "estimated_cost_usd": payload["estimated_cost_usd"],
        "timestamp": payload.get("timestamp"),
    }


def build_comparison(results_dir: str | Path = "evaluation/results") -> dict | None:
    dir_path = Path(results_dir)
    payloads = load_test_results(dir_path)
    if not payloads:
        latest = dir_path / "latest.json"
        if latest.exists():
            with latest.open(encoding="utf-8") as f:
                payloads = [json.load(f)]

    by_mode = latest_by_mode(payloads)
    baseline = by_mode.get("baseline")
    combined = by_mode.get("combined")
    if not baseline or not combined:
        return None

    b = _summary_row(baseline)
    c = _summary_row(combined)

    def pct_of(value: float, base: float) -> float:
        if base == 0:
            return 0.0
        return round(value / base * 100, 1)

    def saved_pct(value: float, base: float) -> float:
        if base == 0:
            return 0.0
        return round((1 - value / base) * 100, 1)

    quality = 100.0
    judge_path = dir_path / "judge_latest.json"
    if judge_path.exists():
        with judge_path.open(encoding="utf-8") as f:
            quality = float(json.load(f).get("avg_overall_pct", 100.0))

    return {
        "baseline": b,
        "combined": c,
        "relative_to_baseline": {
            "tokens_pct": pct_of(c["total_tokens"], b["total_tokens"]),
            "cost_pct": pct_of(c["estimated_cost_usd"], b["estimated_cost_usd"]),
            "latency_pct": pct_of(c["avg_latency_ms"], b["avg_latency_ms"]),
        },
        "savings_vs_baseline": {
            "tokens_pct": saved_pct(c["total_tokens"], b["total_tokens"]),
            "cost_pct": saved_pct(c["estimated_cost_usd"], b["estimated_cost_usd"]),
            "latency_pct": saved_pct(c["avg_latency_ms"], b["avg_latency_ms"]),
        },
        "quality_pct": quality,
        "models_note": "Benchmarks use GEMINI_MODEL_SMALL / GEMINI_MODEL_LARGE from .env (Gemini 3.x).",
    }


def write_comparison_json(results_dir: str | Path = "evaluation/results") -> dict | None:
    comparison = build_comparison(results_dir)
    if not comparison:
        return None
    dir_path = Path(results_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    out_path = dir_path / "comparison.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    return comparison
