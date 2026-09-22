"""Cache threshold sweep and correctness audit."""

import argparse
import json
from pathlib import Path

from backend.cache.semantic import SemanticCache
from backend.gateway.pipeline import GatewayPipeline, PipelineConfig
from backend.llm.gemini import GeminiClient
from backend.schemas import ChatRequest
from database.models import clear_cache
from evaluation.splits import ensure_splits, load_dataset, load_splits

THRESHOLDS = [0.80, 0.90, 0.95, 0.98]


def is_correct_hit(matched_question: str | None, expected_base_question: str) -> bool:
    """Correct if the cache returned the answer for the intended base question."""
    if not matched_question:
        return False
    return matched_question.strip() == expected_base_question.strip()


def run_audit(split: str = "dev", limit: int | None = None) -> dict:
    ensure_splits()
    dataset = load_dataset()
    splits = load_splits()
    allowed = set(splits["dev_ids"] if split == "dev" else splits["test_ids"])

    cache_pairs = [item for item in dataset if item["category"] == "cache_pair" and item["id"] in allowed]
    if limit:
        cache_pairs = cache_pairs[:limit]

    client = GeminiClient()
    pipeline = GatewayPipeline(client)
    cfg = PipelineConfig(enable_cache=True, enable_rag_selection=False, enable_routing=False)

    report = {"thresholds": [], "split": split}

    try:
        clear_cache()
    except Exception:
        pass

    base_ids = {p["similar_to_id"] for p in cache_pairs}
    base_items = [item for item in dataset if item["id"] in base_ids]
    for item in base_items:
        req = ChatRequest(message=item["question"])
        pipeline.run(req, cfg)

    cache = SemanticCache(client)

    for threshold in THRESHOLDS:
        hits = 0
        correct = 0
        details = []

        for pair in cache_pairs:
            result = cache.lookup(pair["question"], threshold=threshold)
            if result.hit:
                hits += 1
                base = next(i for i in dataset if i["id"] == pair["similar_to_id"])
                ok = is_correct_hit(result.matched_question, base["question"])
                if ok:
                    correct += 1
                details.append(
                    {
                        "new_question": pair["question"],
                        "matched_question": result.matched_question,
                        "similarity": result.similarity_score,
                        "correct": ok,
                    }
                )

        total = len(cache_pairs)
        report["thresholds"].append(
            {
                "threshold": threshold,
                "cache_hit_rate": round(hits / total * 100, 1) if total else 0,
                "correct_hit_rate": round(correct / hits * 100, 1) if hits else 0,
                "hits": hits,
                "correct_hits": correct,
                "total_pairs": total,
                "details": details,
            }
        )

    out_path = Path("evaluation/results/cache_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Cache threshold sweep:")
    for row in report["thresholds"]:
        print(
            f"  {row['threshold']}: hit_rate={row['cache_hit_rate']}%, "
            f"correct_among_hits={row['correct_hit_rate']}%"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache threshold audit")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_audit(args.split, args.limit)


if __name__ == "__main__":
    main()
