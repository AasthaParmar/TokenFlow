"""Cache threshold sweep and correctness audit."""

import argparse
import json
import time
from pathlib import Path

from backend.cache.semantic import SemanticCache
from backend.config import settings
from backend.llm.gemini import GeminiClient
from database.models import clear_cache
from evaluation.splits import ensure_splits, load_dataset, load_splits

# Spacing API calls avoids 429/503 bursts (paid tier can use lower delays).
EMBED_DELAY_SEC = 1.0
GENERATE_DELAY_SEC = 2.0

THRESHOLDS = [0.80, 0.90, 0.95, 0.98]


def is_correct_hit(matched_question: str | None, expected_base_question: str) -> bool:
    """Correct if the cache returned the answer for the intended base question."""
    if not matched_question:
        return False
    return matched_question.strip() == expected_base_question.strip()


def run_audit(
    split: str = "dev",
    limit: int | None = None,
    embed_delay: float = EMBED_DELAY_SEC,
    generate_delay: float = GENERATE_DELAY_SEC,
) -> dict:
    ensure_splits()
    dataset = load_dataset()
    splits = load_splits()
    allowed = set(splits["dev_ids"] if split == "dev" else splits["test_ids"])

    cache_pairs = [item for item in dataset if item["category"] == "cache_pair" and item["id"] in allowed]
    if limit:
        cache_pairs = cache_pairs[:limit]

    client = GeminiClient()
    cache = SemanticCache(client)
    report = {"thresholds": [], "split": split}

    try:
        clear_cache()
    except Exception:
        pass

    base_ids = {p["similar_to_id"] for p in cache_pairs}
    by_id = {item["id"]: item for item in dataset}
    base_items: list[dict] = []
    seeded_questions: set[str] = set()
    for base_id in sorted(base_ids):
        item = by_id[base_id]
        if item["question"] in seeded_questions:
            continue
        seeded_questions.add(item["question"])
        base_items.append(item)
    model = settings.gemini_model_small
    print(
        f"Seeding cache with {len(base_items)} unique base questions "
        f"({len(base_ids)} cache-pair links, LLM via {model})..."
    )
    for i, item in enumerate(base_items):
        print(f"  Seed [{i + 1}/{len(base_items)}] {item['question'][:60]}...", flush=True)
        response = client.generate(item["question"], model=model)
        cache.store(item["question"], response.answer)
        if i + 1 < len(base_items):
            time.sleep(generate_delay)

    for threshold in THRESHOLDS:
        hits = 0
        correct = 0
        details = []

        for j, pair in enumerate(cache_pairs):
            if j > 0:
                time.sleep(embed_delay)
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
    parser.add_argument(
        "--embed-delay",
        type=float,
        default=EMBED_DELAY_SEC,
        help="Seconds between cache lookup embedding calls",
    )
    parser.add_argument(
        "--generate-delay",
        type=float,
        default=GENERATE_DELAY_SEC,
        help="Seconds between LLM seed calls when filling the cache",
    )
    args = parser.parse_args()
    run_audit(
        args.split,
        args.limit,
        embed_delay=args.embed_delay,
        generate_delay=args.generate_delay,
    )


if __name__ == "__main__":
    main()
