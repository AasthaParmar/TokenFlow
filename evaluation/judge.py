"""LLM-as-judge quality evaluation."""

import argparse
import json
import random
import re
from pathlib import Path

from backend.llm.gemini import GeminiClient

JUDGE_PROMPT = """You are an evaluation judge. Score the candidate answer against the reference answer.

Question: {question}

Reference answer: {reference}

Candidate answer: {candidate}

Score on this rubric (respond ONLY with JSON):
- correctness: 0-4 (4=completely correct)
- completeness: 0-4 (4=fully complete)
- relevance: 0-2 (2=directly answers question)

Return JSON: {{"correctness": int, "completeness": int, "relevance": int, "notes": "brief reason"}}
"""


def parse_scores(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"correctness": 0, "completeness": 0, "relevance": 0, "notes": "parse error"}
    try:
        data = json.loads(match.group())
        return {
            "correctness": int(data.get("correctness", 0)),
            "completeness": int(data.get("completeness", 0)),
            "relevance": int(data.get("relevance", 0)),
            "notes": data.get("notes", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {"correctness": 0, "completeness": 0, "relevance": 0, "notes": "parse error"}


def overall_score(scores: dict) -> float:
    max_score = 4 + 4 + 2
    total = scores["correctness"] + scores["completeness"] + scores["relevance"]
    return round(total / max_score * 100, 2)


def judge_results(results_path: str, sample_size: int | None = None, seed: int = 42) -> dict:
    with open(results_path, encoding="utf-8") as f:
        payload = json.load(f)

    results = payload["results"]
    if sample_size is not None:
        k = min(sample_size, len(results))
        rng = random.Random(seed)
        results = rng.sample(results, k)

    client = GeminiClient()
    judged = []
    for item in results:
        prompt = JUDGE_PROMPT.format(
            question=item["question"],
            reference=item["reference_answer"],
            candidate=item["answer"],
        )
        raw = client.judge(prompt)
        scores = parse_scores(raw)
        judged.append(
            {
                "id": item["id"],
                "question": item["question"],
                "scores": scores,
                "overall_pct": overall_score(scores),
            }
        )

    avg_correctness = sum(j["scores"]["correctness"] for j in judged) / len(judged)
    avg_completeness = sum(j["scores"]["completeness"] for j in judged) / len(judged)
    avg_relevance = sum(j["scores"]["relevance"] for j in judged) / len(judged)
    avg_overall = sum(j["overall_pct"] for j in judged) / len(judged)

    output = {
        "source": results_path,
        "judged_count": len(judged),
        "avg_correctness": round(avg_correctness, 2),
        "avg_completeness": round(avg_completeness, 2),
        "avg_relevance": round(avg_relevance, 2),
        "avg_overall_pct": round(avg_overall, 2),
        "items": judged,
    }

    out_path = Path("evaluation/results") / "judge_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if sample_size is not None:
        manual_path = Path("evaluation/results") / "manual_validation_sample.json"
        with manual_path.open("w", encoding="utf-8") as f:
            json.dump(judged, f, indent=2)
        print(f"Manual validation sample saved to {manual_path}")

    print(
        f"Judge: correctness={output['avg_correctness']}, "
        f"completeness={output['avg_completeness']}, "
        f"relevance={output['avg_relevance']}, "
        f"overall={output['avg_overall_pct']}%"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge benchmark results")
    parser.add_argument("--results", default="evaluation/results/latest.json")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--manual-sample", type=int, default=20)
    args = parser.parse_args()

    if args.sample:
        judge_results(args.results, sample_size=args.sample, seed=42)
    else:
        judge_results(args.results)
        judge_results(
            args.results,
            sample_size=args.manual_sample,
            seed=42,
        )


if __name__ == "__main__":
    main()
