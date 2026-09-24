"""Generate final benchmark comparison report."""

import argparse
import json
from pathlib import Path

from evaluation.comparison import latest_by_mode, load_test_results, write_comparison_json


def load_result_files(results_dir: Path) -> list[dict]:
    payloads = load_test_results(results_dir)
    by_mode = latest_by_mode(payloads)
    return list(by_mode.values()) if by_mode else payloads


def pct(value: float, baseline: float) -> str:
    if baseline == 0:
        return "N/A"
    return f"{value / baseline * 100:.1f}%"


def generate_report(results_dir: str = "evaluation/results") -> str:
    dir_path = Path(results_dir)
    payloads = load_result_files(dir_path)

    if not payloads:
        # Fall back to any latest results for demo
        latest = dir_path / "latest.json"
        if latest.exists():
            with latest.open(encoding="utf-8") as f:
                payloads = [json.load(f)]

    if not payloads:
        return "No benchmark results found."

    baseline = next((p for p in payloads if p["mode"] == "baseline"), payloads[0])
    b_tokens = baseline["total_input_tokens"] + baseline["total_output_tokens"]
    b_latency = baseline["total_latency_ms"] / max(baseline["total_questions"], 1)
    b_cost = baseline["estimated_cost_usd"]
    b_quality_path = dir_path / "judge_latest.json"
    b_quality = 100.0
    if b_quality_path.exists():
        with b_quality_path.open(encoding="utf-8") as f:
            b_quality = json.load(f).get("avg_overall_pct", 100.0)

    lines = [
        "# TokenFlow Benchmark Report",
        "",
        "Results on held-out test set (or latest available run).",
        "",
        "| Configuration | Tokens | LLM Calls | Latency/query | Cost | Quality |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for p in payloads:
        tokens = p["total_input_tokens"] + p["total_output_tokens"]
        latency = p["total_latency_ms"] / max(p["total_questions"], 1)
        quality = b_quality if p["mode"] != "baseline" else 100.0
        lines.append(
            f"| {p['mode']} | {pct(tokens, b_tokens)} | "
            f"{p['llm_calls']}/{p['total_questions']} | "
            f"{pct(latency, b_latency)} | "
            f"{pct(p['estimated_cost_usd'], b_cost)} | "
            f"{quality:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- Token and cost percentages are relative to baseline.",
            "- Quality from LLM-as-judge rubric (correctness, completeness, relevance).",
            "- Threshold tuning uses dev set; test set numbers are for final reporting.",
            "",
        ]
    )

    report = "\n".join(lines)
    out_md = dir_path / "REPORT.md"
    out_csv = dir_path / "REPORT.csv"
    out_md.write_text(report, encoding="utf-8")

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("mode,tokens_pct,llm_calls,latency_pct,cost_pct,quality_pct\n")
        for p in payloads:
            tokens = p["total_input_tokens"] + p["total_output_tokens"]
            latency = p["total_latency_ms"] / max(p["total_questions"], 1)
            quality = b_quality if p["mode"] != "baseline" else 100.0
            f.write(
                f"{p['mode']},{pct(tokens, b_tokens)},{p['llm_calls']},"
                f"{pct(latency, b_latency)},{pct(p['estimated_cost_usd'], b_cost)},"
                f"{quality:.1f}\n"
            )

    write_comparison_json(dir_path)

    print(report)
    print(f"\nSaved {out_md} and {out_csv}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results-dir", default="evaluation/results")
    args = parser.parse_args()
    generate_report(args.results_dir)


if __name__ == "__main__":
    main()
