from collections import Counter

from evaluation.generate_dataset import (
    CATEGORY_RATIOS,
    build_dataset,
    factual_cs_slot_count,
)


def test_unique_question_strings_at_1000():
    dataset = build_dataset(1000)
    questions = [x["question"] for x in dataset]
    assert len(questions) == len(set(questions))
    cache_pairs = [x for x in dataset if x["category"] == "cache_pair"]
    assert len(cache_pairs) == 150
    for pair in cache_pairs:
        base = next(x for x in dataset if x["id"] == pair["similar_to_id"])
        assert pair["question"] != base["question"]


def test_distinct_concepts_at_1000():
    dataset = build_dataset(1000)
    refs = {x["reference_answer"] for x in dataset}
    assert len(refs) >= 280
    factual = [x for x in dataset if x["category"] == "factual"]
    assert len({x["reference_answer"] for x in factual}) >= 230
    assert max(Counter(x["reference_answer"] for x in dataset).values()) <= 5


def test_balanced_category_mix_at_1000():
    dataset = build_dataset(1000)
    counts = Counter(item["category"] for item in dataset)
    assert sum(counts.values()) == 1000
    for category, ratio in CATEGORY_RATIOS.items():
        assert abs(counts[category] - int(1000 * ratio)) <= 2


def test_balanced_category_mix_at_500():
    dataset = build_dataset(500)
    counts = Counter(item["category"] for item in dataset)
    assert sum(counts.values()) == 500
    for category, ratio in CATEGORY_RATIOS.items():
        assert abs(counts[category] - int(500 * ratio)) <= 1


def test_factual_cs_vs_general_at_500():
    from evaluation.generate_dataset import FACTUAL_CS

    assert factual_cs_slot_count(175) == 60
    dataset = build_dataset(500)
    factual = [x for x in dataset if x["category"] == "factual"]
    cs_refs = {ref for _, ref in FACTUAL_CS}
    cs_count = sum(1 for x in factual if x["reference_answer"] in cs_refs)
    assert cs_count == 60
    assert len(factual) - cs_count == 115
