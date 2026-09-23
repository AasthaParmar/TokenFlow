from collections import Counter

from evaluation.generate_dataset import CATEGORY_RATIOS, build_dataset, factual_cs_slot_count


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
