from evaluation.splits import create_splits


def test_split_sizes():
    dataset = [{"id": i} for i in range(1, 151)]
    splits = create_splits(dataset, seed=42)
    assert len(splits["dev_ids"]) == 120
    assert len(splits["test_ids"]) == 30
    assert len(set(splits["dev_ids"]) & set(splits["test_ids"])) == 0
