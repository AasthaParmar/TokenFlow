from evaluation.splits import create_splits


def test_split_sizes():
    for n in (150, 300):
        dataset = [{"id": i} for i in range(1, n + 1)]
        splits = create_splits(dataset, seed=42)
        assert len(splits["dev_ids"]) == int(n * 0.8)
        assert len(splits["test_ids"]) == n - int(n * 0.8)
    assert len(set(splits["dev_ids"]) & set(splits["test_ids"])) == 0
