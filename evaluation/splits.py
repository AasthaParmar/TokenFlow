import json
import random
from pathlib import Path

SPLIT_SEED = 42
DEV_RATIO = 0.8
SPLITS_PATH = Path("evaluation/splits.json")


def load_dataset(path: str = "evaluation/dataset.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def create_splits(dataset: list[dict], seed: int = SPLIT_SEED) -> dict:
    ids = [item["id"] for item in dataset]
    rng = random.Random(seed)
    rng.shuffle(ids)
    split_idx = int(len(ids) * DEV_RATIO)
    return {
        "seed": seed,
        "dev_ids": sorted(ids[:split_idx]),
        "test_ids": sorted(ids[split_idx:]),
    }


def save_splits(splits: dict, path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: Path = SPLITS_PATH) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_split_items(dataset: list[dict], split: str) -> list[dict]:
    splits = load_splits()
    ids = set(splits["dev_ids"] if split == "dev" else splits["test_ids"])
    return [item for item in dataset if item["id"] in ids]


def ensure_splits(dataset_path: str = "evaluation/dataset.json") -> dict:
    if SPLITS_PATH.exists():
        return load_splits()
    dataset = load_dataset(dataset_path)
    splits = create_splits(dataset)
    save_splits(splits)
    return splits
