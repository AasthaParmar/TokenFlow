import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import settings


def log_request(record: dict[str, Any]) -> None:
    log_path = Path(settings.metrics_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
