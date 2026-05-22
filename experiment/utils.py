import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import LAYER, MODEL_ID


DEFAULT_OUT_DIR = "data/experiments/default"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def save_json(path: Path, payload: dict) -> None:
    with path.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def feature_key(index: int) -> str:
    return f"{MODEL_ID}/{LAYER}/{index}"
