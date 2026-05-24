import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from delphi.latents.latents import ActivatingExample, Latent, LatentRecord


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import LAYER, MODEL_ID


DEFAULT_OUT_DIR = "data/experiments/default"

N_TRAIN = 5
N_TEST = 10


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


def _to_example(a: dict, global_max: float) -> ActivatingExample:
    values = torch.tensor(a["values"], dtype=torch.float)
    return ActivatingExample(
        tokens=torch.zeros(len(a["tokens"]), dtype=torch.long),
        str_tokens=a["tokens"],
        activations=values,
        normalized_activations=(values * 10 / global_max).ceil().clamp(0, 10),
    )


def load_record(path: Path) -> LatentRecord:
    acts = sorted(
        (a for a in json.loads(path.read_text())["activations"] if a.get("maxValue", 0) > 0),
        key=lambda a: a["maxValue"], reverse=True,
    )
    global_max = max(acts[0]["maxValue"], 1e-6)
    examples = [_to_example(a, global_max) for a in acts]
    return LatentRecord(
        latent=Latent(module_name=LAYER, latent_index=int(acts[0]["index"])),
        train=examples[:N_TRAIN],
        test=examples[N_TRAIN : N_TRAIN + N_TEST],
    )
