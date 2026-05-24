import argparse
import asyncio
import json
import os
from pathlib import Path

import torch
from delphi.clients import OpenRouter
from delphi.explainers import DefaultExplainer
from delphi.latents.latents import ActivatingExample, Latent, LatentRecord
from delphi.latents.samplers import split_quantiles

from utils import DEFAULT_OUT_DIR

from server import LAYER, load_env

N_TRAIN = 5
N_TEST = 10
N_QUANTILES = 5
THRESHOLD = 0.6
TEMPERATURE = 0.7
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"


def _to_example(a: dict, global_max: float) -> ActivatingExample:
    values = torch.tensor(a["values"], dtype=torch.float)
    return ActivatingExample(
        tokens=torch.zeros(len(a["tokens"]), dtype=torch.long),
        str_tokens=a["tokens"],
        activations=values,
        normalized_activations=(values * 10 / global_max).ceil().clamp(0, 10),
    )


def load_record(path: Path) -> LatentRecord:
    acts = sorted(json.loads(path.read_text())["activations"], key=lambda a: a["maxValue"], reverse=True)
    global_max = max(acts[0]["maxValue"], 1e-6)
    examples = [_to_example(a, global_max) for a in acts]
    return LatentRecord(
        latent=Latent(module_name=LAYER, latent_index=int(acts[0]["index"])),
        train=examples[:N_TRAIN],
        test=split_quantiles(examples[N_TRAIN:], N_QUANTILES, N_TEST),
    )


async def generate(feature_id: int, raw_dir: Path, model: str) -> str:
    load_env()
    client = OpenRouter(model=model, api_key=os.environ["OPENROUTER_API_KEY"])
    explainer = DefaultExplainer(
        client=client,
        activations=True,
        cot=True,
        threshold=THRESHOLD,
        temperature=TEMPERATURE,
    )
    record = load_record(raw_dir / f"feature_{feature_id}.json")
    return (await explainer(record)).explanation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a label via delphi DefaultExplainer.")
    parser.add_argument("feature_id", type=int)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    explanation = asyncio.run(generate(args.feature_id, Path(args.out_dir) / "raw", args.model))
    print(explanation)


if __name__ == "__main__":
    main()
