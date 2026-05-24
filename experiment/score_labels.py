import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import torch
from delphi.clients import OpenRouter
from delphi.latents.latents import NonActivatingExample
from delphi.scorers import DetectionScorer, EmbeddingScorer, FuzzingScorer
from sentence_transformers import SentenceTransformer

from utils import DEFAULT_OUT_DIR, append_jsonl, load_record, read_jsonl, utc_now

from server import LAYER, MODEL_ID, load_env


SCORER_MODEL = "anthropic/claude-sonnet-4.6"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_NEG = 10 # number of non-activating examples to sample for each feature
N_SHOWN = 5 # number of examples (activating or non-activating) to show to the scorer at each call

def neg_pool(raw_dir: Path, exclude: int, rng: random.Random, k: int) -> list[NonActivatingExample]:
    paths = [p for p in raw_dir.glob("feature_*.json") if p.name != f"feature_{exclude}.json"]
    negs: list[NonActivatingExample] = []
    for path in rng.sample(paths, min(k, len(paths))):
        acts = [a for a in json.loads(path.read_text())["activations"] if a.get("maxValue", 0) > 0]
        if not acts:
            continue
        a = rng.choice(acts)
        v = torch.tensor(a["values"], dtype=torch.float)
        neg = NonActivatingExample(
            tokens=torch.zeros(len(a["tokens"]), dtype=torch.long),
            str_tokens=a["tokens"], activations=v, distance=-1.0,
        )
        neg.source_index = int(path.stem.removeprefix("feature_"))
        negs.append(neg)
    return negs


def cls_summary(score: list) -> dict:
    return {"accuracy": sum(s.correct or 0 for s in score) / max(len(score), 1), "n": len(score)}


def emb_summary(score: list) -> dict:
    pos = [s.similarity for s in score if s.distance >= 0]
    neg = [s.similarity for s in score if s.distance < 0]
    p, n = sum(pos) / max(len(pos), 1), sum(neg) / max(len(neg), 1)
    return {"mean_pos": p, "mean_neg": n, "gap": p - n}


CTX = 6  # tokens of context around the peak on each side


def preview(ex) -> str:
    acts = ex.activations.tolist()
    i = max(range(len(acts)), key=acts.__getitem__)
    lo, hi = max(0, i - CTX), min(len(ex.str_tokens), i + CTX + 1)
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(ex.str_tokens) else ""
    return f"max={acts[i]:.3f} | {prefix}{''.join(ex.str_tokens[lo:hi])}{suffix}"


def log_record(record) -> None:
    test = record.test or []
    print(f"  latent: {record.latent.module_name}/{record.latent.latent_index}")
    print(f"  explanation: {record.explanation!r}")
    print(f"  train (top-{len(record.train)} by maxValue):")
    for j, ex in enumerate(record.train):
        print(f"    [{j}] {preview(ex)}")
    print(f"  test ({len(test)}):")
    for j, ex in enumerate(test):
        print(f"    [{j}] {preview(ex)}")
    print(f"  not_active ({len(record.not_active)}):")
    for j, ex in enumerate(record.not_active):
        print(f"    [{j}|src={getattr(ex, 'source_index', '?')}] {preview(ex)}")


async def score_one(record, llm, emb) -> dict:
    return {
        "detection": cls_summary((await DetectionScorer(llm, n_examples_shown=N_SHOWN)(record)).score),
        "fuzz":      cls_summary((await FuzzingScorer(llm, n_examples_shown=N_SHOWN)(record)).score),
        "embedding": emb_summary((await EmbeddingScorer(emb)(record)).score),
    }


async def run(out_dir: Path, seed: int) -> None:
    raw_dir, scores_path = out_dir / "raw", out_dir / "scores.jsonl"
    emb = SentenceTransformer(EMB_MODEL)
    llm = OpenRouter(model=SCORER_MODEL, api_key=os.environ["OPENROUTER_API_KEY"])

    completed = {(r["feature_key"], r["method"], r["explanation_model"]) for r in read_jsonl(scores_path)}
    labels = read_jsonl(out_dir / "labels.jsonl")
    if not labels:
        raise ValueError(f"no labels found in {out_dir / 'labels.jsonl'}")

    for i, label in enumerate(labels, 1):
        key = (label["feature_key"], label["method"], label["explanation_model"])
        if key in completed:
            print(f"SKIP {label['index']} {label['method']} ({i}/{len(labels)})")
            continue
        record = load_record(raw_dir / f"feature_{label['index']}.json")
        record.not_active = neg_pool(raw_dir, label["index"], random.Random(f"{seed}:{label['index']}"), N_NEG)
        record.explanation = label["result"]["explanation"]
        log_record(record)
        scores = await score_one(record, llm, emb)
        append_jsonl(scores_path, {
            "feature_key": label["feature_key"],
            "index": int(label["index"]),
            "layer": LAYER,
            "method": label["method"],
            "model_id": MODEL_ID,
            "explanation_model": label["explanation_model"],
            "scorer_model": SCORER_MODEL,
            "scores": scores,
            "scored_at": utc_now(),
        })
        print(f"OK {label['index']} {label['method']} ({i}/{len(labels)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated labels with detection/fuzz/embedding.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    load_env()
    asyncio.run(run(Path(args.out_dir), args.seed))


if __name__ == "__main__":
    main()
