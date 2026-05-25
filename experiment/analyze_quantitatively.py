"""Rank features by cosine distance between two methods' label embeddings."""
import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from utils import read_jsonl

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METHODS = ("baez", "delphi")


def labels_by_feature(labels: list[dict], methods: tuple[str, str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in labels:
        if r["method"] in methods:
            out.setdefault(r["feature_key"], {})[r["method"]] = r["result"]["explanation"]
    return {k: v for k, v in out.items() if set(v) == set(methods)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp-dir", default="data/experiments/exp_3")
    args = p.parse_args()

    exp = Path(args.exp_dir)
    a, b = METHODS
    pairs = labels_by_feature(read_jsonl(exp / "labels.jsonl"), METHODS)
    if not pairs:
        raise SystemExit(f"no features with both methods {a!r} and {b!r}")

    keys = sorted(pairs)
    texts_a = [pairs[k][a] for k in keys]
    texts_b = [pairs[k][b] for k in keys]

    emb = SentenceTransformer(EMB_MODEL)
    ea = emb.encode(texts_a, convert_to_tensor=True, normalize_embeddings=True)
    eb = emb.encode(texts_b, convert_to_tensor=True, normalize_embeddings=True)
    sims = cos_sim(ea, eb).diagonal().cpu().tolist()

    df = pd.DataFrame({
        "feature_key": keys,
        f"label__{a}": texts_a,
        f"label__{b}": texts_b,
        "cosine_similarity": sims,
        "cosine_distance": [1 - s for s in sims],
    }).sort_values("cosine_distance", ascending=False).reset_index(drop=True)

    out_path = exp / f"label_distance__{a}__vs__{b}.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
