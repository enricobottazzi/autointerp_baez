import argparse
import os
import random
from pathlib import Path
from urllib.error import HTTPError

from utils import DEFAULT_OUT_DIR, append_jsonl, feature_key, save_json, utc_now

from server import LAYER, MODEL_ID, fetch_neuronpedia_feature, load_env


FEATURE_SPACE_SIZE = 262_000


def sample_features(count: int, out_dir: Path, max_index: int, seed: int | None) -> None:
    if count < 1:
        raise ValueError("count must be positive")

    load_env()
    rng = random.Random(seed)
    api_key = os.environ["NEURONPEDIA_API_KEY"]
    raw_dir = out_dir / "raw"
    features_path = out_dir / "features.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    features_path.write_text("")

    sampled: set[int] = set()
    tried: set[int] = set()

    while len(sampled) < count:
        if len(tried) >= max_index:
            raise RuntimeError(f"exhausted {max_index} candidate feature IDs")

        index = rng.randrange(max_index)
        if index in tried:
            continue
        tried.add(index)

        try:
            feature = fetch_neuronpedia_feature(api_key, str(index))
        except HTTPError as error:
            if error.code == 404:
                print(f"MISS {index}")
                continue
            raise
        if not feature.get("activations"):
            print(f"MISS {index} (no activations)")
            continue

        raw_path = raw_dir / f"feature_{index}.json"
        save_json(raw_path, feature)
        append_jsonl(
            features_path,
            {
                "feature_key": feature_key(index),
                "index": index,
                "layer": LAYER,
                "model_id": MODEL_ID,
                "raw_path": str(raw_path.relative_to(out_dir)),
                "sampled_at": utc_now(),
                "source": "random",
            },
        )
        sampled.add(index)
        print(f"OK {index} ({len(sampled)}/{count})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample random Neuronpedia features.")
    parser.add_argument("count", type=int, help="Number of valid features to sample.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-index", type=int, default=FEATURE_SPACE_SIZE)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    sample_features(args.count, Path(args.out_dir), args.max_index, args.seed)


if __name__ == "__main__":
    main()
