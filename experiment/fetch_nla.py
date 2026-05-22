import argparse
import os
from dataclasses import asdict
from pathlib import Path

from utils import DEFAULT_OUT_DIR, append_jsonl, read_jsonl, utc_now

from server import LAYER, MODEL_ID, load_env, nla


def fetch_nla(out_dir: Path) -> None:
    load_env()
    os.environ["NEURONPEDIA_API_KEY"]

    features = read_jsonl(out_dir / "features.jsonl")
    if not features:
        raise ValueError(f"no features found in {out_dir / 'features.jsonl'}")

    nla_path = out_dir / "nla_examples.jsonl"
    fetched = {record["feature_key"] for record in read_jsonl(nla_path)}

    for i, feature in enumerate(features, 1):
        if feature["feature_key"] in fetched:
            print(f"SKIP {feature['index']} ({i}/{len(features)})")
            continue

        index = str(feature["index"])
        examples = nla(index)
        append_jsonl(
            nla_path,
            {
                "feature_key": feature["feature_key"],
                "index": int(index),
                "layer": LAYER,
                "model_id": MODEL_ID,
                "examples": [asdict(example) for example in examples],
                "fetched_at": utc_now(),
            },
        )
        print(f"OK {index} ({i}/{len(features)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NLA examples for sampled features.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    fetch_nla(Path(args.out_dir))


if __name__ == "__main__":
    main()
