import argparse
import asyncio
import os
from pathlib import Path

from delphi.clients import OpenRouter
from delphi.explainers import DefaultExplainer

from utils import DEFAULT_OUT_DIR, N_TRAIN, append_jsonl, load_record, read_jsonl, utc_now

from server import (
    LAYER,
    MODEL_ID,
    NLAExplanation,
    build_system_prompt,
    build_user_prompt,
    call_openrouter,
    load_env,
    parse_explanation,
)


EXPLANATION_MODEL = "anthropic/claude-sonnet-4.5"
METHODS = ["delphi", "baez", "baez_last"]
DELPHI_THRESHOLD = 0.6
DELPHI_TEMPERATURE = 0.7


def baez_label(examples: list[NLAExplanation], model: str, last: bool = False) -> str:
    if last:
        examples = [
            NLAExplanation(text=e.text[e.text.rfind("\nFinal token") + 1:], activation=e.activation)
            for e in examples
        ]
    return parse_explanation(call_openrouter(model, [
        {"role": "system", "content": build_system_prompt(N_TRAIN)},
        {"role": "user", "content": build_user_prompt(examples)},
    ]))


async def delphi_label(raw_path: Path, model: str) -> str:
    client = OpenRouter(model=model, api_key=os.environ["OPENROUTER_API_KEY"])
    explainer = DefaultExplainer(client=client, activations=True, cot=True,
                                 threshold=DELPHI_THRESHOLD, temperature=DELPHI_TEMPERATURE)
    return (await explainer(load_record(raw_path))).explanation


def generate_labels(out_dir: Path, model: str, methods: list[str]) -> None:
    load_env()
    features = read_jsonl(out_dir / "features.jsonl")
    if not features:
        raise ValueError(f"no features found in {out_dir / 'features.jsonl'}")
    nla_by_feature = {
        r["feature_key"]: [NLAExplanation(**example) for example in r["examples"][:N_TRAIN]]
        for r in read_jsonl(out_dir / "nla_examples.jsonl")
    }
    raw_dir, labels_path = out_dir / "raw", out_dir / "labels.jsonl"
    completed = {(r["feature_key"], r["method"], r.get("explanation_model"))
                 for r in read_jsonl(labels_path)}

    total, seen = len(features) * len(methods), 0
    for feature in features:
        for method in methods:
            seen += 1
            if (feature["feature_key"], method, model) in completed:
                continue
            try:
                if method.startswith("baez"):
                    baez_examples = nla_by_feature.get(feature["feature_key"])
                    if not baez_examples:
                        raise ValueError(f"missing NLA examples for {feature['feature_key']}")
                    explanation = baez_label(baez_examples, model, method == "baez_last")
                else:
                    explanation = asyncio.run(delphi_label(raw_dir / f"feature_{feature['index']}.json", model))
            except Exception as error:
                raise RuntimeError(
                    f"failed for {feature['feature_key']} method={method} ({seen}/{total})"
                ) from error
            append_jsonl(labels_path, {
                "feature_key": feature["feature_key"],
                "index": int(feature["index"]),
                "layer": LAYER,
                "method": method,
                "model_id": MODEL_ID,
                "explanation_model": model,
                "result": {"explanation": explanation},
                "generated_at": utc_now(),
            })
            print(f"OK {feature['index']} {method} ({seen}/{total})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labels via delphi or baez.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    generate_labels(Path(args.out_dir), EXPLANATION_MODEL, METHODS)


if __name__ == "__main__":
    main()
