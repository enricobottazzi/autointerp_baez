import argparse
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from utils import DEFAULT_OUT_DIR, append_jsonl, read_jsonl, utc_now

from server import (
    BASE_URL,
    LAYER,
    MODEL_ID,
    SYSTEM_PROMPT,
    NLAExplanation,
    build_user_prompt,
    call_openrouter,
    load_env,
    parse_explanation,
)


EXPLANATION_MODEL = "claude-sonnet-4.5" # needs to be compatible with Neuronpedia
BAEZ_EXPLANATION_MODEL = "anthropic/claude-sonnet-4.5" # needs to be compatible with OpenRouter
MAX_RETRIES = 3
METHODS = [
    "oai_token-act-pair",
    "np_max-act",
    "np_max-act-logits",
    "oai_attention-head",
    "baez",
]


def post_generate(api_key: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    for attempt in range(1, MAX_RETRIES + 1):
        request = Request(
            f"{BASE_URL}/api/explanation/generate",
            data=data,
            headers={
                "x-api-key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except HTTPError as error:
            body = error.read().decode(errors="replace")
            if error.code < 500 or attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Neuronpedia generate failed "
                    f"HTTP {error.code} for index={payload['index']} "
                    f"method={payload['explanationType']} after {attempt} attempt(s): {body}"
                ) from error
        except URLError as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Neuronpedia generate failed "
                    f"for index={payload['index']} method={payload['explanationType']} "
                    f"after {attempt} attempt(s): {error.reason}"
                ) from error
        print(
            "Retrying Neuronpedia generate "
            f"index={payload['index']} method={payload['explanationType']} "
            f"({attempt}/{MAX_RETRIES})"
        )
        time.sleep(2**attempt)


def generate_baez_label(feature: dict, nla_by_key: dict[str, dict], explanation_model: str) -> dict:
    nla_record = nla_by_key.get(feature["feature_key"])
    if not nla_record:
        raise ValueError(f"no NLA examples found for {feature['feature_key']}")

    examples = [NLAExplanation(**example) for example in nla_record["examples"]]
    response = call_openrouter(
        explanation_model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(examples)},
        ],
    )
    return {
        "modelId": MODEL_ID,
        "layer": LAYER,
        "index": str(feature["index"]),
        "explanation": parse_explanation(response),
    }


def model_for_method(method: str, explanation_model: str, baez_explanation_model: str) -> str:
    return baez_explanation_model if method == "baez" else explanation_model


def generate_labels(
    out_dir: Path,
    explanation_model: str,
    baez_explanation_model: str,
    methods: list[str],
) -> None:
    load_env()
    api_key = os.environ["NEURONPEDIA_API_KEY"] if any(method != "baez" for method in methods) else ""
    features = read_jsonl(out_dir / "features.jsonl")
    if not features:
        raise ValueError(f"no features found in {out_dir / 'features.jsonl'}")
    nla_by_key = {record["feature_key"]: record for record in read_jsonl(out_dir / "nla_examples.jsonl")}

    labels_path = out_dir / "labels.jsonl"
    completed = {
        (record["feature_key"], record["method"], record.get("explanation_model"))
        for record in read_jsonl(labels_path)
    }

    total = len(features) * len(methods)
    seen = 0
    for feature in features:
        for method in methods:
            seen += 1
            active_model = model_for_method(method, explanation_model, baez_explanation_model)
            if (feature["feature_key"], method, active_model) in completed:
                continue
            payload = {
                "modelId": MODEL_ID,
                "layer": LAYER,
                "index": str(feature["index"]),
                "explanationType": method,
                "explanationModelName": active_model,
            }
            try:
                result = (
                    generate_baez_label(feature, nla_by_key, baez_explanation_model)
                    if method == "baez"
                    else post_generate(api_key, payload)
                )
            except Exception as error:
                raise RuntimeError(
                    f"failed to generate label for {feature['feature_key']} "
                    f"index={feature['index']} method={method} ({seen}/{total})"
                ) from error
            append_jsonl(
                labels_path,
                {
                    "feature_key": feature["feature_key"],
                    "index": int(feature["index"]),
                    "layer": LAYER,
                    "method": method,
                    "model_id": MODEL_ID,
                    "explanation_model": active_model,
                    "result": result,
                    "generated_at": utc_now(),
                },
            )
            print(f"OK {feature['index']} {method} ({seen}/{total})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Neuronpedia labels for sampled features.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    args = parser.parse_args()

    generate_labels(
        Path(args.out_dir),
        EXPLANATION_MODEL,
        BAEZ_EXPLANATION_MODEL,
        args.methods,
    )


if __name__ == "__main__":
    main()
