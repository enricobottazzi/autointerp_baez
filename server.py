import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from math import ceil
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


MODEL_ID = "gemma-3-27b-it"
LAYER = "41-gemmascope-2-res-262k"
NLA_SOURCE_ID = "kitft-l41"
BASE_URL = "https://www.neuronpedia.org"
EXPLANATION_MARKER = "[EXPLANATION]:"

SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an short and concise label that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given 20 text examples. Each example has one activation value from 1 to 10, where higher values indicate stronger relevance to the latent/feature being explained.

- Produce a very concise final label. Simply describe the text latent common in the examples, and what pattern you found.
- Give more weight to examples with higher activation values.
- If the examples are uninformative, you don't need to mention them.
- Do not make lists of possible explanations. Keep your explanation short and concise.
- The final label must be at most 8 words.
- The last line of your response must be formatted exactly as:

[EXPLANATION]: <explanation>"""


@dataclass(frozen=True)
class NLAExplanation:
    text: str
    activation: int


def normalize_activation(raw_activation: float, max_activation: float) -> int:
    # score = ceil(raw_activation * 10 / max_activation) where max_activation is the maximum activation score across the 20 examples
    return ceil(raw_activation * 10 / max_activation) 

def build_user_prompt(examples: list[NLAExplanation]) -> str:
    if len(examples) != 20:
        raise ValueError("expected exactly 20 examples")
    return "\n\n".join(
        f"Example {i}: {example.text}\nActivation: {example.activation}"
        for i, example in enumerate(examples, 1)
    )


def parse_explanation(response: str) -> str:
    if EXPLANATION_MARKER not in response:
        return "Explanation could not be parsed."
    return response.rsplit(EXPLANATION_MARKER, 1)[1].strip()


def load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as file:
        for line in file:
            key, sep, value = line.strip().partition("=")
            if sep and key not in os.environ:
                os.environ[key] = value


def nla(index: str) -> list[NLAExplanation]:
    api_key = os.environ["NEURONPEDIA_API_KEY"]
    feature = fetch_neuronpedia_feature(api_key, index)
    activations = sorted(
        feature.get("activations", []),
        key=lambda item: item.get("maxValue", float("-inf")),
        reverse=True,
    )[:20]
    max_activation = max(item["maxValue"] for item in activations)
    return [
        NLAExplanation(
            text=" | ".join(filter(None, nla_descriptions(api_key, activation))),
            activation=normalize_activation(activation["maxValue"], max_activation),
        )
        for activation in activations
    ]


def fetch_neuronpedia_feature(api_key: str, index: str) -> dict:
    parts = (MODEL_ID, LAYER, index)
    path = "/api/feature/" + "/".join(quote(part, safe="") for part in parts)
    request = Request(f"{BASE_URL}{path}", headers={"x-api-key": api_key, "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def nla_descriptions(api_key: str, activation: dict) -> list[str | None]:
    text, position = trim_to_max_token(activation)
    payload = post_neuronpedia(
        api_key,
        "/api/nla/explain",
        {
            "modelId": MODEL_ID,
            "nlaSourceId": NLA_SOURCE_ID,
            "text": text,
            "positions": [position],
            "temperature": 0.7,
        },
    )
    return [item.get("description") for item in payload.get("results", [])]


def trim_to_max_token(activation: dict) -> tuple[str, int]:
    tokens = activation.get("tokens", [])
    index = min(activation.get("maxValueTokenIndex", 0), len(tokens) - 1)
    return "".join(tokens[: index + 1]), index


def post_neuronpedia(api_key: str, path: str, payload: dict) -> dict:
    request = Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def call_openrouter(model: str, messages: list[dict[str, str]]) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "temperature": 0.7}).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read())
    return payload["choices"][0]["message"]["content"]


def generate_explanation(body: dict) -> dict:
    examples = nla(str(body["index"]))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(examples)},
    ]
    explanation = parse_explanation(call_openrouter(body["explanationModelName"], messages))
    return {
        "modelId": MODEL_ID,
        "layer": LAYER,
        "index": body["index"],
        "explanation": explanation,
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/api/explanation/generate":
            return self.respond(404, {"error": "not found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            response = generate_explanation(json.loads(self.rfile.read(length)))
            self.respond(200, response)
        except NotImplementedError as error:
            self.respond(501, {"error": str(error)})
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self.respond(400, {"error": str(error)})
        except HTTPError as error:
            self.respond(error.code, {"error": error.read().decode()})

    def respond(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    load_env()
    HTTPServer(("localhost", int(os.getenv("PORT", "8000"))), Handler).serve_forever()
