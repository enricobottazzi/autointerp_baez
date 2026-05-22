# autointerp_baez

Specs and experiment results available [here](https://docs.google.com/document/d/1PO22yjObWHlsV88DrZJdJZHCjWJYKRkYaurWRX3p0R0/edit?usp=sharing).

## How to use

This repo exposes a minimal autointerp service for generating a natural-language label for a Neuronpedia feature based on NLA explanations.

Requirement: the SAE feature for which the label is to be generated must belong to `gemma-3-27b-it/41-gemmascope-2-res-262k` as this is the only model and layer for which an NLA is available.

As a user, you need to:

1. Add API keys to `.env`:

```env
NEURONPEDIA_API_KEY=<your-neuronpedia-api-key>
OPENROUTER_API_KEY=<your-openrouter-api-key>
```

2. Launch the server:

```sh
python server.py
```

3. Request an autointerp explanation for a feature with an API call shaped like Neuronpedia's explanation generate [endpoint](https://www.neuronpedia.org/api-doc#tag/explanations/POST/api/explanation/generate):

```sh
curl -X POST http://localhost:8000/api/explanation/generate \
  -H "Content-Type: application/json" \
  -d '{
    "index": "43",
    "explanationModelName": "google/gemini-2.5-flash-lite"
  }'
```

Notice that `modelId`  and `layer` are set by default to `gemma-3-27b-it` and `41-gemmascope-2-res-262k`. `explanationModelName` is the OpenRouter model ID used to generate the explanation.

Expected response:

```json
{
  "modelId": "gemma-3-27b-it",
  "layer": "41-gemmascope-2-res-262k",
  "index": "43",
  "explanation": "<natural-language explanation>"
}
```

## Running an experiment

Experiment scripts write local artifacts under `data/experiments/<name>/`.

### 1. Sampling features


Sample `100` random features from the `gemma-3-27b-it/41-gemmascope-2-res-262k` search space:

```sh
python experiment/sample_features.py 100 --out-dir data/experiments/exp_1 --seed 0
```

### 2. Generate NLA explanations

Fetch NLA-derived examples for the sampled features (resumable):

```sh
python experiment/fetch_nla.py --out-dir data/experiments/exp_1
```

### 3. Generate labels

Generate Neuronpedia labels for each sampled feature with the configured methods (resumable):

```sh
python experiment/generate_labels.py --out-dir data/experiments/exp_1
```

Notice that this script will fail if the label for a given combination of `modelId`, `layer`, `index`, `explanationType`, and `explanationModelName` has already been generated.