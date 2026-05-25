# autointerp_baez

Specs and experiment results available [here](https://docs.google.com/document/d/1PO22yjObWHlsV88DrZJdJZHCjWJYKRkYaurWRX3p0R0/edit?usp=sharing).

## How to use

This repo exposes a minimal autointerp service for generating a natural-language label for a Neuronpedia feature based on NLA explanations.

Requirement: the SAE feature for which the label is to be generated must belong to `gemma-3-27b-it/41-gemmascope-2-res-262k` as this is the only model and layer for which an NLA is available.

As a user, you need to:

1. Install dependencies:

```sh
pip install -r requirements.txt
```

2. Add API keys to `.env`:

```env
NEURONPEDIA_API_KEY=<your-neuronpedia-api-key>
OPENROUTER_API_KEY=<your-openrouter-api-key>
```

3. Launch the server:

```sh
python server.py
```

4. Request an autointerp explanation for a feature with an API call shaped like Neuronpedia's explanation generate [endpoint](https://www.neuronpedia.org/api-doc#tag/explanations/POST/api/explanation/generate):

```sh
curl -X POST http://localhost:8000/api/explanation/generate \
  -H "Content-Type: application/json" \
  -d '{
    "index": "43",
    "n": 20,
    "explanationModelName": "google/gemini-2.5-flash-lite"
  }'
```

Notice that `modelId`  and `layer` are set by default to `gemma-3-27b-it` and `41-gemmascope-2-res-262k`. `n` is the number of top activation examples used to generate the label. `explanationModelName` is the OpenRouter model ID used to generate the explanation.

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


Sample `40` random features (each feature must have at least 15 non-zero activation examples) from the `gemma-3-27b-it/41-gemmascope-2-res-262k` search space:

```sh
python experiment/sample_features.py 40 --out-dir data/experiments/exp_1 --seed 0
```

### 2. Generate NLA explanations

Fetch NLA-derived examples for the sampled features (resumable) (only uses top 5 activation examples for each feature):

```sh
python experiment/fetch_nla.py --out-dir data/experiments/exp_1
```

### 3. Generate labels

Generate Neuronpedia labels for each sampled feature with three methods:
- `baez`
- `baez_last`
- [`eleuther_acts_top20`](https://www.neuronpedia.org/explanation-type/eleuther_acts_top20)

Note that the NLA explanations are only used for the `baez` and `baez_last` method.

```sh
python experiment/generate_labels.py --out-dir data/experiments/exp_1
```

### 4. Score labels

Score each (feature, label) with Delphi's `detection`, `fuzz`, and `embedding` scorers. Non-activating examples are sampled cross-feature from the same experiment's `raw/`.

```sh
python experiment/score_labels.py --out-dir data/experiments/exp_1 --seed 0
```

### 5. Analyze scores

Generate a CSV recap and plots (boxplot + mean bar chart) grouped by label generation method:

```sh
python experiment/analyze_scores.py --exp-dir data/experiments/exp_1
```

Outputs under `data/experiments/<name>/`:
- `scores_flat.csv` — one row per (feature, method)
- `scores_summary_by_method.csv` — mean/std/median/min/max/count per method
- `scores_boxplot.png`, `scores_mean_bar.png`

### 6. Quantitative label comparison

Embed each feature's `baez` and `delphi` labels and rank features by cosine distance between the two:

```sh
python experiment/analyze_quantitatively.py --exp-dir data/experiments/exp_1
```

Output: `label_distance__baez__vs__delphi.csv`, sorted by descending cosine distance.