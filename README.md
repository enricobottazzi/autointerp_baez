# autointerp_baez

`autointerp_baez` is a autointerp method that generates a natural-language label for a SAE feature using [NLA](https://transformer-circuits.pub/2026/nla/index.html#nla-training) explanations.

Let's take a step back. Autointerp methods are used to generate natural-language labels for [SAE features](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:1). Current [autointerp methods](https://blog.eleuther.ai/autointerp/) work by feeding the top-activating examples for a given SAE feature to an LLM and asking them to find a common thread across these examples in natural-language, the label.

```text
top-activating examples -> [ Autointerp ] -> label
```

The method proposed here is similar but uses NLA explanations instead of top-activating examples.

```text
top-activating examples -> [ NLA ] -> NLA explanations -> [ autointerp_baez ] -> label
```

## `autointerp_baez` methodology specs

Requirement: the SAE feature for which the label is to be generated must belong to `gemma-3-27b-it/41-gemmascope-2-res-262k` as this is the only model and layer for which an NLA has been trained.

### NLA

For the queried SAE feature, execute the following steps:

1. Fetch the top 20 activation examples from Neuronpedia.
2. Truncate each example up to the token that maximizes the activation.
3. For each example, feed it to the [NLA API](https://www.neuronpedia.org/api-doc#tag/nla/POST/api/nla/explain) and obtain the NLA explanation associated with the last token

### `autointerp_baez`

`autointerp_baez` takes 20 NLA explanations, each paired with an activation score normalized to an integer `0-10`.

It sends one chat-completion request at temperature `0.7`:

- `system`: instruct the model to find the shared language pattern and return one concise label.
- `user`: list the 20 NLA explanations as examples with their activation scores.

The final model response must end with:

```text
[EXPLANATION]: <label>
```

Only the text after `[EXPLANATION]:` is returned.

## How to use

This repo exposes a minimal autointerp service for generating a natural-language label for a Neuronpedia feature based on NLA explanations.

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
    "explanationModelName": "openai/gpt-4o-mini"
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
