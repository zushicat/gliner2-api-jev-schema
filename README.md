# Gliner2 Api with Jev compatible endpoints
Local classification API
- using [GLiNER2](https://github.com/fastino-ai/GLiNER2)
- backed by [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) model
- serving a Typesafe Ai's Jev (System One) compatible endpoint at `POST /v1/systemone`

## Installation

- Download model [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (Huggingface link)
- Copy `.env.example` to `.env` and set `GLINER_MODEL_PATH` to the downloaded model
- Create a virtual environment and install the app (editable install; no
  `PYTHONPATH` / env exports needed — `.env` is read automatically):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # then edit GLINER_MODEL_PATH (+ optional API_KEY / USE_API_KEY)
gliner-api             # serves HOST:PORT from .env → default 0.0.0.0:11101
```

Dev alternative (auto-reload):

```bash
uvicorn gliner_api.app:app --reload --host 0.0.0.0 --port 11101
```

In the terminal, you should see something that looks like this:
```
INFO:     Started server process [20748]
INFO:     Waiting for application startup.
============================================================
🧠 Model Configuration
============================================================
Encoder model      : microsoft/deberta-v3-large
Counting layer     : count_lstm
Token pooling      : first
============================================================
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:11101 (Press CTRL+C to quit)
```

**Note**: The first startup takes a few seconds to load the model. 

## Jev (System One) compatibility

Clients built for Typesafe's System One API can switch to this local API by
only changing the base URL (`https://api.typesafe.ai` → `http://localhost:11101`).
The model is loaded once at startup and reused for every request.

### Endpoint

`POST /v1/systemone` with `Authorization: Bearer <key>` when `USE_API_KEY=true`
in `.env` (Typesafe-style bearer auth).

Request (all three question types in one call — all questions are scored in a
single encoder pass):

```json
{
  "state": "My dog has allergies. I bought your pet food. Is it save if I feed my pet with your food right now? I am worried that this might make my dog sick.",
  "model": "jev-latest",
  "questions": {
    "pet": {
      "type": "choice",
      "instructions": "Which type of pet does the customer have?",
      "criteria": {
        "cat": "The customer seems to have a cat.",
        "dog": "The customer seems to have a dog.",
        "other": "The customer seems to have neither a cat nor a dog."
      }
    },
    "worried": {
      "type": "score",
      "instructions": "How worried is the customer?",
      "criteria": [
        "The customer seems to be quite worried.",
        "The customer seems to be not worried at all."
      ]
    },
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?"
    }
  }
}
```

Response (question ids come back unchanged):

```json
{
  "model": "jev-latest",
  "answers": {
    "pet": {
      "type": "choice",
      "choice": "dog",
      "confidence": 0.6622196714774753,
      "probabilities": {
        "cat": 0.13373973209206605,
        "dog": 0.6622196714774753,
        "other": 0.20404059643045863
      }
    },
    "worried": {
      "type": "score",
      "score": 0.2835056460811368,
      "legend": {
        "0": "The customer seems to be quite worried.",
        "1": "The customer seems to be not worried at all."
      },
      "confidence": 0.7164943539188633,
      "probabilities": {
        "0": 0.7164943539188633,
        "1": 0.2835056460811368
      }
    },
    "is_urgent": {
      "type": "noul",
      "noul": 0.5534040836899954
    }
  },
  "usage": {
    "input_tokens": 39,
    "output_tokens": 0
  }
}
```

#### Note
If you like to see the probabilities from the actual classification results, then dump the result either in [routes/systemone_route.py](/Users/karin/_programming/gliner-api/src/gliner_api/routes/systemone_route.py) or in [services/jev_translator.py](/Users/karin/_programming/gliner-api/src/gliner_api/services/jev_translator.py) with a simple ```print(result.__dict__)```.

The dump of above example:
```
{'text': 'My dog has allergies. I bought your pet food. Is it save if I feed my pet with your food right now? I am worried that this might make my dog sick.', 'tasks': mappingproxy({'pet': TaskResult(task='pet', labels=('dog',), probabilities=mappingproxy({'cat': 0.13373973209206605, 'dog': 0.6622196714774753, 'other': 0.20404059643045863}), utilities=mappingproxy({'cat': -1.4209377765655518, 'dog': 0.17876394093036652, 'other': -0.9985144138336182}), confidence=0.6622196714774753, exclusive=True, ordered=False, level=None), 'worried': TaskResult(task='worried', labels=('0',), probabilities=mappingproxy({'0': 0.7164943539188633, '1': 0.2835056460811368}), utilities=mappingproxy({'0': 0.4381864368915558, '1': -0.4889518916606903}), confidence=0.7164943539188633, exclusive=True, ordered=True, level=0), 'is_urgent': TaskResult(task='is_urgent', labels=('yes',), probabilities=mappingproxy({'yes': 0.5534040836899954, 'no': 0.4465959163100046}), utilities=mappingproxy({'yes': 0.13758087158203125, 'no': -0.07685337960720062}), confidence=0.5534040836899954, exclusive=True, ordered=False, level=None)}), 'feasible': True, 'violations': (), 'objective': 0.7545312494039536, 'decoder': 'independent', 'exact': True, 'include_confidence': True}
``` 


### Mapping to GLiNER2

| Jev question | GLiNER2 build | Jev answer |
|---|---|---|
| `choice` | exclusive task, options as labels with descriptions | `choice` + `probabilities` (sum to 1) + `confidence` = P(selected) |
| `score` | ordinal task, labels `"0"…"N−1"` with level descriptions | `score` = Σ level × P(level) (can fall between levels) + `legend` + `probabilities` + `confidence` = P(argmax level) |
| `noul` | exclusive `yes`/`no` task | `noul` = P("yes") (no `confidence` field) |

`instructions` and criteria entries may be plain strings, objects, or arrays —
structured values are flattened to indented JSON before reaching the model, and
the response always carries the original strings the client sent. GLiNER2
reserves the markers `[P] [L] [C] [E] [R] [DESCRIPTION] [EXAMPLE] [OUTPUT]` and
the characters `(` `)`, so user strings are sanitized for the model while the
response restores them verbatim.

### Differences from Typesafe

- **Confidence semantics**: Typesafe derives `confidence` from how *peaked*
  the distribution is; this API ships P(selected label) — a monotone-ish proxy
  in [0, 1]. Swapping in a peakedness formula later would not change the
  contract.
- **`model` echo**: the request's `model` field is accepted and ignored; the
  response reports the configured `GLINER_MODEL_NAME` (default
  `GLiNER2.5-Decide`).
- **`usage`**: best-effort — `input_tokens` counted with the model's own
  tokenizer, `output_tokens` is always 0 (GLiNER2 generates no tokens).
- **Error bodies**: validation errors use the app's legacy
  `400 {"Status": "Error", "Response": …}` shape instead of Typesafe's `422`.
- **No rate limits**, no `429`s.
- **Size caps**: Typesafe's 255-option / 10-level caps are not enforced
  (GLiNER2 handles more; Typesafe-built clients never exceed them anyway).

## Smoke tests

```bash
gliner-api &                                   # or run in another terminal
.venv/bin/python scripts/jev_smoke_test.py     # asserts the response contract
.venv/bin/python scripts/gliner_test.py        # raw AutoExtractor example
```
