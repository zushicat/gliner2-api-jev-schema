"""Jev endpoint smoke test against a running server.

Posts a request containing all three question types (choice, score, noul) to
`POST /v1/systemone` and asserts the Typesafe response contract:

- response envelope: {model, answers, usage}
- every `choice`/`score` answer has probabilities summing to ≈ 1
- every `choice` answer's `choice` is one of the options sent
- every `score` answer is within [0, levels − 1]
- every `noul` answer is in [0, 1]
- answers come back under the same question ids as the request

Usage (server must be running, e.g. `gliner-api`):
    .venv/bin/python scripts/jev_smoke_test.py [base_url]

Default base URL: http://127.0.0.1:11101 (HOST/PORT from .env).
Set SMOKE_API_KEY to test against a server with USE_API_KEY=true.
"""

import os
import sys

import requests

from gliner_api.settings import config

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else f"http://127.0.0.1:{config.PORT}"

REQUEST = {
    "state": (
        "My running shoes arrived in the wrong size and I need them "
        "before the marathon on Saturday."
    ),
    "model": "jev-latest",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "returns": "Exchanges, wrong or damaged items",
                "shipping": "Delivery status, delays, lost packages",
                "billing": "Charges, invoices, payment problems",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": [
                "Calm; no signs of frustration",
                "Annoyed; some irritation visible",
                "Angry; demanding immediate resolution",
            ],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this convey urgency?",
        },
    },
}

LEVELS = len(REQUEST["questions"]["frustration"]["criteria"])  # score ∈ [0, LEVELS-1]
OPTIONS = set(REQUEST["questions"]["department"]["criteria"])


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    headers = {}
    api_key = os.environ.get("SMOKE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"POST {BASE_URL}/v1/systemone ...")
    response = requests.post(
        f"{BASE_URL}/v1/systemone", json=REQUEST, headers=headers, timeout=120
    )
    response.raise_for_status()
    body = response.json()

    # Envelope
    check(set(body) >= {"model", "answers", "usage"}, f"envelope keys: {sorted(body)}")
    check(isinstance(body["model"], str) and body["model"], f"model: {body['model']!r}")
    usage = body["usage"]
    check(
        set(usage) == {"input_tokens", "output_tokens"}
        and isinstance(usage["input_tokens"], int)
        and isinstance(usage["output_tokens"], int),
        f"usage: {usage}",
    )
    check(usage["output_tokens"] == 0, "GLiNER2 generates no tokens")

    # Answers under the same question ids
    check(
        set(body["answers"]) == set(REQUEST["questions"]),
        f"answer ids {sorted(body['answers'])} != question ids",
    )

    # choice
    dep = body["answers"]["department"]
    check(dep["type"] == "choice", f"department type: {dep['type']}")
    check(dep["choice"] in OPTIONS, f"choice {dep['choice']!r} not an option")
    check(
        set(dep["probabilities"]) == OPTIONS,
        "probabilities keys must be the original option names",
    )
    check(
        abs(sum(dep["probabilities"].values()) - 1.0) < 1e-3,
        f"choice probabilities sum to {sum(dep['probabilities'].values())}",
    )
    check(0.0 <= dep["confidence"] <= 1.0, f"confidence out of range: {dep['confidence']}")

    # score
    fr = body["answers"]["frustration"]
    check(fr["type"] == "score", f"frustration type: {fr['type']}")
    check(0.0 <= fr["score"] <= LEVELS - 1, f"score out of range: {fr['score']}")
    check(
        set(fr["legend"]) == {str(i) for i in range(LEVELS)},
        f"legend keys: {sorted(fr['legend'])}",
    )
    check(
        fr["legend"] == {str(i): d for i, d in enumerate(REQUEST["questions"]["frustration"]["criteria"])},
        "legend must carry the original level descriptions",
    )
    check(
        abs(sum(fr["probabilities"].values()) - 1.0) < 1e-3,
        f"score probabilities sum to {sum(fr['probabilities'].values())}",
    )
    check(0.0 <= fr["confidence"] <= 1.0, f"confidence out of range: {fr['confidence']}")

    # noul
    ur = body["answers"]["is_urgent"]
    check(ur["type"] == "noul", f"is_urgent type: {ur['type']}")
    check("confidence" not in ur, "noul answers carry no confidence field")
    check(0.0 <= ur["noul"] <= 1.0, f"noul out of range: {ur['noul']}")

    print("model:  ", body["model"])
    print("usage:  ", usage)
    print("answers:", {k: (v.get("choice") or v.get("score") or v.get("noul")) for k, v in body["answers"].items()})
    print("\nOK — smoke test passed")


if __name__ == "__main__":
    main()
