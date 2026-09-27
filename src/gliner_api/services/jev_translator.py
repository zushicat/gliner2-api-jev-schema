"""Jev (System One) ↔ GLiNER2 translation.

Three pure functions (plan §3b):

- `state_to_text(state)` — flatten the Jev `state` to model input text.
- `questions_to_schema(questions)` — build ONE ClassificationSchema with one
  task per question (all questions scored in a single encoder pass), plus a
  `restore` record mapping sanitized GLiNER2 strings back to the originals so
  the Jev answer can be rebuilt verbatim.
- `result_to_answers(questions, result, restore)` — rebuild Jev answers from
  the GLiNER2 `ClassificationResult` (full distributions).

Mapping (plan §3.2):

| Jev       | GLiNER2 build                                        | Jev answer                                                     |
|-----------|------------------------------------------------------|----------------------------------------------------------------|
| `choice`  | `single(task, {option: description}, instruction=…)` | `choice`+`probabilities` (keys restored)+`confidence`=P(selected) |
| `score`   | `ordinal(task, {"0": lvl0, …}, instruction=…)`       | `score`=Σ i·P(i), `legend`, `probabilities`, `confidence`=P(argmax) |
| `noul`    | `single(task, {"yes": …, "no": …}, instruction=…)`   | `noul`=P("yes")                                                |

GLiNER2 raises `SchemaError` if any task name / label name / description /
instruction contains a reserved marker (`[P] [L] [C] [E] [R] [DESCRIPTION]
[EXAMPLE] [OUTPUT]`) or a paren. Jev input is free-form, so every string is
sanitized before it enters the schema; the response carries the original
strings via the restore record.
"""

import json
from typing import Any

from gliner2.classification import ClassificationSchema

from gliner_api.models.jev_models import (
    ChoiceAnswer,
    NoulAnswer,
    Question,
    ScoreAnswer,
)

_RESERVED = (
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "[P]",
    "[L]",
    "[C]",
    "[E]",
    "[R]",
)


def sanitize(text: str) -> str:
    """Remove GLiNER2-reserved markers and parens (longest tokens first)."""
    for token in _RESERVED:
        text = text.replace(token, " ")
    return text.replace("(", " ").replace(")", " ").strip()


def flatten(value: Any) -> str | None:
    """Flatten structured guidance to a sanitized plain string.

    None / empty → None (allowed by GLiNER2); dict/list → indented JSON
    (backticks survive JSON, so references like `ticket.messages[0].text`
    still read well); plain strings are sanitized in place.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    text = sanitize(text)
    return text or None


def state_to_text(state: Any) -> str:
    """Jev `state` → model input text (plain strings pass through)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def questions_to_schema(
    questions: dict[str, Question],
) -> tuple[ClassificationSchema, dict]:
    """Build one schema with one exclusive task per question.

    Returns `(schema, restore)` where `restore[qid]` holds everything needed to
    rebuild the Jev answer with the client's original strings:

    - choice: `{"task", "type", "original": {sanitized option → original option}}`
    - score:  `{"task", "type", "levels": [(level-number-string, original description), …]}`
    - noul:   `{"task", "type"}`
    """
    schema = ClassificationSchema()
    restore: dict[str, dict] = {}
    used_task_names: dict[str, str] = {}  # sanitized task name → original question id

    for qid, q in questions.items():
        task = sanitize(qid)
        if not task:
            raise ValueError(f"question id {qid!r} is empty after sanitization")
        if task in used_task_names:
            raise ValueError(
                f"question ids {used_task_names[task]!r} and {qid!r} collide "
                f"after sanitization"
            )
        used_task_names[task] = qid

        if q.type == "choice":
            labels: dict[str, str | None] = {}
            original: dict[str, str] = {}
            for option, description in q.criteria.items():
                label = sanitize(option)
                if not label:
                    raise ValueError(
                        f"question {qid!r}: option {option!r} is empty after sanitization"
                    )
                if label in labels:
                    raise ValueError(
                        f"question {qid!r}: options {original[label]!r} and {option!r} "
                        f"collide after sanitization"
                    )
                labels[label] = flatten(description)
                original[label] = option
            schema.single(task, labels, instruction=flatten(q.instructions))
            restore[qid] = {"task": task, "type": "choice", "original": original}

        elif q.type == "score":
            labels = {}
            levels: list[tuple[str, Any]] = []
            for i, description in enumerate(q.criteria):
                labels[str(i)] = flatten(description)
                levels.append((str(i), description))
            schema.ordinal(task, labels, instruction=flatten(q.instructions))
            restore[qid] = {"task": task, "type": "score", "levels": levels}

        else:  # noul
            criteria = q.criteria
            schema.single(
                task,
                {
                    "yes": flatten(criteria.true_) if criteria else None,
                    "no": flatten(criteria.false_) if criteria else None,
                },
                instruction=flatten(q.instructions),
            )
            restore[qid] = {"task": task, "type": "noul"}

    return schema, restore


def result_to_answers(
    questions: dict[str, Question], result, restore: dict
) -> dict[str, ChoiceAnswer | ScoreAnswer | NoulAnswer]:
    """Rebuild Jev answers from the GLiNER2 result using original strings."""
    answers = {}
    for qid, q in questions.items():
        r = restore[qid]
        task = r["task"]

        # MappingProxyType → plain dict (translator owns its copy).
        probabilities = dict(result.probabilities(task))
        if not probabilities:
            raise ValueError(f"question {qid!r}: the model returned no distribution")

        if r["type"] == "choice":
            original = r["original"]
            selected = result.value(task)
            if selected is None or selected not in original:
                raise ValueError(
                    f"question {qid!r}: model selected unknown label {selected!r}"
                )
            answers[qid] = ChoiceAnswer(
                type="choice",
                choice=original[selected],
                confidence=result.confidence(task),  # = P(selected label)
                probabilities={original[k]: v for k, v in probabilities.items()},
            )

        elif r["type"] == "score":
            level_probs = {int(k): v for k, v in probabilities.items()}
            answers[qid] = ScoreAnswer(
                type="score",
                score=sum(i * p for i, p in level_probs.items()),
                legend={level: desc for level, desc in r["levels"]},
                confidence=max(level_probs.values()),  # P(argmax level)
                probabilities={str(i): p for i, p in sorted(level_probs.items())},
            )

        else:  # noul
            answers[qid] = NoulAnswer(type="noul", noul=probabilities["yes"])

    return answers
