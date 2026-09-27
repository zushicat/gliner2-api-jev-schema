"""Pydantic models for the Typesafe Jev (System One) contract.

`POST /v1/systemone` accepts and returns this schema so clients can switch
from the Typesafe cloud API by only changing the base URL.

- Discriminated unions on `type` for questions and answers.
- `str | dict | list` accepted wherever Typesafe allows structured strings;
  the translator flattens dicts/lists before they reach GLiNER2.
- Extra/unknown fields in the request body are ignored (Pydantic v2 default)
  for drop-in tolerance.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Structured guidance: plain string, object, or array (flattened by the translator).
Instructions = str | dict | list
Structured = str | dict | list | None

# *************
# questions
# *************


class NoulCriteria(BaseModel):
    """Optional `{ "true": …, "false": … }` criteria for a `noul` question."""

    true_: Structured = Field(default=None, alias="true")
    false_: Structured = Field(default=None, alias="false")


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Instructions
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Instructions
    criteria: dict[str, Structured]  # option → description (description may be None)

    @model_validator(mode="after")
    def _at_least_one_option(self):
        if not self.criteria:
            raise ValueError("choice question needs at least one option in 'criteria'")
        return self


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Instructions
    criteria: list[Structured]  # ordered level descriptions; level number = index

    @field_validator("criteria")
    @classmethod
    def _at_least_two_levels(cls, v):
        if len(v) < 2:
            raise ValueError("score question needs at least two levels in 'criteria'")
        return v


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion,
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    state: str | dict | list
    model: str | None = None  # accepted and ignored (echoed back by the route)
    questions: dict[str, Question]

    @model_validator(mode="after")
    def _non_empty_questions(self):
        if not self.questions:
            raise ValueError("'questions' must contain at least one question")
        return self


# *************
# answers
# *************


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float  # probability that the answer is yes (no confidence field)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]  # sums to 1


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float  # probability-weighted mean of level numbers, can fall between levels
    legend: dict[str, str]  # level-number-string → original level description
    confidence: float
    probabilities: dict[str, float]  # level-number-string → float, sums to 1


Answer = Annotated[
    ChoiceAnswer | ScoreAnswer | NoulAnswer,
    Field(discriminator="type"),
]


# *************
# response envelope
# *************


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
