"""`POST /v1/systemone` — Typesafe Jev (System One) compatible endpoint.

Accepts and returns the Jev schema so clients can switch from the Typesafe
cloud API by only changing the base URL. The response intentionally uses the
Typesafe shape (`{model, answers, usage}`), not the app's legacy
`{"Status": …, "Response": …}` wrapper.
"""

from fastapi import APIRouter, Depends, Request

from gliner_api.models.jev_models import SystemOneRequest, SystemOneResponse, Usage
from gliner_api.services.extractor import ExtractorService
from gliner_api.services.jev_translator import (
    questions_to_schema,
    result_to_answers,
    state_to_text,
)
from gliner_api.settings import config

router = APIRouter()


def get_service(request: Request) -> ExtractorService:
    return request.app.state.extractor


@router.post("/v1/systemone", response_model=SystemOneResponse)
def system_one(
    payload: SystemOneRequest,
    service: ExtractorService = Depends(get_service),
):
    text = state_to_text(payload.state)
    schema, restore = questions_to_schema(payload.questions)
    result = service.classify(text, schema)
    answers = result_to_answers(payload.questions, result, restore)
    return SystemOneResponse(
        model=payload.model or config.GLIDER_MODEL_NAME,
        answers=answers,
        usage=Usage(
            input_tokens=service.count_tokens(text),
            output_tokens=0,  # GLiNER2 generates no tokens — semantically correct
        ),
    )
