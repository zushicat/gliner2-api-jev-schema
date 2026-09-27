# ********
# Startup silencers (interop noise from gliner2/transformers/torch; no logic impact)
# see also: https://github.com/fastino-ai/GLiNER2/pull/175
# ********
import warnings

# Warning 1: transformers doesn't know model_type "extractor" (GLiNER2 checkpoint)
# when AutoTokenizer resolves the config — pure log noise, tokenizer loads fine.
# NOTE: transformers 5.x moved this API out of the top-level namespace.
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

# Warning 2: transformers' deberta_v2 module still uses @torch.jit.script at
# import time; torch 2.14 deprecates it. Message-based filter, because the
# warning is attributed to torch/jit/_script.py, not to transformers.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.jit\.script` is deprecated.*",
)

# Warning 3: gliner2 retries the encoder with eager attention after DebertaV2
# rejects sdpa (the model runs eager either way). Message-based filter, because
# gliner2 warns with stacklevel=2, attributing it to gliner2/models/span/model.py.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"Encoder rejected attn_implementation.*",
)
# end silencers

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from gliner_api.auth import bearer_auth_dependency
from gliner_api.routes import root_route
from gliner_api.routes.systemone_route import router as systemone_router
from gliner_api.services.extractor import ExtractorService
from gliner_api.settings import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup; failure to load fails the boot.
    service = ExtractorService(config)
    service.load()
    app.state.extractor = service
    yield


app = FastAPI(dependencies=[Depends(bearer_auth_dependency)], lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
app.include_router(root_route.router)
app.include_router(systemone_router)


# add exception handling
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=404,
        content={"Status": "Error", "Response": "Endpoint not existing"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_msg = str(exc.errors()[0]["msg"])
    return JSONResponse(
        status_code=400,
        content={"Status": "Error", "Response": f"Value error, {error_msg}"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=400, content={"Status": "Error", "Response": str(exc)}
    )
