"""Model loading and inference (load once at startup, serve many requests).

The GLiNER2 model is loaded exactly once during FastAPI's lifespan startup;
failure to load fails the boot (better than 500s per request). Inference is
serialized with a single lock for predictable latency on CPU; the endpoint
itself runs in FastAPI's threadpool so the event loop is never blocked.

The `Classifier`'s internal schema-compile LRU cache (keyed by schema
fingerprint) means repeated identical question sets skip recompilation.
"""

import threading

from gliner2 import AutoExtractor
from gliner2.classification import (
    ClassificationResult,
    Classifier,
    ClassificationSchema,
)


class ExtractorService:
    def __init__(self, config):
        self._config = config
        self._classifier: Classifier | None = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def load(self) -> None:
        model = AutoExtractor.from_pretrained(
            self._config.GLINER_MODEL_PATH,
            local_files_only=True,
        )
        model.eval()
        # Composes around the already-loaded model — no second model load.
        self._classifier = Classifier(model=model)
        # Tokenizer for the usage estimate (best-effort input token count).
        self._tokenizer = getattr(model, "tokenizer", None) or model.processor.tokenizer

    @property
    def loaded(self) -> bool:
        return self._classifier is not None

    def classify(self, text: str, schema: ClassificationSchema) -> ClassificationResult:
        with self._lock:  # serialize inference
            return self._classifier.classify(text, schema)

    def count_tokens(self, text: str) -> int:
        """Best-effort input token count using the model's own tokenizer."""
        return len(self._tokenizer(text, add_special_tokens=True)["input_ids"])
