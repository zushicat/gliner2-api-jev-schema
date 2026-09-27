# More examples: https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/1-classification.md
# Run from the project root: .venv/bin/python scripts/gliner_test.py

from gliner2 import AutoExtractor

from gliner_api.settings import config


import warnings
from transformers.utils import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

print("start loading model for extractor")
extractor = AutoExtractor.from_pretrained(
    config.GLIDER_MODEL_PATH, local_files_only=True
)
print("extractor loaded")

schema = extractor.create_schema().classification(
    "customer_pet",
    {
        "cat": "The customer seems to have a cat as a pet",
        "dog": "The customer seems to have a dog as a pet",
        "other": "The customer has neither a cat or a dog but any other pet.",
    },
)


def make_prediction(text: str) -> any:
    result = extractor.extract(text, schema)
    print(result)


texts = [
    "My cat has allergies. Can I use your pet food products?",
    "My bird has allergies. Can I use your pet food products?",
    "My dog likes your pet food.",
]

print("start predictions")
for t in texts:
    make_prediction(t)
