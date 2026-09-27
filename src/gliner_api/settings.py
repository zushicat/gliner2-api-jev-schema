"""Project-wide settings, loaded from `.env` at the project root.

Real environment variables still win over `.env` (pydantic-settings
precedence), so CI/production overrides keep working. Anchoring `env_file`
to the project root (not the cwd) makes the server startable from anywhere.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"  # project root, cwd-independent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_KEY: str | None = None
    USE_API_KEY: bool = False
    GLIDER_MODEL_PATH: str
    GLIDER_MODEL_NAME: str = "GLiNER2.5-Decide"  # echoed in /v1/systemone responses
    HOST: str = "0.0.0.0"
    PORT: int = 11101


config = Settings()
