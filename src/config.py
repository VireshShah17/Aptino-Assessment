import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv


_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """
        Configure root logging once for the whole application.
        Safe to call multiple times - only the first call has any effect,
        so every entry-point script can call it unconditionally.
    """
    global _CONFIGURED

    if _CONFIGURED:
        return

    logging.basicConfig(level = level, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _CONFIGURED = True


@dataclass(frozen = True)
class Settings:
    """
        Typed, validated application settings.
        Centralizing these means the model name / temperature / retry policy
        live in exactly one place instead of being copy-pasted into every
        agent's __init__.
    """
    google_api_key: str
    model_name: str = "gemini-3.1-pro-preview"
    temperature: float = 0.0
    max_retries: int = 2
    request_timeout: int = 60


def load_settings() -> Settings:
    """
        Load environment variables once and return validated Settings.
        Raises: ValueError: if a required environment variable is missing.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing GOOGLE_API_KEY environment variable. "
            "Set it in your .env file or environment before starting the app."
        )

    return Settings(
        google_api_key = api_key,
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-pro-preview"),
        temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
        max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "2")),
        request_timeout = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60")),
    )
