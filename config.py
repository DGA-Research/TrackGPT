import os
import sys
from typing import Optional

from dotenv import load_dotenv

try:
    import streamlit as st
except ImportError:  # Allow non-Streamlit contexts (e.g., CLI usage)
    st = None  # type: ignore

load_dotenv()


class ConfigError(Exception):
    """Custom exception for configuration errors."""


def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Fetch a credential from Streamlit secrets when available, else environment variables.

    Accessing st.secrets[...] raises StreamlitSecretNotFoundError when missing; guard accordingly so
    CLI usage (without Streamlit) still works.
    """
    # Streamlit present and secrets populated
    if st is not None and getattr(st, "secrets", None) is not None:
        try:
            value = st.secrets[name]
            if isinstance(value, str) and value.strip():
                return value
        except Exception:
            pass

    value = os.getenv(name, default)
    if isinstance(value, str) and value.strip():
        return value
    return default


class Config:
    """
    Central configuration class for the application.

    Credentials are sourced from Streamlit secrets (when running via UI) or environment variables
    / .env files for CLI and local workflows.
    """

    # --- Essential ---
    OPENAI_API_KEY: Optional[str] = _get_secret("OPENAI_API_KEY")
    ASSEMBLYAI_API_KEY: Optional[str] = _get_secret("ASSEMBLYAI_API_KEY")

    # --- Models ---
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-1")
    ANALYSIS_MODEL: str = os.getenv("ANALYSIS_MODEL", "gpt-4.1-mini")

    # --- Processing ---
    AUDIO_FORMAT: str = os.getenv("AUDIO_FORMAT", "mp3")

    # --- Output ---
    DEFAULT_OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ConfigError(
                "ERROR: Missing OPENAI_API_KEY. Set it in .env or Streamlit secrets."
            )
        if not cls.ASSEMBLYAI_API_KEY:
            raise ConfigError(
                "ERROR: Missing ASSEMBLYAI_API_KEY. Set it in .env or Streamlit secrets."
            )


# Validate configuration immediately upon import
try:
    Config.validate()
except ConfigError as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(1)
