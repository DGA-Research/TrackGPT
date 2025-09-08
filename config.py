import os
from dotenv import load_dotenv
import sys
import streamlit as st

load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

def _get_secret_or_env(key: str, default: str = "") -> str:
    """Prefer Streamlit secrets, fall back to environment, else default."""
    try:
        val = st.secrets[key]
        if val is None:
            return default
        return str(val)
    except Exception:
        return os.getenv(key, default)

def _get_bool_secret_or_env(key: str, default: bool = False) -> bool:
    raw = _get_secret_or_env(key, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}

class Config:
    """
    Central configuration class for the application.

    Sensitive credentials should live in Streamlit secrets for prod;
    .env/env vars are fine for local dev.
    """

    # --- Essential (unchanged) ---
    OPENAI_API_KEY: str = st.secrets["OPENAI_API_KEY"]
    ASSEMBLYAI_API_KEY: str = st.secrets["ASSEMBLYAI_API_KEY"]

    # --- Models (unchanged defaults) ---
    WHISPER_MODEL: str = _get_secret_or_env("WHISPER_MODEL", "whisper-1")
    ANALYSIS_MODEL: str = _get_secret_or_env("ANALYSIS_MODEL", "gpt-4.1-mini")

    # --- Processing (unchanged default) ---
    AUDIO_FORMAT: str = _get_secret_or_env("AUDIO_FORMAT", "mp3")

    # --- Output (unchanged default) ---
    DEFAULT_OUTPUT_DIR: str = _get_secret_or_env("OUTPUT_DIR", "output")

    # --- NEW: yt-dlp hardening (all optional) ---
    YTDLP_COOKIES_FILE: str = os.getenv("YTDLP_COOKIES_FILE", "").strip()
    YTDLP_COOKIES_FROM_BROWSER: str = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()
    YTDLP_USER_AGENT: str = os.getenv("YTDLP_USER_AGENT", "").strip()
    YTDLP_GEO_BYPASS: bool = os.getenv("YTDLP_GEO_BYPASS", "true").lower() in {"1","true","yes"}
    YTDLP_GEO_COUNTRY: str = os.getenv("YTDLP_GEO_COUNTRY", "US").strip()
    YTDLP_RETRIES: int = int(os.getenv("YTDLP_RETRIES", "3"))

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ConfigError(
                "ERROR: OPENAI_API_KEY is not set. "
                "Please add it to Streamlit secrets or your environment."
            )
        # ASSEMBLYAI_API_KEY is required by your current code; keep as-is:
        if not cls.ASSEMBLYAI_API_KEY:
            raise ConfigError(
                "ERROR: ASSEMBLYAI_API_KEY is not set. "
                "Please add it to Streamlit secrets or your environment."
            )
        print("Configuration validated.")

# Validate configuration immediately upon import
try:
    Config.validate()
except ConfigError as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)

