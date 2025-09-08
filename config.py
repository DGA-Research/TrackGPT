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
    # Provide either a cookies file or a browser profile (prefer file).
    YTDLP_COOKIES_FILE: str = _get_secret_or_env("YTDLP_COOKIES_FILE", "").strip()      # e.g., "C:/path/cookies.txt" or "/app/cookies.txt"
    YTDLP_COOKIES_FROM_BROWSER: str = _get_secret_or_env("YTDLP_COOKIES_FROM_BROWSER", "").strip()  # e.g., "chrome:Default", "edge", "firefox:default-release"

    # Some sites behave better with an explicit UA.
    YTDLP_USER_AGENT: str = _get_secret_or_env("YTDLP_USER_AGENT", "").strip()

    # Try different YouTube client to bypass throttling/403s.
    # Common choice: "youtube:player_client=android"
    YTDLP_EXTRACTOR_ARGS: str = _get_secret_or_env("YTDLP_EXTRACTOR_ARGS", "").strip()

    # Region/consent issues:
    YTDLP_GEO_BYPASS: bool = _get_bool_secret_or_env("YTDLP_GEO_BYPASS", True)
    YTDLP_GEO_COUNTRY: str = _get_secret_or_env("YTDLP_GEO_COUNTRY", "US").strip()

    # How many fallback attempts to try (primary + N extra)
    YTDLP_RETRIES: int = int(_get_secret_or_env("YTDLP_RETRIES", "2"))

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
