import os
from dotenv import load_dotenv
import sys # Import sys for sys.stderr and sys.exit
import streamlit as st

load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Central configuration class for the application.
    
    All sensitive credentials are loaded from environment variables.
    Required variables must be set in a .env file or environment.
    
    Usage:
    1. Create a .env file with required variables
    2. Access config via Config.CONSTANT_NAME
    3. Call Config.validate() to check required settings
    
    Security Note:
    - Never commit .env files to version control
    - Use environment variables in production
    """

    # --- Essential ---
    OPENAI_API_KEY: str = st.secrets["OPENAI_API_KEY"]  # Required OpenAI API key
    ASSEMBLYAI_API_KEY: str = st.secrets["ASSEMBLYAI_API_KEY"] # AssemblyAI API key

    # --- Models ---
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-1")  # Default transcription model
    ANALYSIS_MODEL: str = os.getenv("ANALYSIS_MODEL", "gpt-4.1-mini")  # Default analysis model

    # --- Processing ---
    AUDIO_FORMAT: str = os.getenv("AUDIO_FORMAT", "mp3")  # Default audio format for downloads

    # --- Output ---
    DEFAULT_OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")  # Directory for generated files

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ConfigError(
                "ERROR: OPENAI_API_KEY is not set. "
                "Please create a .env file and add your OpenAI API key."
            )
        print("Configuration validated.") # Add confirmation

# Validate configuration immediately upon import
try:
    Config.validate()
except ConfigError as e:
    print(str(e), file=sys.stderr)
    sys.exit(1) # Exit if config is invalid
