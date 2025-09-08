"""
Module for downloading audio from URLs using yt-dlp.

Handles:
- Finding yt-dlp and ffmpeg executables
- Downloading audio in specified format
- Extracting standardized metadata
- Error handling and fallback behavior
"""
import subprocess
import logging
import sys
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from config import Config

logging.info(f"yt-dlp cfg: extractor_args={getattr(Config, 'YTDLP_EXTRACTOR_ARGS', '')!r} cookies_file={getattr(Config, 'YTDLP_COOKIES_FILE', '')!r} ua_set={bool(getattr(Config, 'YTDLP_USER_AGENT', ''))}")


# --- Dependency Checks ---
try:
    import yt_dlp
except ImportError:
    print("ERROR: 'yt-dlp' library not found. Install using: pip install yt-dlp", file=sys.stderr)
    sys.exit(1)

def find_yt_dlp_executable() -> Optional[str]:
    """
    Locates the yt-dlp executable on the system.

    It first attempts to use `yt_dlp.utils.exe_path()` if available (for bundled
    executables), and falls back to searching the system's PATH using `shutil.which()`.

    Returns:
        The full path to the yt-dlp executable if found, otherwise None.
    """
    try:
        # Attempt to find executable using yt-dlp's internal helper
        return yt_dlp.utils.exe_path()
    except AttributeError:
        # Fallback to searching system PATH if internal helper is not available
        import shutil
        return shutil.which("yt-dlp")

def find_ffmpeg_executable() -> Optional[str]:
    """
    Locates the ffmpeg executable on the system by searching the system's PATH.

    Returns:
        The full path to the ffmpeg executable if found, otherwise None.
    """
    import shutil
    return shutil.which("ffmpeg")

YT_DLP_PATH = find_yt_dlp_executable()
if not YT_DLP_PATH:
    print("ERROR: 'yt-dlp' command not found in system PATH or via library helper.", file=sys.stderr)
    print("Please ensure yt-dlp is installed and accessible.", file=sys.stderr)

FFMPEG_PATH = find_ffmpeg_executable()
if not FFMPEG_PATH:
    print("ERROR: 'ffmpeg' command not found in system PATH.", file=sys.stderr)
    print("Please ensure ffmpeg is installed and accessible.", file=sys.stderr)
    sys.exit(1) # Exit if ffmpeg is not found

# --- Core Function ---
def download_audio(url: str, output_dir: Path, base_filename: str, type_input) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Downloads audio from a given URL using the yt-dlp command-line tool.

    This function first attempts to extract video metadata using the yt-dlp
    library and then executes the yt-dlp CLI to download and convert the
    audio to the format specified in the configuration. It handles potential
    errors during both metadata extraction and the download process.

    Args:
        url: The URL of the video or audio source (e.g., YouTube, Vimeo).
        output_dir: The directory where the downloaded audio file should be saved.
                    The directory will be created if it does not exist.
        base_filename: The base name for the output audio file (without the file extension).

    Returns:
        A tuple containing the full path to the downloaded audio file (as a string)
        and a dictionary containing standardized metadata if the download is
        successful. Returns None if the download or metadata extraction fails
        after handling errors.
    """
    # Check if yt-dlp executable was found during initial checks
    if not YT_DLP_PATH:
         logging.error("yt-dlp executable not found. Cannot download.")
         return None

    # Define output paths using the base filename and configured audio format
    output_path_template = output_dir / f"{base_filename}.%(ext)s"
    final_audio_path = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"

    # Ensure the output directory exists, creating it if necessary
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {output_dir}: {e}")
        return None

    # --- Metadata Extraction ---
    # Extract metadata using the yt-dlp library without downloading the video.
    # This allows us to get information even if the download later fails.
    ydl_opts = {
        'quiet': True,          # Suppress console output from yt-dlp library
        'no_warnings': True,    # Hide warnings from yt-dlp library
        'extract_flat': False,  # Ensure full metadata is extracted
    }
        # Optional: pass cookies/user-agent to metadata extraction, if configured
    try:
        if Config.YTDLP_COOKIES_FILE:
            ydl_opts['cookiefile'] = Config.YTDLP_COOKIES_FILE
        elif Config.YTDLP_COOKIES_FROM_BROWSER:
            # Note: library supports this as 'cookiesfrombrowser'
            ydl_opts['cookiesfrombrowser'] = Config.YTDLP_COOKIES_FROM_BROWSER
        if Config.YTDLP_USER_AGENT:
            ydl_opts['user_agent'] = Config.YTDLP_USER_AGENT
    except AttributeError:
        # If those config fields don't exist (older Config), just ignore
        pass
    # Metadata extraction strategy:
    # - Attempt to extract comprehensive metadata first.
    # - If extraction fails (e.g., due to geo-restrictions, private video),
    #   fall back to a minimal metadata dictionary.
    # - Always include the original URL as a fallback for webpage_url.
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            # Standardize metadata keys for consistent access
            metadata = {
                'title': info_dict.get('title', 'Unknown Title'),
                'uploader': info_dict.get('uploader') or info_dict.get('channel') or info_dict.get('uploader_id') or 'Unknown Uploader',
                'upload_date': info_dict.get('upload_date'),  # YYYYMMDD format or None
                'webpage_url': info_dict.get('webpage_url', url),  # Use canonical URL if available, else original URL
                'duration': info_dict.get('duration'), # Duration in seconds or None
                'extractor': info_dict.get('extractor_key', info_dict.get('extractor', 'unknown')), # Platform identifier
                'type_input': type_input,
                # Include additional potentially useful fields
                'view_count': info_dict.get('view_count'),
                'thumbnail': info_dict.get('thumbnail'),
            }
    except yt_dlp.utils.DownloadError as e:
        # Log a warning if metadata extraction fails and use default values
        logging.warning(f"yt-dlp metadata extraction failed for {url}: {e}. Using default metadata.")
        metadata = {
            'title': 'Unknown Title',
            'uploader': 'Unknown Uploader',
            'upload_date': None,
            'webpage_url': url,
            'duration': None,
            'extractor': 'unknown',
            'view_count': None,
            'thumbnail': None,
        }
    except Exception as e:
        # Catch any other unexpected errors during metadata extraction
        logging.error(f"An unexpected error occurred during metadata extraction for {url}: {e}", exc_info=True)
        return None

    # --- Download Command Construction ---
    # Construct the command to execute yt-dlp via subprocess.
    # Key options used:
    # -x (--extract-audio): Extract the audio stream.
    # --audio-format: Specify the desired output audio format (e.g., mp3). Requires ffmpeg.
    # --no-playlist: Prevent accidental download of entire playlists.
    # --progress: Display download progress in the console output.
    # --no-write-info-json: Avoid creating a separate JSON file for metadata (we already extracted it).
    # --no-simulate: Ensure the actual download happens.
    # --no-abort-on-error: Attempt to continue if parts of the download fail.
    # -o (--output): Define the output filename template.
        # --- Download Command Construction ---
    # --- Download Command Construction ---
    base_cmd = [
        YT_DLP_PATH,
        url,
        "-x",                           # extract audio
        "--audio-format", Config.AUDIO_FORMAT,
        "--no-playlist",
        "--no-write-info-json",
        "--progress",
        "--no-simulate",
        "--no-abort-on-error",
        "-o", str(output_path_template),
    ]

    # Optional enrichments, only added if present in Config
    enrichments = []
    try:
        if getattr(Config, "YTDLP_USER_AGENT", ""):
            enrichments += ["--user-agent", Config.YTDLP_USER_AGENT]

        # Region/consent
        if getattr(Config, "YTDLP_GEO_BYPASS", True):
            enrichments += ["--geo-bypass", "--geo-bypass-country", getattr(Config, "YTDLP_GEO_COUNTRY", "US")]

        # Cookies: prefer file over browser
        if getattr(Config, "YTDLP_COOKIES_FILE", ""):
            enrichments += ["--cookies", Config.YTDLP_COOKIES_FILE]
        elif getattr(Config, "YTDLP_COOKIES_FROM_BROWSER", ""):
            enrichments += ["--cookies-from-browser", Config.YTDLP_COOKIES_FROM_BROWSER]
    except Exception:
        # If Config doesn't have these attributes, just continue with base
        pass

    # Prepare attempts: primary (as-is), then fallback with extractor-args if configured
    attempts = []
    attempts.append(("primary", base_cmd + enrichments))

    extractor_args = getattr(Config, "YTDLP_EXTRACTOR_ARGS", "").strip() if hasattr(Config, "YTDLP_EXTRACTOR_ARGS") else ""
    if extractor_args:
        attempts.append(("extractor_args", attempts[0][1] + ["--extractor-args", extractor_args]))

    retries = 0
    try:
        retries = int(getattr(Config, "YTDLP_RETRIES", 2))
    except Exception:
        retries = 2

    # Optionally add one more aggressive attempt if retries > 1
    if retries > 1 and extractor_args:
        attempts.append(("aggressive", base_cmd + enrichments + ["--extractor-args", extractor_args]))

    logging.info(f"Attempting to download audio from: {url}")

    last_err = None
    # Execute up to 1 + retries attempts
    for label, cmd in attempts[: 1 + retries]:
        logging.debug(f"[yt-dlp] Attempt '{label}': {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            logging.info(f"yt-dlp stdout:\n{result.stdout}")
            if result.stderr:
                logging.warning(f"yt-dlp stderr:\n{result.stderr}")

            # Check expected file
            if final_audio_path.exists():
                logging.info(f"Successfully downloaded audio (attempt '{label}') to: {final_audio_path}")
                return (str(final_audio_path), metadata)

            # Sometimes YouTube yields a different extension (e.g., .webm with Opus)
            audio_files = list(output_dir.glob(f"{base_filename}.*"))
            possible_audio = [f for f in audio_files if f.suffix.lower() in ['.mp3', '.m4a', '.wav', '.ogg', '.opus', '.webm']]
            if possible_audio:
                found_path = str(possible_audio[0])
                logging.warning(
                    f"Expected '{final_audio_path}' not found; returning '{found_path}' instead "
                    f"(attempt '{label}')."
                )
                return (found_path, metadata)

            logging.error(f"Attempt '{label}' completed but no output file was produced.")
            last_err = RuntimeError("yt-dlp completed without producing expected output.")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            logging.error(f"yt-dlp failed (Exit Code {e.returncode}) on attempt '{label}'. URL: {url}")
            logging.error(f"Command: {' '.join(e.cmd)}")
            logging.error(f"Stderr:\n{stderr}")
            last_err = e
            continue
        except FileNotFoundError:
            logging.error(f"'{YT_DLP_PATH}' command not found. Is yt-dlp installed and in PATH?")
            return None
        except Exception as e:
            logging.error(f"Unexpected error during download (attempt '{label}'): {e}", exc_info=True)
            last_err = e
            continue

    logging.error("All yt-dlp attempts failed.")
    if last_err:
        logging.error(f"Last error: {last_err}")
    return None



