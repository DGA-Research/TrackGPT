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
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from config import Config

from urllib.parse import urlsplit


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

    logging.info("Starting download for %s", url)
    logging.debug("Requested output_dir=%s base_filename=%s type_input=%s", output_dir, base_filename, type_input)
    # Check if yt-dlp executable was found during initial checks
    if not YT_DLP_PATH:
         logging.error("yt-dlp executable not found. Cannot download.")
         return None

    # Define output paths using the base filename and configured audio format
    output_path_template = output_dir / f"{base_filename}.%(ext)s"
    final_audio_path = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"

    cookies_source = os.getenv("YTDLP_COOKIES_FILE") or getattr(Config, "YTDLP_COOKIES_FILE", None)
    cookies_args = []
    if cookies_source:
        cookies_path = Path(cookies_source).expanduser()
        if cookies_path.exists():
            cookies_args = ["--cookies", str(cookies_path)]
            logging.info(f"Using cookies file: {cookies_path}")
        else:
            logging.warning(f"Cookies file not found at {cookies_path}. Continuing without cookies.")
    else:
        logging.debug("No cookies file configured for yt-dlp.")

    # Ensure the output directory exists, creating it if necessary
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {output_dir}: {e}")
        return None

    logging.debug("Output directory ready: %s", output_dir)

    # --- Metadata Extraction ---
    # Extract metadata using the yt-dlp library without downloading the video.
    # This allows us to get information even if the download later fails.
    ydl_opts = {
        'quiet': True,          # Suppress console output from yt-dlp library
        'no_warnings': True,    # Hide warnings from yt-dlp library
        'extract_flat': False,  # Ensure full metadata is extracted
    }
    logging.info("Extracting metadata via yt-dlp for %s", url)
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
            logging.info("Metadata extraction succeeded: title='%s', extractor='%s'", metadata.get('title'), metadata.get('extractor'))
            logging.debug("Metadata details: duration=%s view_count=%s", metadata.get('duration'), metadata.get('view_count'))
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
    cmd = [
        YT_DLP_PATH,
        url,
        "-x",  # Extract audio only
        "--audio-format", Config.AUDIO_FORMAT,  # Convert to specified format (requires ffmpeg)
        "--no-playlist",       # Avoid downloading entire playlists
        "--no-write-info-json", # Skip writing metadata JSON file
        "--progress",          # Show download progress
        "--no-simulate",       # Actually download (no dry run)
        "--no-abort-on-error", # Continue if parts fail
        "-o", str(output_path_template), # Output filename template
    ]
    if cookies_args:
        cmd.extend(cookies_args)

    # Note: ffmpeg must be installed and in the system's PATH for audio format conversion.
    # The command execution will handle the download, audio extraction, conversion, and saving.

    logging.info(f"Attempting to download audio from: {url}")
    # Log the command being executed for debugging purposes
    logging.debug(f"yt-dlp command: {' '.join(cmd)}")
    logging.info("Invoking yt-dlp CLI to download audio...")

    try:
        # Execute the yt-dlp command using subprocess
        result = subprocess.run(
            cmd,
            check=True,         # Raise CalledProcessError if the command returns a non-zero exit code
            capture_output=True, # Capture standard output and standard error
            text=True,          # Decode stdout/stderr as text
            encoding='utf-8'    # Specify encoding for text decoding
        )
        # Log the standard output and standard error from the yt-dlp process
        logging.info(f"yt-dlp stdout:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"yt-dlp stderr:\n{result.stderr}")

        # Verify if the expected final audio file exists after the download
        if final_audio_path.exists():
            logging.info(f"Successfully downloaded audio to: {final_audio_path}")
            # Return the path to the downloaded file and the extracted metadata
            return (str(final_audio_path), metadata)
        else:
            # Log an error if the expected file is not found, even if the process exited successfully
            logging.error(f"yt-dlp completed but expected output file '{final_audio_path}' not found.")
            logging.error("Please check yt-dlp output above for clues.")
            # Attempt to find any audio file that might have been created with a different extension
            audio_files = list(output_dir.glob(f"{base_filename}.*"))
            possible_audio = [f for f in audio_files if f.suffix.lower() in ['.mp3', '.m4a', '.wav', '.ogg', '.opus']]
            if possible_audio:
                 # If an alternative audio file is found, log a warning and return its path
                 found_path = str(possible_audio[0])
                 logging.warning(f"Found an alternative audio file: {found_path}. Returning this path.")
                 return (found_path, metadata)
            # If no suitable audio file is found, return None
            return None

    # --- Error Handling Strategy for subprocess execution ---
    # Handle specific exceptions that can occur during subprocess execution:
    # 1. CalledProcessError: Raised when the yt-dlp command returns a non-zero exit code.
    #    - Log the error code, command, and stderr for debugging.
    # 2. FileNotFoundError: Raised if the yt-dlp executable is not found.
    #    - Log a clear error message indicating the missing executable.
    # 3. Other exceptions: Catch any other unexpected errors during the process.
    #    - Log the error with traceback information.
    except subprocess.CalledProcessError as e:
        logging.error(f"yt-dlp failed (Exit Code {e.returncode}). URL: {url}")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Stderr:\n{e.stderr}")
        return None
    except FileNotFoundError:
        logging.error(f"'{YT_DLP_PATH}' command not found. Is yt-dlp installed and in PATH?")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during download: {e}", exc_info=True)
        return None


