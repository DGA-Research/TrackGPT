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

logging.info(
    f"yt-dlp cfg: cookies_file={getattr(Config,'YTDLP_COOKIES_FILE','')!r} "
    f"ua_set={bool(getattr(Config,'YTDLP_USER_AGENT',''))} "
    f"retries={getattr(Config,'YTDLP_RETRIES',2)}"
)


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
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': False}

    if Config.YTDLP_COOKIES_FILE:
        ydl_opts['cookiefile'] = Config.YTDLP_COOKIES_FILE
    elif Config.YTDLP_COOKIES_FROM_BROWSER:
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
        YT_DLP_PATH, url,
        "-x",
        "--audio-format", Config.AUDIO_FORMAT,
        "--no-playlist", "--no-write-info-json",
        "--progress", "--no-simulate", "--no-abort-on-error",
        "-o", str(output_path_template),
        "--force-ipv4",  # often helps on container hosts
    ]

    # Optional enrichments
    enrich = []
    ua = getattr(Config, "YTDLP_USER_AGENT", "").strip()
    if ua:
        enrich += ["--user-agent", ua, "--add-header", "Accept-Language: en-US,en;q=0.9",
                   "--add-header", "Referer: https://www.youtube.com/"]

    if getattr(Config, "YTDLP_GEO_BYPASS", True):
        enrich += ["--geo-bypass", "--geo-bypass-country", getattr(Config, "YTDLP_GEO_COUNTRY", "US")]

    cookies_file = getattr(Config, "YTDLP_COOKIES_FILE", "").strip()
    cookies_from_browser = getattr(Config, "YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if cookies_file:
        enrich += ["--cookies", cookies_file]
    elif cookies_from_browser:
        enrich += ["--cookies-from-browser", cookies_from_browser]

    # Attempt A: original (with enrichments)
    attempts = []
    attempts.append(("primary", base_cmd + enrich))

    # Attempt B: tv_embedded client (no PO token)
    tv_cmd = base_cmd + enrich + ["--extractor-args", "youtube:player_client=tv_embedded,player_skip=webpage,client_location=US"]
    attempts.append(("tv_embedded", tv_cmd))

    # Attempt C: ios client (no PO token)
    ios_cmd = base_cmd + enrich + ["--extractor-args", "youtube:player_client=ios,client_location=US"]
    attempts.append(("ios_client", ios_cmd))

    # Attempt D: force web path (some hosts need explicit webpage download)
    web_cmd = base_cmd + enrich + ["--extractor-args", "youtube:webpage_download_web=1,client_location=US"]
    attempts.append(("web_forced", web_cmd))

    # Attempt E: direct bestaudio ladder; convert with ffmpeg if needed
    fmt_ladder = "251/140/bestaudio/best"
    ba_cmd = [
        YT_DLP_PATH, url,
        "-f", fmt_ladder,
        "--no-playlist", "--no-write-info-json",
        "--progress", "--no-simulate", "--no-abort-on-error",
        "-o", str(output_path_template),
        "--force-ipv4",
    ] + enrich
    attempts.append(("bestaudio_ladder", ba_cmd))

    # Converter
    def _ensure_mp3(path_in: Path, path_out: Path) -> Optional[str]:
        try:
            if path_in.suffix.lower() == f".{Config.AUDIO_FORMAT.lower()}":
                return str(path_in)
            conv = subprocess.run(
                [FFMPEG_PATH, "-y", "-i", str(path_in), "-vn",
                 "-acodec", "libmp3lame", "-q:a", "2", str(path_out)],
                check=True, capture_output=True, text=True, encoding="utf-8"
            )
            logging.info(f"ffmpeg stdout:\n{conv.stdout}")
            if conv.stderr:
                logging.warning(f"ffmpeg stderr:\n{conv.stderr}")
            if path_out.exists():
                return str(path_out)
        except subprocess.CalledProcessError as e:
            logging.error(f"ffmpeg failed: {e.stderr}")
        except Exception as e:
            logging.error(f"ffmpeg unexpected error: {e}", exc_info=True)
        return None

    logging.info(f"Attempting to download audio from: {url}")

    last_err = None
    retries = int(getattr(Config, "YTDLP_RETRIES", 2))
    for label, cmd in attempts[: 1 + retries + 3]:  # allow us to hit the laddered clients
        logging.debug(f"[yt-dlp] Attempt '{label}': {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding='utf-8'
            )
            logging.info(f"yt-dlp stdout:\n{result.stdout}")
            if result.stderr:
                logging.warning(f"yt-dlp stderr:\n{result.stderr}")

            # mp3 already?
            if final_audio_path.exists():
                logging.info(f"Success ({label}) → {final_audio_path}")
                return (str(final_audio_path), metadata)

            # check other audio, convert if needed
            audio_candidates = sorted(
                output_dir.glob(f"{base_filename}.*"),
                key=lambda p: p.stat().st_mtime, reverse=True
            )
            for cand in audio_candidates:
                if cand.suffix.lower() in [".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav"]:
                    if cand.suffix.lower() == ".mp3":
                        logging.info(f"Success ({label}) → {cand}")
                        return (str(cand), metadata)
                    out = _ensure_mp3(cand, final_audio_path)
                    if out:
                        logging.info(f"Success ({label}) + convert → {out}")
                        return (out, metadata)

            logging.error(f"Attempt '{label}' completed but no usable audio file was produced.")
            last_err = RuntimeError("yt-dlp completed without producing expected output.")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            logging.error(f"yt-dlp failed (Exit {e.returncode}) on attempt '{label}'. URL: {url}")
            logging.error(f"Command: {' '.join(e.cmd)}")
            logging.error(f"Stderr:\n{stderr}")
            last_err = e
            continue
        except FileNotFoundError:
            logging.error(f"'{YT_DLP_PATH}' not found. Is yt-dlp installed and on PATH?")
            return None
        except Exception as e:
            logging.error(f"Unexpected error during download (attempt '{label}'): {e}", exc_info=True)
            last_err = e
            continue

    logging.error("All yt-dlp attempts failed.")
    if last_err:
        logging.error(f"Last error: {last_err}")
    return None

