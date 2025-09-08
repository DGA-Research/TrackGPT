"""
Module for downloading audio from URLs using yt-dlp.

Handles:
- Finding yt-dlp and ffmpeg executables
- Downloading audio in specified format
- Extracting standardized metadata
- Error handling and fallback behavior
"""
import os
import shutil
import tempfile
import subprocess
import logging
import sys
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
    Returns the full path if found, otherwise None.
    """
    try:
        return yt_dlp.utils.exe_path()  # bundled path if available
    except AttributeError:
        import shutil as _shutil
        return _shutil.which("yt-dlp")


def find_ffmpeg_executable() -> Optional[str]:
    """
    Locates the ffmpeg executable on the system by searching PATH.
    """
    import shutil as _shutil
    return _shutil.which("ffmpeg")


YT_DLP_PATH = find_yt_dlp_executable()
if not YT_DLP_PATH:
    print("ERROR: 'yt-dlp' command not found in PATH or via library helper.", file=sys.stderr)
    print("Please ensure yt-dlp is installed and accessible.", file=sys.stderr)

FFMPEG_PATH = find_ffmpeg_executable()
if not FFMPEG_PATH:
    print("ERROR: 'ffmpeg' command not found in system PATH.", file=sys.stderr)
    print("Please ensure ffmpeg is installed and accessible.", file=sys.stderr)
    sys.exit(1)


# --- Core Function ---
def download_audio(url: str, output_dir: Path, base_filename: str, type_input) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Downloads audio from a given URL using yt-dlp.

    Args:
        url: The URL of the video or audio source (e.g., YouTube, Vimeo).
        output_dir: Directory where the downloaded audio file should be saved.
        base_filename: Base name for the output audio file (without extension).
        type_input: Arbitrary tag passed into metadata for upstream use.

    Returns:
        (path_to_audio, metadata) if successful; otherwise None.
    """
    if not YT_DLP_PATH:
        logging.error("yt-dlp executable not found. Cannot download.")
        return None

    # Ensure output dir exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {output_dir}: {e}")
        return None

    # Paths & filenames
    output_path_template = output_dir / f"{base_filename}.%(ext)s"
    final_audio_path = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"

    # --- Prepare writable cookies (yt-dlp writes back to cookiefile) ---
    orig_cookies_file = getattr(Config, "YTDLP_COOKIES_FILE", "").strip()
    cookies_from_browser = getattr(Config, "YTDLP_COOKIES_FROM_BROWSER", "").strip()
    temp_cookies_file: Optional[str] = None

    def _prepare_writable_cookies(src_path: str) -> Optional[str]:
        if not src_path:
            return None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".cookies.txt")
            os.close(fd)
            shutil.copyfile(src_path, tmp_path)
            os.chmod(tmp_path, 0o600)  # ensure only this process can write
            logging.info(f"Using temp cookies file at: {tmp_path}")
            return tmp_path
        except Exception as e:
            logging.warning(f"Could not create temp cookies file from '{src_path}': {e}. Continuing without cookies.")
            return None

    if orig_cookies_file:
        temp_cookies_file = _prepare_writable_cookies(orig_cookies_file)

    def _cleanup_temp_cookies():
        if temp_cookies_file:
            try:
                os.remove(temp_cookies_file)
                logging.info(f"Removed temp cookies file: {temp_cookies_file}")
            except Exception as e:
                logging.debug(f"Could not remove temp cookies file '{temp_cookies_file}': {e}")

    # --- Metadata Extraction (no download) ---
    ydl_opts: Dict[str, Any] = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }

    user_agent = getattr(Config, "YTDLP_USER_AGENT", "").strip()
    if temp_cookies_file:
        ydl_opts['cookiefile'] = temp_cookies_file
    elif cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = cookies_from_browser
    if user_agent:
        ydl_opts['user_agent'] = user_agent

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            metadata = {
                'title': info_dict.get('title', 'Unknown Title'),
                'uploader': info_dict.get('uploader') or info_dict.get('channel') or info_dict.get('uploader_id') or 'Unknown Uploader',
                'upload_date': info_dict.get('upload_date'),
                'webpage_url': info_dict.get('webpage_url', url),
                'duration': info_dict.get('duration'),
                'extractor': info_dict.get('extractor_key', info_dict.get('extractor', 'unknown')),
                'type_input': type_input,
                'view_count': info_dict.get('view_count'),
                'thumbnail': info_dict.get('thumbnail'),
            }
    except yt_dlp.utils.DownloadError as e:
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
        logging.error(f"An unexpected error occurred during metadata extraction for {url}: {e}", exc_info=True)
        _cleanup_temp_cookies()
        return None

    # --- Build command attempts ---
    base_cmd = [
        YT_DLP_PATH, url,
        "-x",
        "--audio-format", Config.AUDIO_FORMAT,
        "--no-playlist", "--no-write-info-json",
        "--progress", "--no-simulate", "--no-abort-on-error",
        "--restrict-filenames",
        "-o", str(output_path_template),
        "--force-ipv4",
    ]

    enrich = []
    if user_agent:
        enrich += [
            "--user-agent", user_agent,
            "--add-header", "Accept-Language: en-US,en;q=0.9",
            "--add-header", "Referer: https://www.youtube.com/",
        ]
    if getattr(Config, "YTDLP_GEO_BYPASS", True):
        enrich += ["--geo-bypass", "--geo-bypass-country", getattr(Config, "YTDLP_GEO_COUNTRY", "US")]

    # Cookies for CLI
    if temp_cookies_file:
        enrich += ["--cookies", temp_cookies_file]
    elif cookies_from_browser:
        enrich += ["--cookies-from-browser", cookies_from_browser]

    attempts = []
    # A) Primary (with extraction & conversion)
    attempts.append(("primary", base_cmd + enrich))

    # B) tv_embedded client (avoid PO tokens seen on android/ios)
    tv_cmd = base_cmd + enrich + ["--extractor-args", "youtube:player_client=tv_embedded"]
    attempts.append(("tv_embedded", tv_cmd))

    # C) Force web page path
    web_cmd = base_cmd + enrich + ["--extractor-args", "youtube:webpage_download_web=1"]
    attempts.append(("web_forced", web_cmd))

    # D) Direct bestaudio ladder; convert locally to mp3 if needed
    fmt_ladder = "251/140/bestaudio/best"
    ba_cmd = [
        YT_DLP_PATH, url,
        "-f", fmt_ladder,
        "--no-playlist", "--no-write-info-json",
        "--progress", "--no-simulate", "--no-abort-on-error",
        "--restrict-filenames",
        "-o", str(output_path_template),
        "--force-ipv4",
    ] + enrich
    attempts.append(("bestaudio_ladder", ba_cmd))

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
            logging.error(f"ffmpeg failed to convert '{path_in}' → '{path_out}':\n{e.stderr}")
        except Exception as e:
            logging.error(f"Unexpected ffmpeg error: {e}", exc_info=True)
        return None

    logging.info(f"Attempting to download audio from: {url}")

    last_err = None
    retries = int(getattr(Config, "YTDLP_RETRIES", 2))

    for label, cmd in attempts[: 1 + retries + 3]:
        logging.debug(f"[yt-dlp] Attempt '{label}': {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding='utf-8'
            )
            logging.info(f"yt-dlp stdout:\n{result.stdout}")
            if result.stderr:
                logging.warning(f"yt-dlp stderr:\n{result.stderr}")

            # Success case 1: final mp3 exists
            if final_audio_path.exists():
                logging.info(f"Success ({label}) → {final_audio_path}")
                _cleanup_temp_cookies()
                return (str(final_audio_path), {**metadata, "download_attempt": label})

            # Success case 2: another audio exists; convert if needed
            audio_candidates = sorted(
                output_dir.glob(f"{base_filename}.*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            for cand in audio_candidates:
                if cand.suffix.lower() in [".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav"]:
                    if cand.suffix.lower() == ".mp3":
                        logging.info(f"Success ({label}) → {cand}")
                        _cleanup_temp_cookies()
                        return (str(cand), {**metadata, "download_attempt": label})
                    out = _ensure_mp3(cand, final_audio_path)
                    if out:
                        logging.info(f"Success ({label}) + convert → {out}")
                        _cleanup_temp_cookies()
                        return (out, {**metadata, "download_attempt": label})

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
            logging.error(f"'{YT_DLP_PATH}' not found. Is yt-dlp installed and in PATH?")
            _cleanup_temp_cookies()
            return None
        except Exception as e:
            logging.error(f"Unexpected error during download (attempt '{label}'): {e}", exc_info=True)
            last_err = e
            continue

    _cleanup_temp_cookies()
    logging.error("All yt-dlp attempts failed.")
    if last_err:
        logging.error(f"Last error: {last_err}")
    return None
