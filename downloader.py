"""
Module for downloading audio from URLs using yt-dlp.

Handles:
- Finding yt-dlp and ffmpeg executables
- Downloading audio in specified format
- Extracting standardized metadata
- Cookies (file, base64, browser) and UTF-8 normalization
- Geo + client-rotation retry ladder
- Error handling and fallback behavior
"""
import os
import shutil
import tempfile
import subprocess
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import urllib.request
import base64

from config import Config

log = logging.getLogger(__name__)

def _log_egress_ip(proxy_url: Optional[str]) -> None:
    try:
        handlers = []
        if proxy_url:
            handlers.append(urllib.request.ProxyHandler({'http': proxy_url, 'https': proxy_url}))
        opener = urllib.request.build_opener(*handlers)
        with opener.open("https://api.ipify.org", timeout=5) as r:
            ip = r.read().decode("utf-8", "ignore")
            log.info("Egress IP (via proxy=%s): %s", bool(proxy_url), ip)
    except Exception as e:
        log.debug("Egress IP check failed: %s", e)

# --- Dependency Checks ---
try:
    import yt_dlp
except ImportError:
    print("ERROR: 'yt-dlp' library not found. Install using: pip install yt-dlp", file=sys.stderr)
    sys.exit(1)

def find_yt_dlp_executable() -> Optional[str]:
    """Locates the yt-dlp executable on the system."""
    try:
        return yt_dlp.utils.exe_path()  # bundled path if available
    except AttributeError:
        import shutil as _shutil
        return _shutil.which("yt-dlp")

def find_ffmpeg_executable() -> Optional[str]:
    """Locates the ffmpeg executable on the system by searching PATH."""
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

# --- Helpers: cookies handling ---
def _ensure_utf8_netscape(cookie_path: str, temp_list: List[str]) -> str:
    """
    Ensure cookie file is UTF-8 text. If not, re-encode to UTF-8 in a temp file.
    Also detect accidental SQLite DBs.
    """
    try:
        with open(cookie_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        log.warning("Cannot read cookies file '%s': %s", cookie_path, e)
        return cookie_path

    if data.startswith(b"SQLite format 3"):
        log.error("Provided cookies file appears to be a Chrome/SQLite DB, not a Netscape cookies.txt export. "
                  "Please export with a 'cookies.txt' extension/format.")
        return cookie_path

    # already utf-8?
    try:
        data.decode('utf-8')
        return cookie_path
    except UnicodeDecodeError:
        pass

    # try utf-8-sig else latin-1
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = data.decode('latin-1')

    text = text.replace('\r\n', '\n')
    fd, newp = tempfile.mkstemp(suffix=".cookies.utf8.txt")
    os.close(fd)
    with open(newp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.chmod(newp, 0o600)
    temp_list.append(newp)
    log.info("Re-encoded cookies to UTF-8 at: %s", newp)
    return newp

def _download_cookies_to_temp(url: str) -> Optional[str]:
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".cookies.txt")
        os.close(fd)
        with urllib.request.urlopen(url, timeout=10) as resp, open(tmp_path, "wb") as out:
            out.write(resp.read())
        os.chmod(tmp_path, 0o600)
        log.info("Fetched cookies from URL into: %s", tmp_path)
        return tmp_path
    except Exception as e:
        log.warning("Could not fetch cookies from %s: %s", url, e)
        return None

def _materialize_cookies(temp_list: List[str]) -> Optional[str]:
    """
    Materialize cookies from (in order of precedence):
      1) YTDLP_COOKIES_B64
      2) YTDLP_COOKIES_FILE
      3) YTDLP_COOKIES_URL
    Returns a readable temp cookies file path or None.
    """
    # 1) base64
    cookies_b64 = os.getenv(
        "YTDLP_COOKIES_B64",
        getattr(Config, "YTDLP_COOKIES_B64", "") if hasattr(Config, "YTDLP_COOKIES_B64") else ""
    ).strip()
    if cookies_b64:
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".cookies.txt")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(cookies_b64))
            os.chmod(tmp_path, 0o600)
            temp_list.append(tmp_path)
            log.info("Decoded cookies from secrets into: %s", tmp_path)
            return _ensure_utf8_netscape(tmp_path, temp_list)
        except Exception as e:
            log.warning("Failed to decode YTDLP_COOKIES_B64: %s", e)

    # 2) file path
    file_path = os.getenv("YTDLP_COOKIES_FILE", getattr(Config, "YTDLP_COOKIES_FILE", "")).strip()
    if file_path:
        try:
            fd, tmp_copy = tempfile.mkstemp(suffix=".cookies.txt")
            os.close(fd)
            shutil.copyfile(file_path, tmp_copy)
            os.chmod(tmp_copy, 0o600)
            temp_list.append(tmp_copy)
            log.info("Using temp cookies file at: %s", tmp_copy)
            return _ensure_utf8_netscape(tmp_copy, temp_list)
        except Exception as e:
            log.warning("Could not create temp cookies file from '%s': %s. Continuing without cookies.", file_path, e)

    # 3) URL
    cookies_url = os.getenv("YTDLP_COOKIES_URL", getattr(Config, "YTDLP_COOKIES_URL", "") if hasattr(Config, "YTDLP_COOKIES_URL") else "").strip()
    if cookies_url:
        fetched = _download_cookies_to_temp(cookies_url)
        if fetched:
            temp_list.append(fetched)
            return _ensure_utf8_netscape(fetched, temp_list)

    return None

# --- Helper: ffmpeg ensure mp3 ---
def _ensure_mp3(path_in: Path, path_out: Path) -> Optional[str]:
    try:
        if path_in.suffix.lower() == f".{Config.AUDIO_FORMAT.lower()}":
            return str(path_in)
        conv = subprocess.run(
            [FFMPEG_PATH, "-y", "-i", str(path_in), "-vn",
             "-acodec", "libmp3lame", "-q:a", "2", str(path_out)],
            check=True, capture_output=True, text=True, encoding="utf-8"
        )
        log.info("ffmpeg stdout:\n%s", conv.stdout)
        if conv.stderr:
            log.warning("ffmpeg stderr:\n%s", conv.stderr)
        if path_out.exists():
            return str(path_out)
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg failed to convert '%s' → '%s':\n%s", path_in, path_out, e.stderr)
    except Exception as e:
        log.error("Unexpected ffmpeg error: %s", e, exc_info=True)
    return None

# --- Helper: build attempt ladder ---
from typing import Tuple as _TupleList  # alias just to emphasize Tuple use in return type

def _build_attempts(
    yt_path: str,
    url: str,
    out_tmpl: str,
    enrich: List[str],
    prefer_format: str,
    geo_countries: List[str],
    supports_cookies: bool,
    use_geo_bypass: bool,
) -> List[Tuple[str, List[str]]]:
    """
    Build a retry ladder that rotates:
      - geo country (default: US)
      - player_client (web-first when cookies are present)
      - direct '-x' audio extraction
    """
    geo_list = geo_countries or ["US"]
    attempts: List[Tuple[str, List[str]]] = []

    for country in geo_list:
        base_common = [
            yt_path, url,
            "--ignore-config",
            "--no-playlist", "--no-write-info-json",
            "--progress", "--no-simulate", "--no-abort-on-error",
            "--restrict-filenames",
            "-o", out_tmpl,
            "--force-ipv4",
        ] + enrich
        if use_geo_bypass:
            base_common += ["--geo-bypass", "--geo-bypass-country", country]

        # Prefer grabbing a container stream first, then extract/convert if needed
        common_fmt = base_common + ["-f", prefer_format]

        if supports_cookies:
            # android/ios clients do not support cookies → web clients only
            attempts += [
                (f"{country}_mweb",       common_fmt + ["--extractor-args", "youtube:player_client=mweb"]),
                (f"{country}_tv_embed",   common_fmt + ["--extractor-args", "youtube:player_client=tv_embedded"]),
                (f"{country}_web_forced", common_fmt + ["--extractor-args", "youtube:webpage_download_web=1"]),
                (f"{country}_extract",    base_common + ["-x", "--audio-format", Config.AUDIO_FORMAT]),
            ]
        else:
            # No cookies: rotate through all clients
            attempts += [
                (f"{country}_android",    common_fmt + ["--extractor-args", "youtube:player_client=android"]),
                (f"{country}_ios",        common_fmt + ["--extractor-args", "youtube:player_client=ios"]),
                (f"{country}_mweb",       common_fmt + ["--extractor-args", "youtube:player_client=mweb"]),
                (f"{country}_tv_embed",   common_fmt + ["--extractor-args", "youtube:player_client=tv_embedded"]),
                (f"{country}_web_forced", common_fmt + ["--extractor-args", "youtube:webpage_download_web=1"]),
                (f"{country}_extract",    base_common + ["-x", "--audio-format", Config.AUDIO_FORMAT]),
            ]

    return attempts

# --- Core Function ---
def download_audio(url: str, output_dir: Path, base_filename: str, type_input) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Downloads audio from a given URL using yt-dlp with resilient fallbacks.

    Returns:
        (path_to_audio, metadata) if successful; otherwise None.
    """
    # Sanity
    if not YT_DLP_PATH:
        log.error("yt-dlp executable not found. Cannot download.")
        return None
    if not FFMPEG_PATH:
        log.error("ffmpeg not found in PATH. Cannot convert audio.")
        return None

    # Ensure output dir exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("Failed to create output directory %s: %s", output_dir, e)
        return None

    # Paths & filenames
    output_path_template = str(output_dir / f"{base_filename}.%(ext)s")
    final_audio_path = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"

    # --- Cookies & UA setup ---
    temp_paths_to_cleanup: List[str] = []
    temp_cookies_file: Optional[str] = _materialize_cookies(temp_paths_to_cleanup)

    cookies_from_browser = os.getenv(
        "YTDLP_COOKIES_FROM_BROWSER",
        getattr(Config, "YTDLP_COOKIES_FROM_BROWSER", "") if hasattr(Config, "YTDLP_COOKIES_FROM_BROWSER") else ""
    ).strip()

    user_agent = os.getenv("YTDLP_USER_AGENT", getattr(Config, "YTDLP_USER_AGENT", "")).strip()

    proxy_url = os.getenv(
        "YTDLP_PROXY_URL",
        getattr(Config, "YTDLP_PROXY_URL", "") if hasattr(Config, "YTDLP_PROXY_URL") else ""
    ).strip()

    # Log egress IP (helps diagnose region blocks)
    _log_egress_ip(proxy_url or None)

    # --- Metadata (no download) ---
    ydl_opts: Dict[str, Any] = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    if temp_cookies_file:
        ydl_opts['cookiefile'] = temp_cookies_file
    elif cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = cookies_from_browser
    if user_agent:
        ydl_opts['user_agent'] = user_agent
    if proxy_url:
        ydl_opts['proxy'] = proxy_url

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
        log.warning("yt-dlp metadata extraction failed for %s: %s. Using default metadata.", url, e)
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
        log.error("Unexpected error during metadata extraction for %s: %s", url, e, exc_info=True)
        for p in temp_paths_to_cleanup:
            try:
                os.remove(p)
            except Exception:
                pass
        return None

    # --- Enrichment flags (headers/cookies/proxy) for CLI ---
    enrich: List[str] = []
    if user_agent:
        enrich += [
            "--user-agent", user_agent,
            "--add-header", "Accept-Language: en-US,en;q=0.9",
            "--add-header", "Referer: https://www.youtube.com/",
        ]
    if temp_cookies_file:
        enrich += ["--cookies", temp_cookies_file]
    elif cookies_from_browser:
        enrich += ["--cookies-from-browser", cookies_from_browser]
    if proxy_url:
        enrich += ["--proxy", proxy_url]

    # --- Build attempts ladder (exactly once) ---
    geo_countries_cfg = getattr(Config, "YTDLP_GEO_COUNTRIES", None)
    if geo_countries_cfg:
        geo_countries = [c.strip() for c in str(geo_countries_cfg).split(",") if c.strip()]
    else:
        # Prefer Puerto Rico first, then US by default
        geo_countries_env = os.getenv("YTDLP_GEO_COUNTRIES", "PR,US")
        geo_countries = [c.strip() for c in geo_countries_env.split(",") if c.strip()]

    supports_cookies = bool(temp_cookies_file or cookies_from_browser)
    # If a proxy is supplied, let the proxy decide region; skip geo-bypass flags.
    use_geo_bypass = not bool(proxy_url)

    attempts = _build_attempts(
        yt_path=YT_DLP_PATH,
        url=url,
        out_tmpl=output_path_template,
        enrich=enrich,
        prefer_format="251/140/bestaudio/best",
        geo_countries=geo_countries,
        supports_cookies=supports_cookies,
        use_geo_bypass=use_geo_bypass,
    )

    # --- Run attempts ---
    last_err: Optional[Exception] = None
    try:
        for label, cmd in attempts:
            try:
                log.debug("[yt-dlp] Attempt '%s': %s", label, " ".join(cmd))
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                if result.stdout:
                    log.info("yt-dlp stdout:\n%s", result.stdout)
                if result.stderr:
                    log.debug("yt-dlp stderr:\n%s", result.stderr)

                # Success case 1: final mp3 exists
                if final_audio_path.exists():
                    log.info("Success (%s) → %s", label, final_audio_path)
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
                            log.info("Success (%s) → %s", label, cand)
                            return (str(cand), {**metadata, "download_attempt": label})
                        out = _ensure_mp3(cand, final_audio_path)
                        if out:
                            log.info("Success (%s) + convert → %s", label, out)
                            return (out, {**metadata, "download_attempt": label})

                log.error("Attempt '%s' completed but no usable audio file was produced.", label)
                last_err = RuntimeError("yt-dlp completed without producing expected output.")

            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                log.error("yt-dlp failed (Exit %s) on attempt '%s'. URL: %s", e.returncode, label, url)
                log.error("Command: %s", " ".join(e.cmd if isinstance(e.cmd, list) else [str(e.cmd)]))
                if stderr:
                    log.error("Stderr:\n%s", stderr)
                last_err = e
                continue
            except FileNotFoundError:
                log.error("'%s' not found. Is yt-dlp installed and in PATH?", YT_DLP_PATH)
                last_err = FileNotFoundError("yt-dlp not found")
                break
            except Exception as e:
                log.error("Unexpected error during download (attempt '%s'): %s", label, e, exc_info=True)
                last_err = e
                continue
    finally:
        # Always cleanup temp cookies we created
        for p in temp_paths_to_cleanup:
            try:
                os.remove(p)
                log.info("Removed temp cookies file: %s", p)
            except Exception:
                pass

    log.error("All yt-dlp attempts failed.")
    if last_err:
        log.error("Last error: %s", last_err)
    return None
