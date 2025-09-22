"""
Module for downloading audio from URLs using yt-dlp.

Handles:
- Finding yt-dlp and ffmpeg executables
- Downloading audio in specified format
- Extracting standardized metadata
- Error handling and Apify fallback for region-locked videos
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import time

import requests
from config import Config


from urllib.parse import urlsplit

def _looks_like_youtube(u: str) -> bool:
    try:
        host = urlsplit(u).netloc.lower()
    except Exception:
        return False
    return ("youtube.com" in host) or ("youtu.be" in host) or ("youtube-nocookie.com" in host)

def _download_non_youtube(
    url: str,
    output_dir: Path,
    base_filename: str,
    *,
    user_agent: str = "",
    cookies_from_browser: str = "",
    proxy_url: str = "",
    metadata: Dict[str, Any] | None = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Generic path for non-YouTube URLs (Brightcove, JWPlayer, news sites, etc.)
    - Adds UA and a same-origin Referer
    - Supports cookies-from-browser and proxy
    - Enforces a subprocess timeout to avoid 'Processing input…' hangs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(output_dir / f"{base_filename}.%(ext)s")
    final_mp3 = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"

    hdrs: list[str] = []
    if user_agent:
        hdrs += ["--user-agent", user_agent, "--add-header", "Accept-Language: en-US,en;q=0.9"]

    # Same-origin referer helps many embeds
    try:
        u = urlsplit(url)
        origin = f"{u.scheme}://{u.netloc}/"
        hdrs += ["--add-header", f"Referer: {origin}"]
    except Exception:
        pass

    if cookies_from_browser:
        hdrs += ["--cookies-from-browser", cookies_from_browser]
    if proxy_url:
        hdrs += ["--proxy", proxy_url]

    cmd = [
        YT_DLP_PATH,
        url,
        "-x", "--audio-format", Config.AUDIO_FORMAT,
        "--no-playlist", "--no-write-info-json",
        "--progress", "--no-simulate", "--no-abort-on-error",
        "-o", out_tpl,
    ] + hdrs

    import subprocess, os
    subproc_timeout_s = int(os.getenv("YTDLP_SUBPROC_TIMEOUT_S", "240"))

    logging.info("Non-YouTube URL detected; using generic download path.")
    logging.debug("yt-dlp (non-yt) command: %s", " ".join(cmd))
    try:
        cp = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=subproc_timeout_s,
        )
        logging.info("yt-dlp stdout:\n%s", cp.stdout)
        if cp.stderr:
            logging.debug("yt-dlp stderr:\n%s", cp.stderr)

        # With -x --audio-format mp3, yt-dlp should produce an mp3
        if final_mp3.exists():
            return str(final_mp3), {**(metadata or {}), "download_attempt": "non_yt"}

        # Fallback: if site produced a different audio container, still accept it
        for cand in sorted(output_dir.glob(f"{base_filename}.*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if cand.suffix.lower() in [".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav"]:
                if cand.suffix.lower() == ".mp3":
                    return str(cand), {**(metadata or {}), "download_attempt": "non_yt"}
                # Let yt-dlp’s -x handle conversion in most cases;
                # if a site skipped it, your existing ffmpeg path can be used instead.
                return str(cand), {**(metadata or {}), "download_attempt": "non_yt"}

        logging.error("Non-YT: yt-dlp completed but no audio file was produced.")
        return None

    except subprocess.TimeoutExpired:
        logging.error("Non-YT: yt-dlp timed out after %s s", subproc_timeout_s)
        return None
    except subprocess.CalledProcessError as e:
        logging.error("Non-YT: yt-dlp failed (exit %s)\nCmd: %s\nStderr:\n%s",
                      e.returncode, " ".join(e.cmd if isinstance(e.cmd, list) else [str(e.cmd)]), e.stderr or "")
        return None


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

def _download_non_youtube(
    url: str,
    output_dir: Path,
    base_filename: str,
    *,
    user_agent: str = "",
    cookies_from_browser: str = "",
    proxy_url: str = "",
    metadata: Dict[str, Any] | None = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Generic path for non-YouTube URLs (Brightcove, JWPlayer, news sites, etc.)
    - Adds UA and a same-origin Referer
    - Supports cookies-from-browser and proxy
    - Enforces a subprocess timeout to avoid 'Processing input…' hangs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(output_dir / f"{base_filename}.%(ext)s")
    final_mp3 = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"

    hdrs: list[str] = []
    if user_agent:
        hdrs += ["--user-agent", user_agent, "--add-header", "Accept-Language: en-US,en;q=0.9"]

    # Same-origin referer helps many embeds
    try:
        u = urlsplit(url)
        origin = f"{u.scheme}://{u.netloc}/"
        hdrs += ["--add-header", f"Referer: {origin}"]
    except Exception:
        pass

    if cookies_from_browser:
        hdrs += ["--cookies-from-browser", cookies_from_browser]
    if proxy_url:
        hdrs += ["--proxy", proxy_url]

    cmd = [
        YT_DLP_PATH,
        url,
        "-x", "--audio-format", Config.AUDIO_FORMAT,
        "--no-playlist", "--no-write-info-json",
        "--progress", "--no-simulate", "--no-abort-on-error",
        "-o", out_tpl,
    ] + hdrs

    import subprocess, os
    subproc_timeout_s = int(os.getenv("YTDLP_SUBPROC_TIMEOUT_S", "240"))

    logging.info("Non-YouTube URL detected; using generic download path.")
    logging.debug("yt-dlp (non-yt) command: %s", " ".join(cmd))
    try:
        cp = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=subproc_timeout_s,
        )
        logging.info("yt-dlp stdout:\n%s", cp.stdout)
        if cp.stderr:
            logging.debug("yt-dlp stderr:\n%s", cp.stderr)

        # With -x --audio-format mp3, yt-dlp should produce an mp3
        if final_mp3.exists():
            return str(final_mp3), {**(metadata or {}), "download_attempt": "non_yt"}

        # Fallback: if site produced a different audio container, still accept it
        for cand in sorted(output_dir.glob(f"{base_filename}.*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if cand.suffix.lower() in [".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav"]:
                if cand.suffix.lower() == ".mp3":
                    return str(cand), {**(metadata or {}), "download_attempt": "non_yt"}
                # Let yt-dlp’s -x handle conversion in most cases;
                # if a site skipped it, your existing ffmpeg path can be used instead.
                return str(cand), {**(metadata or {}), "download_attempt": "non_yt"}

        logging.error("Non-YT: yt-dlp completed but no audio file was produced.")
        return None

    except subprocess.TimeoutExpired:
        logging.error("Non-YT: yt-dlp timed out after %s s", subproc_timeout_s)
        return None
    except subprocess.CalledProcessError as e:
        logging.error("Non-YT: yt-dlp failed (exit %s)\nCmd: %s\nStderr:\n%s",
                      e.returncode, " ".join(e.cmd if isinstance(e.cmd, list) else [str(e.cmd)]), e.stderr or "")
        return None

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
    Downloads audio from a given URL using yt-dlp with resilient fallbacks.
    For non-YouTube links we take a generic path; for YouTube we run a ladder.
    """
    enrich: list[str] = []
    metadata: Dict[str, Any] = {}

    # ---- Proxy first ----
    proxy_url = os.getenv("YTDLP_PROXY_URL", "").strip()
    if not proxy_url:
        ap_pw = os.getenv("APIFY_PROXY_PASSWORD", "").strip()
        ap_cty = os.getenv("APIFY_PROXY_COUNTRY", "US").strip()
        if ap_pw:
            proxy_url = f"http://auto:{ap_pw}@proxy.apify.com:8000/?country={ap_cty}"
    if proxy_url:
        enrich += ["--proxy", proxy_url]

    # ---- Sanity: binaries ----
    if not YT_DLP_PATH:
        log.error("yt-dlp executable not found.")
        return None
    if not FFMPEG_PATH:
        log.error("ffmpeg not found in PATH.")
        return None

    # ---- Ensure output dir ----
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("Failed to create %s: %s", output_dir, e)
        return None

    output_path_template = str(output_dir / f"{base_filename}.%(ext)s")
    final_audio_path = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"

    # ---- Cookies & UA -> temp file ----
    temp_paths_to_cleanup: list[str] = []
    temp_cookies_file: Optional[str] = None

    orig_cookies_file = (os.getenv("YTDLP_COOKIES_FILE") or getattr(Config, "YTDLP_COOKIES_FILE", None)) or None
    cookies_b64 = (os.getenv("YTDLP_COOKIES_B64") or getattr(Config, "YTDLP_COOKIES_B64", None)) or None
    cookies_from_browser = (os.getenv("YTDLP_COOKIES_FROM_BROWSER") or getattr(Config, "YTDLP_COOKIES_FROM_BROWSER", None)) or None
    user_agent = (os.getenv("YTDLP_USER_AGENT") or getattr(Config, "YTDLP_USER_AGENT", None)) or None


    if cookies_b64:
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".cookies.txt")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(cookies_b64))
            os.chmod(tmp_path, 0o600)
            temp_paths_to_cleanup.append(tmp_path)
            orig_cookies_file = tmp_path
            log.info("Decoded cookies into: %s", tmp_path)
        except Exception as e:
            log.warning("Failed to decode YTDLP_COOKIES_B64: %s", e)

    if orig_cookies_file:
        try:
            fd, tmp_copy = tempfile.mkstemp(suffix=".cookies.txt")
            os.close(fd)
            shutil.copyfile(orig_cookies_file, tmp_copy)
            os.chmod(tmp_copy, 0o600)
            temp_paths_to_cleanup.append(tmp_copy)
            temp_cookies_file = _ensure_utf8_netscape(tmp_copy, temp_paths_to_cleanup)
            log.info("Using temp cookies file at: %s", temp_cookies_file)
        except Exception as e:
            log.warning("Could not prepare temp cookies: %s", e)
            temp_cookies_file = None

    def _cleanup_temp_cookies():
        for p in temp_paths_to_cleanup:
            try:
                os.remove(p)
                log.info("Removed temp cookies file: %s", p)
            except Exception:
                pass

    # ---- Metadata FIRST (so both branches can use it) ----
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': False}
    if temp_cookies_file:
        ydl_opts['cookiefile'] = temp_cookies_file
    elif cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = cookies_from_browser
    if user_agent:
        ydl_opts['user_agent'] = user_agent

    allow_non_yt = os.getenv("ALLOW_NON_YT", "0").lower() in ("1", "true", "yes")

    # --- Early branch for non-YouTube hosts ---
    if not _looks_like_youtube(url):
        # NEW: require switch
        if not allow_non_yt:
            log.info("Non-YouTube URL blocked by config; set ALLOW_NON_YT=1 to enable.")
            raise ValueError("Non-YouTube URLs are disabled. Paste a YouTube link or enable ALLOW_NON_YT.")

        # existing code…
        proxy_url = os.getenv("YTDLP_PROXY_URL", "").strip()
        if not proxy_url:
            ap_pw = os.getenv("APIFY_PROXY_PASSWORD", "").strip()
            ap_cty = os.getenv("APIFY_PROXY_COUNTRY", "US").strip()
            if ap_pw:
                proxy_url = f"http://auto:{ap_pw}@proxy.apify.com:8000/?country={ap_cty}"

        # Before calling _download_non_youtube(...)
        try:
            import yt_dlp
            ydl_opts = {"quiet": True, "no_warnings": True, "socket_timeout": 15}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # validate extractor support quickly
                ydl.extract_info(url, download=False)
        except Exception as e:
            log.error("Non-YouTube quick probe failed: %s", e)
            raise ValueError("This site isn’t supported by yt-dlp (or needs cookies/login).")

        return _download_non_youtube(
            url,
            output_dir,
            base_filename,
            user_agent=getattr(Config, "YTDLP_USER_AGENT", ""),
            cookies_from_browser=getattr(Config, "YTDLP_COOKIES_FROM_BROWSER", ""),
            proxy_url=proxy_url,
            metadata=metadata,
        )


    
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
    except Exception as e:
        log.warning("yt-dlp metadata extraction failed for %s: %s. Using defaults.", url, e)
        metadata = {
            'title': 'Unknown Title',
            'uploader': 'Unknown Uploader',
            'upload_date': None,
            'webpage_url': url,
            'duration': None,
            'extractor': 'unknown',
            'view_count': None,
            'thumbnail': None,
            'type_input': type_input,
        }

    # ---- If NOT YouTube: go generic now (metadata is defined) ----
    if not _looks_like_youtube(url):
        try:
            res = _download_non_youtube(
                url,
                output_dir,
                base_filename,
                user_agent=user_agent,
                cookies_file=temp_cookies_file,
                cookies_from_browser=cookies_from_browser,
                proxy_url=proxy_url,
                metadata=metadata,
            )
            return res
        finally:
            _cleanup_temp_cookies()

    # ---- YouTube ladder ----
    if user_agent:
        enrich += [
            "--user-agent", user_agent,
            "--add-header", "Accept-Language: en-US,en;q=0.9",
            "--add-header", "Referer: https://www.youtube.com/",
        ]
    if getattr(Config, "YTDLP_GEO_BYPASS", True):
        enrich += ["--geo-bypass", "--geo-bypass-country", getattr(Config, "YTDLP_GEO_COUNTRY", "US")]
    if temp_cookies_file:
        enrich += ["--cookies", temp_cookies_file]
    elif cookies_from_browser:
        enrich += ["--cookies-from-browser", cookies_from_browser]

    base_cmd = [
        YT_DLP_PATH, url,
        "-x", "--audio-format", Config.AUDIO_FORMAT,
        "--no-playlist", "--no-write-info-json",
        "--progress", "--no-simulate", "--no-abort-on-error",
        "--restrict-filenames",
        "-o", output_path_template,
        "--force-ipv4",
    ] + enrich

    attempts: list[tuple[str, list[str]]] = [
        ("primary", base_cmd),
        ("tv_embedded", base_cmd + ["--extractor-args", "youtube:player_client=tv_embedded"]),
        ("web_forced", base_cmd + ["--extractor-args", "youtube:webpage_download_web=1"]),
        ("bestaudio_ladder",
            [YT_DLP_PATH, url, "-f", "251/140/bestaudio/best",
             "--no-playlist", "--no-write-info-json", "--progress", "--no-simulate",
             "--no-abort-on-error", "--restrict-filenames", "-o", output_path_template,
             "--force-ipv4"] + enrich),
    ]

    log.info("Attempting to download audio from: %s", url)
    apify_tried = False
    last_err: Optional[Exception] = None

    try:
        for label, cmd in attempts:
            try:
                log.debug("[yt-dlp] Attempt '%s': %s", label, " ".join(cmd))
                cp = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
                if cp.stdout:
                    log.info("yt-dlp stdout:\n%s", cp.stdout)
                if cp.stderr:
                    log.debug("yt-dlp stderr:\n%s", cp.stderr)

                if final_audio_path.exists():
                    log.info("Success (%s) → %s", label, final_audio_path)
                    return (str(final_audio_path), {**metadata, "download_attempt": label})

                # any produced audio → convert if needed
                for cand in sorted(output_dir.glob(f"{base_filename}.*"), key=lambda p: p.stat().st_mtime, reverse=True):
                    if cand.suffix.lower() in [".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav"]:
                        if cand.suffix.lower() == ".mp3":
                            log.info("Success (%s) → %s", label, cand)
                            return (str(cand), {**metadata, "download_attempt": label})
                        out = _ensure_mp3(cand, final_audio_path)
                        if out:
                            log.info("Success (%s) + convert → %s", label, out)
                            return (out, {**metadata, "download_attempt": label})

                log.error("Attempt '%s' finished but no usable audio produced.", label)
                last_err = RuntimeError("yt-dlp completed without producing expected output.")

            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                log.error("yt-dlp failed (Exit %s) on '%s'. URL: %s", e.returncode, label, url)
                log.error("Command: %s", " ".join(e.cmd if isinstance(e.cmd, list) else [str(e.cmd)]))
                if stderr:
                    log.error("Stderr:\n%s", stderr)
                last_err = e

                # Early Apify on region lock
                if not apify_tried and _is_region_lock(stderr):
                    apify_tried = True
                    log.info("Region-lock detected on '%s'. Trying Apify…", label)
                    ap = _apify_download_audio(url, output_dir, base_filename)
                    if ap:
                        ap_path, ap_meta = ap
                        return (ap_path, {**metadata, **ap_meta})
                    log.error("Apify fallback failed; continuing ladder.")
                continue

            except FileNotFoundError:
                log.error("'%s' not found. Is yt-dlp in PATH?", YT_DLP_PATH)
                last_err = FileNotFoundError("yt-dlp not found")
                break

            except Exception as e:
                log.error("Unexpected error during attempt '%s': %s", label, e, exc_info=True)
                last_err = e
                continue

    finally:
        _cleanup_temp_cookies()

    log.error("All yt-dlp attempts failed.")
    if not apify_tried:
        log.info("Trying Apify fallback at end…")
        ap = _apify_download_audio(url, output_dir, base_filename)
        if ap:
            ap_path, ap_meta = ap
            return (ap_path, {**metadata, **ap_meta})

    if last_err:
        log.error("Last error: %s", last_err)
    return None




log = logging.getLogger(__name__)

APIFY_ACTOR = os.getenv("APIFY_ACTOR", "streamers~youtube-video-downloader")

# ---- Optional: use Apify Proxy automatically if available ----
apify_pw = os.getenv("APIFY_PROXY_PASSWORD", "")
apify_country = os.getenv("APIFY_PROXY_COUNTRY", "US")
proxy_url = os.getenv("YTDLP_PROXY_URL", "").strip()

log.info(
    "yt-dlp cfg: cookies_file=%r ua_set=%s retries=%s",
    getattr(Config, "YTDLP_COOKIES_FILE", ""),
    bool(getattr(Config, "YTDLP_USER_AGENT", "")),
    getattr(Config, "YTDLP_RETRIES", 2),
)

# --- Dependency Checks ---
try:
    import yt_dlp
except ImportError:
    print("ERROR: 'yt-dlp' library not found. Install using: pip install yt-dlp", file=sys.stderr)
    sys.exit(1)

def _gcs_find_and_download(
    bucket_name: str,
    service_json_str: str,
    output_dir: Path,
    desired_basename: str,
    url: str | None = None,
    max_age_minutes: int = 60,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Search GCS for an audio object produced by the Apify actor and download it.
    Strategy:
      - Try exact desired_basename with *any* audio extension.
      - Try prefix matches for desired_basename.
      - Prefer files that contain the YouTube video ID.
      - Otherwise pick the newest audio file (optionally within recent time window).
    Convert to MP3 if the downloaded file isn't already .mp3.
    """
    try:
        from google.cloud import storage  # lazy import
    except Exception as e:
        log.error("google-cloud-storage not installed: %s", e)
        return None

    try:
        sa_info = json.loads(service_json_str)
        client = storage.Client.from_service_account_info(sa_info)
        bucket = client.bucket(bucket_name)
    except Exception as e:
        log.error("Failed to init GCS client/bucket: %s", e)
        return None

    # Accept many audio extensions, including the actor's occasional ".mpga"
    AUDIO_EXTS = (".mp3", ".mpga", ".m4a", ".webm", ".opus", ".ogg", ".wav")

    def _is_audio_name(name: str) -> bool:
        n = name.lower()
        return n.endswith(AUDIO_EXTS) or n.endswith(".mp3.mpga")

    # Extract YT video ID (for Apify’s naming: "<videoId>_<title>.<ext>")
    def _extract_youtube_id(u: str) -> Optional[str]:
        import re
        m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", u or "")
        if m: return m.group(1)
        m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", u or "")
        return m.group(1) if m else None

    video_id = _extract_youtube_id(url) if url else None

    candidates = []

    # 1) exact desired_basename + any ext (in case you forced a template)
    for ext in AUDIO_EXTS + (".mp3.mpga",):
        try:
            blob = bucket.blob(f"{desired_basename}{ext}")
            if blob.exists(client):
                candidates.append(blob)
        except Exception:
            pass

    # 2) prefix match with desired_basename
    try:
        for blob in client.list_blobs(bucket_name, prefix=desired_basename):
            if _is_audio_name(blob.name):
                candidates.append(blob)
    except Exception as e:
        log.debug("GCS prefix list failed: %s", e)

    # 3) prefer files containing the video ID (most reliable with this actor)
    if video_id:
        try:
            # fast path: objects that start with the id
            for blob in client.list_blobs(bucket_name, prefix=video_id):
                if _is_audio_name(blob.name):
                    candidates.append(blob)
        except Exception as e:
            log.debug("GCS id-prefix list failed: %s", e)
        try:
            # fallback: any object that contains the id
            for blob in client.list_blobs(bucket_name):
                if _is_audio_name(blob.name) and video_id in blob.name:
                    candidates.append(blob)
        except Exception as e:
            log.debug("GCS full scan failed: %s", e)

    # 4) last resort: newest audio in the (recent) bucket
    try:
        import datetime as _dt
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(minutes=max_age_minutes)
    except Exception:
        cutoff = None

    try:
        for blob in client.list_blobs(bucket_name):
            if _is_audio_name(blob.name):
                upd = getattr(blob, "updated", None)
                if (not cutoff) or (upd and upd.replace(tzinfo=None) >= cutoff):
                    candidates.append(blob)
    except Exception as e:
        log.debug("GCS list for fallback failed: %s", e)

    # Dedup & pick newest
    uniq = {b.name: b for b in candidates}
    if not uniq:
        log.error("GCS fallback: no audio-like objects found in '%s'. Looked for %s", bucket_name, AUDIO_EXTS)
        return None

    blobs = list(uniq.values())
    blobs.sort(key=lambda b: getattr(b, "updated", None) or getattr(b, "time_created", None), reverse=True)
    best = blobs[0]
    log.info("GCS candidate: gs://%s/%s", bucket_name, best.name)

    # Download with its original extension, then ensure MP3
    orig_ext = Path(best.name).suffix.lower()
    tmp_path = output_dir / (desired_basename + (orig_ext if orig_ext else ".bin"))
    out_mp3 = output_dir / f"{desired_basename}.mp3"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        best.download_to_filename(tmp_path)

        if tmp_path.suffix != ".mp3":
            final = _ensure_mp3(tmp_path, out_mp3)
            try:
                tmp_path.unlink()
            except Exception:
                pass
            if not final:
                log.error("GCS fallback: conversion failed for %s", best.name)
                return None
            final_path = final
        else:
            final_path = str(tmp_path)

        log.info("GCS fallback success → %s (source: gs://%s/%s)", final_path, bucket_name, best.name)
        meta = {
            "extractor": "apify_gcs",
            "webpage_url": url or "",
            "gcs_bucket": bucket_name,
            "gcs_object": best.name,
            "download_attempt": "apify_gcs_fallback",
        }
        return final_path, meta
    except Exception as e:
        log.error("GCS fallback: error downloading %s: %s", best.name, e)
        return None


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



def _apify_try_gcs_pull(url: str, output_dir: Path, base_filename: str) -> Optional[str]:
    import json
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except Exception as e:
        log.error("google-cloud-storage not available: %s", e)
        return None

    gcs_json = os.getenv("APIFY_GCS_SERVICE_JSON", "").strip()
    bucket_name = os.getenv("APIFY_GCS_BUCKET", "").strip()
    if not gcs_json or not bucket_name:
        log.info("GCS env vars missing; skipping GCS lookup.")
        return None

    # Build creds from the JSON string (no local file needed)
    try:
        info = json.loads(gcs_json)
        creds = service_account.Credentials.from_service_account_info(info)
        client = storage.Client(project=info.get("project_id"), credentials=creds)
        bucket = client.bucket(bucket_name)
    except Exception as e:
        log.error("GCS init failed: %s", e)
        return None

    # ----- search strategy -----
    video_id = _extract_youtube_id(url)
    candidates = [f"{base_filename}.mp3"]
    if video_id:
        candidates += [f"{video_id}.mp3", f"{video_id}_{base_filename}.mp3"]

    # 1) exact-name hits
    for name in candidates:
        blob = bucket.blob(name)
        try:
            if blob.exists(client):
                return _gcs_download(blob, output_dir, base_filename)
        except Exception:
            pass

    # 2) prefix search by base_filename
    for blob in client.list_blobs(bucket_name, prefix=base_filename):
        if blob.name.lower().endswith(".mp3"):
            return _gcs_download(blob, output_dir, base_filename)

    # 3) fallback: newest *.mp3 in bucket
    newest = None
    for blob in client.list_blobs(bucket_name):
        if blob.name.lower().endswith(".mp3"):
            if newest is None or (getattr(blob, "updated", None) and blob.updated > newest.updated):
                newest = blob
    if newest:
        return _gcs_download(newest, output_dir, base_filename)

    log.error("GCS lookup failed to find a matching audio object.")
    return None

def _gcs_download(blob, output_dir: Path, base_filename: str) -> str:
    out_path = output_dir / f"{base_filename}.mp3"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(out_path)
    log.info("Pulled from GCS: gs://%s/%s → %s", blob.bucket.name, blob.name, out_path)
    return str(out_path)



# ---------- Shared helpers ----------

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
        log.error(
            "Provided cookies file appears to be a Chrome/SQLite DB, not a Netscape cookies.txt export. "
            "Please export with a 'cookies.txt' extension/format."
        )
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


def _ensure_mp3(path_in: Path, path_out: Path) -> Optional[str]:
    """Convert any input audio/video to MP3 at path_out using ffmpeg; return output path or None."""
    try:
        if path_in.suffix.lower() == f".{Config.AUDIO_FORMAT.lower()}":
            return str(path_in)
        conv = subprocess.run(
            [FFMPEG_PATH, "-y", "-i", str(path_in), "-vn",
             "-acodec", "libmp3lame", "-q:a", "2", str(path_out)],
            check=True, capture_output=True, text=True, encoding="utf-8"
        )
        if conv.stdout:
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


# ---------- Apify fallbacks ----------

# --- Apify fallbacks ---
import time

# --- Small HTTP downloader used by Apify fallback -----------------------------
def _apify_http_download(src_url: str, dst_path: Path) -> bool:
    import requests
    try:
        with requests.get(src_url, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(262_144):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        log.error("Apify download failed: %s", e)
        return False

# --- Robust Apify fallback (run -> poll dataset; then run-sync fallback) ------
def _is_region_lock(msg: str) -> bool:
    if not msg:
        return False
    needles = [
        "not made this video available in your country",
        "video is not available in your country",
        "playback on other websites has been disabled",
        "Playback on other websites has been disabled",
    ]
    m = msg.lower()
    return any(n in m for n in needles)



def _pick_audio_url_from_items(items: list[dict]) -> Optional[str]:
    if not items:
        return None
    x = items[0] or {}
    dls = x.get("downloads") or []
    for d in dls:
        t = (d.get("type") or "").lower()
        f = (d.get("format") or "").lower()
        u = d.get("url")
        if u and ("audio" in t or f in ("mp3", "mpga", "m4a", "opus", "webm", "ogg", "wav")):
            return u
    for key in ("audioUrl", "audio", "audio_link", "url"):
        if x.get(key):
            return x[key]
    return None


def _apify_get(url: str, **kw):
    r = requests.get(url, timeout=kw.pop("timeout", 30))
    r.raise_for_status()
    return r

def _apify_post(url: str, payload: dict, **kw):
    r = requests.post(url, json=payload, timeout=kw.pop("timeout", 60))
    r.raise_for_status()
    return r

def _apify_fetch_dataset_items(dataset_id: str, token: str) -> list[dict]:
    # no token required for public dataset read, but include to be safe
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?clean=true&format=json&token={token}"
    try:
        r = _apify_get(url, timeout=60)
        return r.json() if r.content else []
    except Exception as e:
        log.debug("Apify dataset fetch failed: %s", e)
        return []

def _apify_fetch_output_record(kv_id: str, token: str) -> dict:
    url = f"https://api.apify.com/v2/key-value-stores/{kv_id}/records/OUTPUT?token={token}"
    try:
        r = _apify_get(url, timeout=60)
        # OUTPUT may be JSON or binary; assume JSON for this actor
        return r.json()
    except Exception as e:
        log.debug("Apify OUTPUT fetch failed: %s", e)
        return {}

def _apify_poll_run(actor: str, run_id: str, token: str,
                    poll_timeout_ms: int, poll_interval_ms: int) -> dict:
    deadline = time.time() + (poll_timeout_ms / 1000.0)
    # primary endpoint
    run_url_acts = f"https://api.apify.com/v2/acts/{actor}/runs/{run_id}?token={token}"
    # fallback endpoint used by some clients / older docs
    run_url_legacy = f"https://api.apify.com/v2/actor-runs/{run_id}?token={token}"

    last = {}
    while time.time() < deadline:
        try:
            r = _apify_get(run_url_acts, timeout=30)
            j = _apify_unwrap(r.json())                # NEW
            last = j
            status = (j.get("status") or "").upper()
            if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                return j
        except requests.HTTPError as e:
            # If the new endpoint 404s for any reason, try the legacy one
            try:
                r = _apify_get(run_url_legacy, timeout=30)
                j = _apify_unwrap(r.json())            # NEW
                last = j
                status = (j.get("status") or "").upper()
                if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                    return j
            except Exception as ee:
                log.debug("Apify legacy poll failed: %s", ee)

        time.sleep(max(0.5, poll_interval_ms / 1000.0))
    return last


# put this helper near your other utils (top of file is fine)
def _apify_unwrap(obj: dict) -> dict:
    """Apify sometimes returns {data: {...}}. Unwrap to the inner dict."""
    if isinstance(obj, dict) and isinstance(obj.get("data"), dict):
        return obj["data"]
    return obj


def _apify_download_audio(url: str, output_dir: Path, base_filename: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Run 'streamers/youtube-video-downloader' via API with the correct schema:
      - videos: [{ "url": <string> }]
      - preferredFormat: 'mp3'
      - useApifyProxy / proxyCountry
      - uploader selection via 'uploadTo' (e.g. 'gcs')
      - optional GCS: googleCloudServiceKey, googleCloudBucketName
    Then pull dataset/KV results and download the audio locally.
    """
    token = os.getenv("APIFY_TOKEN", "").strip()
    if not token:
        log.warning("APIFY_TOKEN not set; skipping Apify fallback.")
        return None

    # Build base payload (per actor input schema)
    payload: dict[str, Any] = {
        "videos": [{"url": url}],
        "preferredFormat": "mp3",
        "useApifyProxy": True,
        "proxyCountry": os.getenv("APIFY_PROXY_COUNTRY", "US"),
        "fileNameTemplate": base_filename,
    }

    # If GCS creds present, enable uploading; otherwise return links only
    gcs_key_raw = os.getenv("APIFY_GCS_SERVICE_JSON", "").strip()
    gcs_bucket  = os.getenv("APIFY_GCS_BUCKET", "").strip()
    if gcs_key_raw and gcs_bucket:
        payload["uploadTo"] = "gcs"
        # googleCloudServiceKey must be a STRING containing the JSON
        payload["googleCloudServiceKey"] = gcs_key_raw
        payload["googleCloudBucketName"] = gcs_bucket
    else:
        payload["uploadTo"] = "none"
        payload["returnOnlyInfo"] = True

    # Time controls (tweak via env)
    wait_for_finish = int(os.getenv("APIFY_WAIT_FOR_FINISH_SECS", "60"))
    poll_timeout_ms = int(os.getenv("APIFY_POLL_TIMEOUT_MS", "600000"))  # 10 min
    poll_interval_ms = int(os.getenv("APIFY_POLL_INTERVAL_MS", "3000"))

    # Start the actor run
    actor_slug = os.getenv("APIFY_ACTOR", "streamers~youtube-video-downloader")
    act_base = f"https://api.apify.com/v2/acts/{actor_slug}"

    try:
        start_url = f"{act_base}/runs?token={token}&waitForFinish={wait_for_finish}"
        log.info("Apify fallback: start run (payload keys=%s)", list(payload.keys()))
        r = requests.post(start_url, json=payload, timeout=90)
        if r.status_code >= 400:
            body = (r.text or "").strip()
            log.error("Apify start HTTP %s. Body: %s", r.status_code, body[:1000])
            return None
        start_json = r.json()
        run_obj = _apify_unwrap(start_json)                 # <-- FIX #1: unwrap
    except Exception as e:
        log.error("Apify start failed: %s", repr(e))
        return None

    status = (run_obj.get("status") or "").upper()
    run_id = run_obj.get("id")
    if not run_id:
        log.error("Apify start: missing run id in response: %s", start_json)
        return None

    # If not terminal, poll until done
    if status not in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
        polled = _apify_poll_run(actor_slug, run_id, token, poll_timeout_ms, poll_interval_ms)
        run_obj = _apify_unwrap(polled)                     # <-- FIX #3: unwrap after polling
        status = (run_obj.get("status") or "").upper()

    if status != "SUCCEEDED":
        msg = run_obj.get("statusMessage") or run_obj.get("errorMessage") or "Unknown error"
        log.error("Apify run FAILED (status=%s, id=%s, msg=%s)", status, run_id, msg)
        # Retry once in "info-only" mode if uploader was the issue
        if "uploader" in msg.lower() and payload.get("uploadTo") == "gcs":
            log.info("Retrying Apify with uploadTo='none' and returnOnlyInfo=True…")
            payload_retry = dict(payload)
            payload_retry["uploadTo"] = "none"
            payload_retry["returnOnlyInfo"] = True
            payload_retry.pop("googleCloudServiceKey", None)
            payload_retry.pop("googleCloudBucketName", None)
            try:
                rr = requests.post(f"{act_base}/runs?token={token}&waitForFinish={wait_for_finish}",
                                   json=payload_retry, timeout=90)
                if rr.status_code >= 400:
                    log.error("Apify retry start HTTP %s. Body: %s", rr.status_code, (rr.text or "")[:1000])
                    return None
                run_retry_json = rr.json()
                run_retry = _apify_unwrap(run_retry_json)   # <-- unwrap retry start
                if (run_retry.get("status") or "").upper() not in ("SUCCEEDED",):
                    polled_retry = _apify_poll_run(actor_slug, run_retry.get("id"), token, poll_timeout_ms, poll_interval_ms)
                    run_retry = _apify_unwrap(polled_retry) # <-- unwrap retry poll
                if (run_retry.get("status") or "").upper() != "SUCCEEDED":
                    log.error("Apify retry still failed: %s", run_retry)
                    return None
                run_obj = run_retry
            except Exception as e:
                log.error("Apify retry failed: %s", repr(e))
                return None
        else:
            return None

    # Read dataset & KV IDs from the (possibly unwrapped) run object
    ds_id = run_obj.get("defaultDatasetId")                 # <-- FIX #3 uses unwrapped run_obj
    kv_id = run_obj.get("defaultKeyValueStoreId")

    # Did we ask Apify to upload to GCS?
    gcs_key_raw = os.getenv("APIFY_GCS_SERVICE_JSON", "").strip()
    gcs_bucket  = os.getenv("APIFY_GCS_BUCKET", "").strip()
    asked_gcs   = bool(gcs_key_raw and gcs_bucket)  # because you set uploadTo="gcs" when both exist
    
    # 1) Try dataset/OUTPUT as you already do
    items = _apify_fetch_dataset_items(ds_id, token) if ds_id else []
    if not items and kv_id:
        out = _apify_fetch_output_record(kv_id, token)
        items = [out] if isinstance(out, dict) else (out or [])


    audio_url = _pick_audio_url_from_items(items)

    # 2) If no URL found but we uploaded to GCS, search the bucket directly
    if not audio_url and asked_gcs:
        log.info("Apify yielded no direct URL; trying GCS bucket lookup…")
        gcs_res = _gcs_find_and_download(
            bucket_name=gcs_bucket,
            service_json_str=gcs_key_raw,
            output_dir=output_dir,
            desired_basename=base_filename,
            url=url,
        )
        if gcs_res:
            # Done – we found and downloaded the mp3 from GCS
            return gcs_res
        else:
            log.error("GCS lookup failed to find a matching audio object.")


    # 3) If you DID get a direct URL from the Apify item, keep your current download logic
    if audio_url:
        out_path = output_dir / f"{base_filename}.mp3"
        try:
            with requests.get(audio_url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            if out_path.exists() and out_path.stat().st_size > 0:
                meta = {
                    "extractor": "apify",
                    "webpage_url": url,
                    "download_attempt": "apify_fallback",
                    "apify_run_id": run_obj.get("id"),
                    "dataset_id": ds_id,
                    "kv_store_id": kv_id,
                }
                log.info("Apify fallback success → %s", out_path)
                return str(out_path), meta
        except Exception as e:
            log.error("Apify: error downloading returned audio URL: %s", e)

    # 4) Nothing worked
    log.error("Apify: no audio URL found and GCS lookup failed.")
    return None


def _apify_ytdl_fallback(
    url: str,
    output_dir: Path,
    base_filename: str,
    audio_format: str,
    ffmpeg_path: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    token = os.getenv("APIFY_TOKEN", getattr(Config, "APIFY_TOKEN", "")).strip()
    if not token:
        log.warning("APIFY_TOKEN not configured; skipping Apify fallback.")
        return None

    timeout_ms = int(os.getenv("APIFY_TIMEOUT_MS", getattr(Config, "APIFY_TIMEOUT_MS", 180_000)))
    timeout_s = max(60, timeout_ms / 1000)

    endpoint = (
        "https://api.apify.com/v2/acts/streamers~youtube-video-downloader/"
        f"run-sync-get-dataset-items?token={token}&timeout={timeout_ms}"
    )

    payloads = [
        {"videos": [{"url": url}], "proxy": {"useApifyProxy": True, "apifyProxyCountry": "US"}},
        {"videoUrl": url, "proxy": {"useApifyProxy": True, "apifyProxyCountry": "US"}},
        {"videoUrls": [url], "proxy": {"useApifyProxy": True, "apifyProxyCountry": "US"}},
    ]

    for payload in payloads:
        try:
            log.info("Apify fallback (ytdl path): requesting actor (payload keys=%s)", list(payload.keys()))
            resp = requests.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=(30, timeout_s),
            )
            if resp.status_code != 200:
                log.error("Apify fallback HTTP %s: %s", resp.status_code, resp.text[:4000])
                continue

            try:
                items = resp.json()
            except Exception:
                text = resp.text.strip()
                first_line = (text.splitlines() or [""])[0]
                items = json.loads(first_line) if first_line else []

            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list) or not items:
                log.error("Apify fallback: unexpected response structure. Body: %s", resp.text[:1000])
                continue

            # items already loaded from dataset or OUTPUT
            item0 = items[0] if isinstance(items, list) and items else (items or {})
            downloads = item0.get("downloads") or []


            best_audio_url = None
            best_video_url = None

            # Prefer direct audio in the downloads array
            for d in downloads:
                t = (d.get("type") or "").lower()
                f = (d.get("format") or "").lower()
                u = d.get("url")
                if u and ("audio" in t or f in ("m4a", "webm", "opus", "mp3", "ogg")):
                    best_audio_url = u
                    break

            # Otherwise fall back to a video download we can convert
            if not best_audio_url:
                for d in downloads:
                    f = (d.get("format") or "").lower()
                    u = d.get("url")
                    if u and (f in ("mp4", "webm") or "video" in (d.get("type") or "").lower()):
                        best_video_url = u
                        break

            # Try common flat fields some actor versions emit
            if not best_audio_url:
                for key in ("fileUrl", "downloadUrl", "directUrl", "kvStoreRecordUrl", "audioUrl", "audio", "audio_link"):
                    val = item0.get(key)
                    if isinstance(val, str) and val:
                        best_audio_url = val
                        break

            # As a last resort, if the item only tells us the stored filename,
            # build a signed record URL from the run’s default KV store
            if not (best_audio_url or best_video_url):
                kv_key = item0.get("fileName") or item0.get("key") or item0.get("storageFileKey")
                if kv_key and kv_id:
                    from urllib.parse import quote
                    best_audio_url = (
                        f"https://api.apify.com/v2/key-value-stores/{kv_id}/records/{quote(kv_key, safe='')}"
                        + (f"?token={token}" if token else "")
                    )
            if not best_audio_url and not best_video_url:
                log.error("Apify fallback: no downloadable URLs in actor output / KV.")
                continue

            tmp_in = output_dir / f"{base_filename}.apify.tmp"
            tmp_out = output_dir / f"{base_filename}.{audio_format}"

            def _http_download(src_url: str, dst_path: Path) -> bool:
                try:
                    with requests.get(src_url, stream=True, timeout=(15, 120)) as r:
                        r.raise_for_status()
                        with open(dst_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=262144):
                                if chunk:
                                    f.write(chunk)
                    return True
                except Exception as e:
                    log.error("Apify fallback: download failed: %s", e)
                    return False

            if best_audio_url:
                log.info("Apify fallback: downloading direct audio…")
                if not _http_download(best_audio_url, tmp_out):
                    continue
                if tmp_out.suffix.lower() != f".{audio_format}":
                    temp_src = tmp_in.with_suffix(Path(best_audio_url).suffix or ".bin")
                    try:
                        tmp_out.rename(temp_src)
                    except Exception:
                        if not _http_download(best_audio_url, temp_src):
                            continue
                    final = _ensure_mp3(temp_src, tmp_out)
                    try:
                        if temp_src.exists():
                            temp_src.unlink()
                    except Exception:
                        pass
                    if not final:
                        continue
                audio_path = str(tmp_out)
            else:
                log.info("Apify fallback: downloading video then extracting audio…")
                if not _http_download(best_video_url, tmp_in):
                    continue
                final = _ensure_mp3(tmp_in, tmp_out)
                try:
                    if tmp_in.exists():
                        tmp_in.unlink()
                except Exception:
                    pass
                if not final:
                    continue
                audio_path = final

            meta = {
                "title": item0.get("title") or "Unknown Title",
                "uploader": item0.get("uploader") or item0.get("channel") or "Unknown Uploader",
                "duration": item0.get("duration"),
                "webpage_url": url,
                "extractor": "apify-streamers/youtube-video-downloader",
                "download_attempt": "apify_fallback",
                "apify_run_id": item0.get("id") or item0.get("runId"),
                "thumbnail": item0.get("thumbnail"),
                "view_count": item0.get("viewCount"),
            }
            log.info("Apify fallback succeeded → %s", audio_path)
            return (audio_path, meta)

        except requests.RequestException as e:
            log.error("Apify fallback network error: %s", e)
        except Exception as e:
            log.error("Apify fallback unexpected error: %s", e, exc_info=True)

    return None




