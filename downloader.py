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

log = logging.getLogger(__name__)

# ---- Optional: use Apify Proxy automatically if available ----
apify_pw = os.getenv("APIFY_PROXY_PASSWORD", "")
apify_country = os.getenv("APIFY_PROXY_COUNTRY", "US")
proxy_url = os.getenv("YTDLP_PROXY_URL", "").strip()

if apify_pw and not proxy_url:
    # Apify Proxy format: http://auto:<PASSWORD>@proxy.apify.com:8000/?country=US
    proxy_url = f"http://DGA_Scrapes:{apify_pw}@proxy.apify.com:8000/?country={apify_country}"
    log.info("Using Apify Proxy for yt-dlp with country=%s", apify_country)

# When building the CLI args:
enrich = []
# ... UA / headers here ...
if proxy_url:
    enrich += ["--proxy", proxy_url]


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


def _apify_download_audio(url: str, output_dir: Path, base_filename: str) -> tuple[str, dict] | None:
    """
    Use Apify 'streamers/youtube-video-downloader' with proxy + GCS uploader.
    Returns (audio_path, metadata) or None.
    """
    import time
    import json as _json
    import requests

    token = os.getenv("APIFY_TOKEN", getattr(Config, "APIFY_TOKEN", "")).strip()
    if not token:
        log.warning("APIFY_TOKEN not set; skipping Apify fallback.")
        return None

    act_id = os.getenv("APIFY_ACT_ID", getattr(Config, "APIFY_ACT_ID", "streamers~youtube-video-downloader")).strip()

    # Proxy (by country)
    proxy_country = os.getenv("APIFY_PROXY_COUNTRY", getattr(Config, "APIFY_PROXY_COUNTRY", "US")).strip()
    proxy_cfg = {"useApifyProxy": True, "apifyProxyCountry": proxy_country} if proxy_country else {"useApifyProxy": True}

    # Uploader → GCS
    upload_to = os.getenv("APIFY_UPLOAD_TO", getattr(Config, "APIFY_UPLOAD_TO", "gcs")).strip().lower()
    gcs_bucket = os.getenv("APIFY_GCS_BUCKET", getattr(Config, "APIFY_GCS_BUCKET", "")).strip()
    gcs_sa_json = os.getenv("APIFY_GCS_SERVICE_JSON", getattr(Config, "APIFY_GCS_SERVICE_JSON", "")).strip()

    payload: dict[str, Any] = {
        # IMPORTANT: this actor expects 'videos' (not videoUrl/videoUrls)
        "videos": [{"url": url}],
        "convertToAudio": True,
        "audioFormat": "mp3",
        "proxy": proxy_cfg,
    }

    # If GCS creds are present, enable uploader
    if upload_to == "gcs" and gcs_bucket and gcs_sa_json:
        payload["uploadTo"] = "gcs"
        payload["uploader"] = "gcs"            # some builds require this, harmless otherwise
        payload["gcs"] = {
            "bucket": gcs_bucket,
            "serviceAccountJson": gcs_sa_json, # pass as string; actor parses JSON internally
            # optional prefix inside the bucket
            "path": f"apify/{base_filename}/"
        }

    # Try 1: run-sync-get-dataset-items (returns items directly)
    run_sync_items = f"https://api.apify.com/v2/acts/{act_id}/run-sync-get-dataset-items?token={token}"
    try:
        log.info("Apify fallback: run-sync (payload keys=%s)", list(payload.keys()))
        r = requests.post(run_sync_items, json=payload, timeout=(30, 240))
        if r.status_code == 200:
            items = r.json() if r.content else []
            if isinstance(items, dict):
                items = [items]
            if items:
                item = items[0]
                audio_url = (
                    (item.get("downloads") or {}).get("audio")
                    or item.get("audio")
                    or item.get("audioUrl")
                    or item.get("url")
                )
                if audio_url:
                    # Download to local file
                    out_path = output_dir / f"{base_filename}.mp3"
                    with requests.get(audio_url, stream=True, timeout=(15, 180)) as rr:
                        rr.raise_for_status()
                        with open(out_path, "wb") as f:
                            for chunk in rr.iter_content(1024 * 256):
                                if chunk:
                                    f.write(chunk)
                    if out_path.exists() and out_path.stat().st_size > 0:
                        meta = {"extractor": "apify", "webpage_url": url, "download_attempt": "apify_fallback"}
                        log.info("Apify fallback success → %s", out_path)
                        return str(out_path), meta
                else:
                    log.error("Apify: dataset returned no direct audio URL; item keys: %s", list(item.keys()))
            else:
                log.error("Apify: dataset returned no items.")
        else:
            log.error("Apify run-sync HTTP %s. Body: %s", r.status_code, r.text[:2000])
    except Exception as e:
        log.error("Apify run-sync failed: %s", e)

    # Try 2: start run, then poll dataset for items
    start_url = f"https://api.apify.com/v2/acts/{act_id}/runs?token={token}"
    try:
        log.info("Apify fallback: start run (payload keys=%s)", list(payload.keys()))
        r = requests.post(start_url, json=payload, timeout=(30, 60))
        if r.status_code not in (200, 201):
            log.error("Apify start HTTP %s. Body: %s", r.status_code, r.text[:2000])
            return None
        run = r.json()
        ds_id = run.get("data", {}).get("defaultDatasetId") or run.get("defaultDatasetId")
        run_id = run.get("data", {}).get("id") or run.get("id")

        if not ds_id:
            log.error("Apify run started but defaultDatasetId missing. Run: %s", _json.dumps(run)[:800])
            return None

        # poll dataset
        items_url = f"https://api.apify.com/v2/datasets/{ds_id}/items?limit=1"
        for _ in range(30):  # ~30 * 2s = 60s max
            time.sleep(2)
            rr = requests.get(items_url, timeout=20)
            if rr.status_code != 200:
                continue
            items = rr.json() if rr.content else []
            if isinstance(items, dict):
                items = [items]
            if items:
                item = items[0]
                audio_url = (
                    (item.get("downloads") or {}).get("audio")
                    or item.get("audio")
                    or item.get("audioUrl")
                    or item.get("url")
                )
                if audio_url:
                    out_path = output_dir / f"{base_filename}.mp3"
                    with requests.get(audio_url, stream=True, timeout=(15, 180)) as ar:
                        ar.raise_for_status()
                        with open(out_path, "wb") as f:
                            for chunk in ar.iter_content(1024 * 256):
                                if chunk:
                                    f.write(chunk)
                    if out_path.exists() and out_path.stat().st_size > 0:
                        meta = {
                            "extractor": "apify",
                            "webpage_url": url,
                            "download_attempt": "apify_fallback",
                            "apify_run_id": run_id,
                        }
                        log.info("Apify fallback success (polled) → %s", out_path)
                        return str(out_path), meta
                else:
                    # If you want, inspect for GCS-only outputs:
                    # e.g. item.get("gcsUrl") or item.get("gcsPath")
                    log.error("Apify: item present but no direct audio URL; keys: %s", list(item.keys()))
                    return None
        log.error("Apify: dataset never produced items in time (ds_id=%s).", ds_id)
    except Exception as e:
        log.error("Apify start/poll failed: %s", e)

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

            item0 = items[0]
            downloads = item0.get("downloads") or []

            best_audio_url = None
            best_video_url = None

            for d in downloads:
                t = (d.get("type") or "").lower()
                f = (d.get("format") or "").lower()
                u = d.get("url")
                if u and ("audio" in t or f in ("m4a", "webm", "opus", "mp3", "ogg")):
                    best_audio_url = u
                    break

            if not best_audio_url:
                for d in downloads:
                    f = (d.get("format") or "").lower()
                    u = d.get("url")
                    if u and (f in ("mp4", "webm") or "video" in (d.get("type") or "").lower()):
                        best_video_url = u
                        break

            if not best_audio_url:
                for key in ("audioUrl", "audio", "audio_link"):
                    if item0.get(key):
                        best_audio_url = item0[key]
                        break

            if not best_audio_url and not best_video_url:
                if item0.get("url"):
                    best_video_url = item0["url"]

            if not best_audio_url and not best_video_url:
                log.error("Apify fallback: no downloadable URLs in actor output.")
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


# ---------- Core function ----------

def download_audio(url: str, output_dir: Path, base_filename: str, type_input) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Downloads audio from a given URL using yt-dlp with resilient fallbacks.
    Immediately triggers Apify fallback on first region-lock error.
    """
    import base64

    # Sanity: yt-dlp + ffmpeg present?
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

    # --- Cookies (file, b64, browser) → temp file (writable) ---
    temp_paths_to_cleanup: list[str] = []
    temp_cookies_file: Optional[str] = None

    orig_cookies_file = os.getenv("YTDLP_COOKIES_FILE", getattr(Config, "YTDLP_COOKIES_FILE", "")).strip()
    cookies_b64 = os.getenv("YTDLP_COOKIES_B64", getattr(Config, "YTDLP_COOKIES_B64", "") if hasattr(Config, "YTDLP_COOKIES_B64") else "").strip()
    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", getattr(Config, "YTDLP_COOKIES_FROM_BROWSER", "") if hasattr(Config, "YTDLP_COOKIES_FROM_BROWSER") else "").strip()
    user_agent = os.getenv("YTDLP_USER_AGENT", getattr(Config, "YTDLP_USER_AGENT", "")).strip()

    # If B64 is provided, materialize to temp and prefer it
    if cookies_b64:
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".cookies.txt")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(cookies_b64))
            os.chmod(tmp_path, 0o600)
            temp_paths_to_cleanup.append(tmp_path)
            orig_cookies_file = tmp_path
            log.info("Decoded cookies from secrets into: %s", tmp_path)
        except Exception as e:
            log.warning("Failed to decode YTDLP_COOKIES_B64: %s", e)

    # If cookies path present, copy to temp (and normalize encoding)
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
            log.warning("Could not create temp cookies file from '%s': %s. Continuing without cookies.", orig_cookies_file, e)
            temp_cookies_file = None

    def _cleanup_temp_cookies():
        for p in temp_paths_to_cleanup:
            try:
                os.remove(p)
                log.info("Removed temp cookies file: %s", p)
            except Exception:
                pass

    # --- Metadata extraction (best-effort; never crashes the flow) ---
    ydl_opts: Dict[str, Any] = {'quiet': True, 'no_warnings': True, 'extract_flat': False}
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
    except Exception as e:
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
            'type_input': type_input,
        }

    # --- Build base yt-dlp command once ---
    enrich: list[str] = []
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

    # Track if we’ve already tried Apify (to avoid spamming)
    apify_tried = False
    last_err: Optional[Exception] = None

    try:
        for label, cmd in attempts:
            # Prepare defaults so we never reference unbound locals
            cp = None
            try:
                log.debug("[yt-dlp] Attempt '%s': %s", label, " ".join(cmd))
                cp = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
                if cp.stdout:
                    log.info("yt-dlp stdout:\n%s", cp.stdout)
                if cp.stderr:
                    log.debug("yt-dlp stderr:\n%s", cp.stderr)

                # Success case 1: expected mp3 exists
                if final_audio_path.exists():
                    log.info("Success (%s) → %s", label, final_audio_path)
                    _cleanup_temp_cookies()
                    return (str(final_audio_path), {**metadata, "download_attempt": label})

                # Success case 2: any audio; convert if needed
                for cand in sorted(output_dir.glob(f"{base_filename}.*"),
                                   key=lambda p: p.stat().st_mtime, reverse=True):
                    if cand.suffix.lower() in [".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav"]:
                        if cand.suffix.lower() == ".mp3":
                            log.info("Success (%s) → %s", label, cand)
                            _cleanup_temp_cookies()
                            return (str(cand), {**metadata, "download_attempt": label})
                        # convert
                        out = _ensure_mp3(cand, final_audio_path)
                        if out:
                            log.info("Success (%s) + convert → %s", label, out)
                            _cleanup_temp_cookies()
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

                # EARLY APIFY: if first time we detect region-lock, jump to Apify immediately
                if not apify_tried and _is_region_lock(stderr):
                    apify_tried = True
                    log.info("Detected region lock on attempt '%s'. Trying Apify fallback immediately…", label)
                    ap = _apify_download_audio(url, output_dir, base_filename)
                    if ap:
                        _cleanup_temp_cookies()
                        # merge any basic metadata we had
                        ap_path, ap_meta = ap
                        return (ap_path, {**metadata, **ap_meta})
                    else:
                        log.error("Apify fallback failed; continuing ladder.")

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
        _cleanup_temp_cookies()

    # If we got here, yt-dlp ladder failed and either Apify never triggered or failed
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








