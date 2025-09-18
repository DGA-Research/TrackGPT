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

APIFY_ACTOR = os.getenv("APIFY_ACTOR", "streamers~youtube-video-downloader")

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



def _pick_audio_url_from_items(items: list[dict]) -> Optional[str]:
    """Scan common shapes returned by the actor for a direct audio URL."""
    if not items:
        return None
    x = items[0] or {}

    # new style: explicit 'downloads' list
    dls = x.get("downloads") or []
    for d in dls:
        t = (d.get("type") or "").lower()
        f = (d.get("format") or "").lower()
        u = d.get("url")
        if u and ("audio" in t or f in ("mp3", "m4a", "opus", "webm", "ogg")):
            return u

    # flat fields some builds expose
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
    """Poll correct run endpoint until finished."""
    deadline = time.time() + (poll_timeout_ms / 1000.0)
    # per docs you can poll either /actor-runs/{id} or /acts/{actor}/runs/{id}
    run_url = f"https://api.apify.com/v2/acts/{actor}/runs/{run_id}?token={token}"
    last = {}
    while time.time() < deadline:
        try:
            r = _apify_get(run_url, timeout=30)
            last = r.json()
            status = (last.get("status") or "").upper()
            if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                return last
        except requests.HTTPError as e:
            log.debug("Apify run poll error: %s", e)
        time.sleep(max(0.5, poll_interval_ms / 1000.0))
    return last  # may still be RUNNING

def _apify_download_audio(url: str, output_dir: Path, base_filename: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Run 'streamers/youtube-video-downloader' via API with proper schema:
      - videos: [url]
      - preferredFormat: 'mp3'
      - useApifyProxy / proxyCountry
      - optional GCS: googleCloudServiceKey, googleCloudBucketName
    Then pull dataset/KV results and download the audio locally.
    """
    token = os.getenv("APIFY_TOKEN", "").strip()
    if not token:
        log.warning("APIFY_TOKEN not set; skipping Apify fallback.")
        return None

    # Inputs per actor schema (videos, preferredFormat, etc.).  🚩 Required field name is *videos*.
    # https://apify.com/streamers/youtube-video-downloader/input-schema
    payload: dict[str, Any] = {
        "videos": [url],
        "preferredFormat": "mp3",
        "useApifyProxy": True,
        "proxyCountry": os.getenv("APIFY_PROXY_COUNTRY", "US"),
        # Give the output a stable base filename (actor adds extension)
        "fileNameTemplate": base_filename,
    }

    # If GCS creds are provided, enable direct upload
    gcs_key = os.getenv("APIFY_GCS_SERVICE_JSON", "").strip()
    gcs_bucket = os.getenv("APIFY_GCS_BUCKET", "").strip()
    gcs_enabled = bool(gcs_key and gcs_bucket)
    if gcs_enabled:
        payload["googleCloudServiceKey"] = gcs_key  # raw JSON string
        payload["googleCloudBucketName"] = gcs_bucket
    else:
        # If we don't provide any uploader credentials, avoid the actor error by returning only info
        payload["returnOnlyInfo"] = True  # matches "Return only info" in schema

    # Time controls
    wait_for_finish = min(int(os.getenv("APIFY_WAIT_FOR_FINISH_SECS", "60")), 60)
    poll_timeout_ms = int(os.getenv("APIFY_POLL_TIMEOUT_MS", "300000"))
    poll_interval_ms = int(os.getenv("APIFY_POLL_INTERVAL_MS", "3000"))

    act_base = f"https://api.apify.com/v2/acts/{APIFY_ACTOR}"
    # 1) Try a sync-ish start (server waits up to 60s)
    try:
        log.info("Apify fallback: start run (payload keys=%s)", list(payload.keys()))
        start_url = f"{act_base}/runs?token={token}&waitForFinish={wait_for_finish}"
        run_obj = _apify_post(start_url, payload, timeout=90).json()
    except requests.HTTPError as e:
        log.error("Apify start HTTP %s. Body: %s", e.response.status_code if e.response else "?", e.response.text if e.response else "")
        return None
    except Exception as e:
        log.error("Apify start failed: %s", e)
        return None

    status = (run_obj.get("status") or "").upper()
    run_id = run_obj.get("id")
    if not run_id:
        log.error("Apify start: missing run id in response.")
        return None

    # 2) If still RUNNING, poll until finished
    if status not in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
        run_obj = _apify_poll_run(APIFY_ACTOR, run_id, token, poll_timeout_ms, poll_interval_ms)
        status = (run_obj.get("status") or "").upper()

    if status != "SUCCEEDED":
        msg = run_obj.get("statusMessage") or run_obj.get("errorMessage") or "Unknown error"
        log.error("Apify run FAILED (status=%s, id=%s, msg=%s)", status, run_id, msg)
        # If we failed because uploader wasn't configured and we didn't set returnOnlyInfo,
        # retry once with returnOnlyInfo=True to at least get downloadable links.
        if gcs_enabled and "uploader" in (msg or "").lower():
            log.info("Retrying Apify with returnOnlyInfo=True…")
            payload2 = dict(payload)
            payload2.pop("googleCloudServiceKey", None)
            payload2.pop("googleCloudBucketName", None)
            payload2["returnOnlyInfo"] = True
            try:
                run_obj = _apify_post(f"{act_base}/runs?token={token}&waitForFinish={wait_for_finish}", payload2, timeout=90).json()
                if (run_obj.get("status") or "").upper() not in ("SUCCEEDED",):
                    run_obj = _apify_poll_run(APIFY_ACTOR, run_obj.get("id"), token, poll_timeout_ms, poll_interval_ms)
            except Exception as e:
                log.error("Apify retry failed: %s", e)
                return None
            if (run_obj.get("status") or "").upper() != "SUCCEEDED":
                log.error("Apify retry still failed.")
                return None
        else:
            return None

    ds_id = run_obj.get("defaultDatasetId")
    kv_id = run_obj.get("defaultKeyValueStoreId")

    items = _apify_fetch_dataset_items(ds_id, token) if ds_id else []
    if not items and kv_id:
        out = _apify_fetch_output_record(kv_id, token)
        # OUTPUT may be a single object; normalize to list for the same picker
        items = [out] if isinstance(out, dict) else (out or [])

    audio_url = _pick_audio_url_from_items(items)
    if not audio_url:
        log.error("Apify: no audio URL found in dataset/OUTPUT.")
        return None

    # Download the audio to local mp3 path
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
                "apify_run_id": run_id,
                "dataset_id": ds_id,
                "kv_store_id": kv_id,
            }
            log.info("Apify fallback success → %s", out_path)
            return str(out_path), meta
    except Exception as e:
        log.error("Apify: error downloading returned audio URL: %s", e)

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












