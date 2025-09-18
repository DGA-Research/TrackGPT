"""
Clean downloader module: yt-dlp with Apify + GCS fallbacks.

Flow:
  1) Try yt-dlp (several extractor strategies).
  2) On region/403/bot blocks -> run Apify actor and fetch direct URL.
  3) If actor uploaded to GCS (no direct URL), find & pull object from bucket,
     handling funky extensions like ".mp3.mpga" and converting to .mp3.

ENV (examples):
  APIFY_TOKEN
  APIFY_ACTOR=streamers~youtube-video-downloader
  APIFY_PROXY_COUNTRY=US
  APIFY_GCS_SERVICE_JSON=<service account JSON string>
  APIFY_GCS_BUCKET=<bucket name>

Optional yt-dlp tweaks (ENV or Config attr):
  YTDLP_USER_AGENT, YTDLP_COOKIES_FILE, YTDLP_COOKIES_B64,
  YTDLP_COOKIES_FROM_BROWSER, YTDLP_GEO_BYPASS, YTDLP_GEO_COUNTRY, YTDLP_RETRIES,
  YTDLP_PROXY_URL
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

# Optional user config module
try:
    from config import Config as _UserConfig  # type: ignore
except Exception:
    class _UserConfig:  # minimal shim
        AUDIO_FORMAT = "mp3"

# ------------------------------ config helpers ------------------------------ #

def _cfg(name: str, default: Any = None):
    """ENV first, then Config attr, else default."""
    v = os.getenv(name)
    if v is not None and v != "":
        return v
    return getattr(_UserConfig, name, default)

AUDIO_FORMAT = str(_cfg("AUDIO_FORMAT", "mp3"))

# Apify
APIFY_ACTOR = _cfg("APIFY_ACTOR", "streamers~youtube-video-downloader")
APIFY_TOKEN = (_cfg("APIFY_TOKEN", "") or "").strip()
APIFY_PROXY_COUNTRY = str(_cfg("APIFY_PROXY_COUNTRY", "US"))
APIFY_WAIT_FOR_FINISH_SECS = int(float(_cfg("APIFY_WAIT_FOR_FINISH_SECS", "60")))
APIFY_POLL_TIMEOUT_MS = int(float(_cfg("APIFY_POLL_TIMEOUT_MS", "600000")))
APIFY_POLL_INTERVAL_MS = int(float(_cfg("APIFY_POLL_INTERVAL_MS", "3000")))

# GCS
APIFY_GCS_SERVICE_JSON = (_cfg("APIFY_GCS_SERVICE_JSON", "") or "").strip()
APIFY_GCS_BUCKET = (_cfg("APIFY_GCS_BUCKET", "") or "").strip()

# yt-dlp
YTDLP_RETRIES = int(float(_cfg("YTDLP_RETRIES", "3")))
YTDLP_USER_AGENT = (_cfg("YTDLP_USER_AGENT", "") or "").strip()

# ------------------------------ exec discovery ------------------------------ #

def _find_yt_dlp() -> Optional[str]:
    try:
        import yt_dlp  # type: ignore
        try:
            return yt_dlp.utils.exe_path()
        except Exception:
            pass
    except Exception:
        pass
    return shutil.which("yt-dlp")

def _find_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")

YT_DLP_PATH = _find_yt_dlp()
FFMPEG_PATH = _find_ffmpeg()

# ------------------------------ utilities ----------------------------------- #

def _extract_youtube_id(url: str) -> Optional[str]:
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None

def _is_region_lock(msg: str) -> bool:
    if not msg:
        return False
    m = msg.lower()
    needles = [
        "not made this video available in your country",
        "video is not available in your country",
        "playback on other websites has been disabled",
        "sign in to confirm you’re not a bot",
        "please sign in",
        "http error 403",
        "http error 429",
    ]
    return any(n in m for n in needles)

def _ensure_mp3(path_in: Path, path_out: Path) -> Optional[str]:
    """Convert any input media to MP3 using ffmpeg."""
    try:
        if path_in.suffix.lower() == ".mp3":
            return str(path_in)
        if not FFMPEG_PATH:
            log.error("ffmpeg not found; cannot convert %s → %s", path_in, path_out)
            return None
        cmd = [FFMPEG_PATH, "-y", "-i", str(path_in), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(path_out)]
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if cp.stdout:
            log.debug("ffmpeg stdout:\n%s", cp.stdout)
        if cp.stderr:
            log.debug("ffmpeg stderr:\n%s", cp.stderr)
        return str(path_out) if path_out.exists() else None
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg failed: %s", e.stderr or e)
        return None
    except Exception as e:
        log.error("ffmpeg error: %s", e, exc_info=True)
        return None

def _http_download(src_url: str, dst_path: Path, timeout=(15, 120)) -> bool:
    try:
        with requests.get(src_url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(262_144):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        log.error("HTTP download failed: %s", e)
        return False

def _ensure_utf8_netscape(cookie_path: str, temp_list: List[str]) -> str:
    """Ensure cookie file is UTF-8 Netscape text; re-save if needed."""
    try:
        data = Path(cookie_path).read_bytes()
    except Exception as e:
        log.warning("Cannot read cookies file '%s': %s", cookie_path, e)
        return cookie_path
    if data.startswith(b"SQLite format 3"):
        log.error("Cookies file looks like a SQLite DB; export to 'cookies.txt' (Netscape) first.")
        return cookie_path
    try:
        data.decode("utf-8")
        return cookie_path
    except UnicodeDecodeError:
        pass
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    text = text.replace("\r\n", "\n")
    fd, newp = tempfile.mkstemp(suffix=".cookies.utf8.txt")
    os.close(fd)
    Path(newp).write_text(text, encoding="utf-8")
    os.chmod(newp, 0o600)
    temp_list.append(newp)
    log.info("Re-encoded cookies to UTF-8 at: %s", newp)
    return newp

# ------------------------------ Apify helpers -------------------------------- #

def _apify_unwrap(obj: dict) -> dict:
    """Apify often returns {'data': {...}}; return the inner dict if present."""
    if isinstance(obj, dict) and isinstance(obj.get("data"), dict):
        return obj["data"]
    return obj

def _apify_get(url: str, timeout: int = 30) -> requests.Response:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r

def _apify_post(url: str, payload: dict, timeout: int = 90) -> requests.Response:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r

def _apify_poll_run(actor: str, run_id: str, token: str,
                    poll_timeout_ms: int, poll_interval_ms: int) -> dict:
    deadline = time.time() + (poll_timeout_ms / 1000.0)
    url_new = f"https://api.apify.com/v2/acts/{actor}/runs/{run_id}?token={token}"
    url_old = f"https://api.apify.com/v2/actor-runs/{run_id}?token={token}"
    last = {}
    while time.time() < deadline:
        try:
            j = _apify_unwrap(_apify_get(url_new, timeout=30).json())
        except requests.HTTPError:
            try:
                j = _apify_unwrap(_apify_get(url_old, timeout=30).json())
            except Exception:
                j = {}
        if j:
            last = j
            st = (j.get("status") or "").upper()
            if st in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                return j
        time.sleep(max(0.5, poll_interval_ms / 1000.0))
    return last

def _apify_fetch_dataset_items(dataset_id: str, token: str) -> List[dict]:
    if not dataset_id:
        return []
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?clean=true&format=json&token={token}"
    try:
        return _apify_get(url, timeout=60).json()
    except Exception as e:
        log.debug("Apify dataset fetch failed: %s", e)
        return []

def _apify_fetch_output_record(kv_id: str, token: str) -> dict:
    if not kv_id:
        return {}
    url = f"https://api.apify.com/v2/key-value-stores/{kv_id}/records/OUTPUT?token={token}"
    try:
        return _apify_get(url, timeout=60).json()
    except Exception as e:
        log.debug("Apify OUTPUT fetch failed: %s", e)
        return {}

def _pick_audio_url_from_items(items: List[dict]) -> Optional[str]:
    if not items:
        return None
    x = items[0] or {}

    # Newer actor builds expose a downloads[] array
    for d in (x.get("downloads") or []):
        t = (d.get("type") or "").lower()
        f = (d.get("format") or "").lower()
        u = d.get("url")
        if u and ("audio" in t or f in ("mp3", "mpga", "m4a", "opus", "webm", "ogg", "wav")):
            return u

    # Flat fields, depending on build
    for key in ("audioUrl", "audio", "audio_link", "url"):
        u = x.get(key)
        if isinstance(u, str) and u:
            return u

    return None

def _apify_download_audio(url: str, output_dir: Path, base_filename: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not APIFY_TOKEN:
        log.warning("APIFY_TOKEN not set; skipping Apify fallback.")
        return None

    payload: Dict[str, Any] = {
        "videos": [{"url": url}],
        "preferredFormat": "mp3",
        "useApifyProxy": True,
        "proxyCountry": APIFY_PROXY_COUNTRY,
        "fileNameTemplate": base_filename,
    }

    asked_gcs = bool(APIFY_GCS_SERVICE_JSON and APIFY_GCS_BUCKET)
    if asked_gcs:
        payload["uploadTo"] = "gcs"
        payload["googleCloudServiceKey"] = APIFY_GCS_SERVICE_JSON  # STRING JSON
        payload["googleCloudBucketName"] = APIFY_GCS_BUCKET
    else:
        payload["uploadTo"] = "none"
        payload["returnOnlyInfo"] = True

    start_url = f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/runs?token={APIFY_TOKEN}&waitForFinish={APIFY_WAIT_FOR_FINISH_SECS}"
    try:
        log.info("Apify fallback: start run (payload keys=%s)", list(payload.keys()))
        start_json = _apify_post(start_url, payload, timeout=90).json()
        run_obj = _apify_unwrap(start_json)
    except Exception as e:
        log.error("Apify start failed: %s", e)
        return None

    run_id = run_obj.get("id")
    status = (run_obj.get("status") or "").upper()
    if not run_id:
        log.error("Apify start: missing run id in response: %s", start_json)
        return None

    if status not in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
        polled = _apify_poll_run(APIFY_ACTOR, run_id, APIFY_TOKEN, APIFY_POLL_TIMEOUT_MS, APIFY_POLL_INTERVAL_MS)
        run_obj = _apify_unwrap(polled)
        status = (run_obj.get("status") or "").upper()

    if status != "SUCCEEDED":
        msg = run_obj.get("statusMessage") or run_obj.get("errorMessage") or "Unknown error"
        log.error("Apify run FAILED (status=%s, id=%s, msg=%s)", status, run_id, msg)
        # Smart retry: if uploader config caused it, try info-only
        if "uploader" in msg.lower() and asked_gcs:
            retry = dict(payload)
            retry["uploadTo"] = "none"
            retry["returnOnlyInfo"] = True
            retry.pop("googleCloudServiceKey", None)
            retry.pop("googleCloudBucketName", None)
            try:
                start_json = _apify_post(start_url, retry, timeout=90).json()
                run_retry = _apify_unwrap(start_json)
                if (run_retry.get("status") or "").upper() not in ("SUCCEEDED",):
                    polled = _apify_poll_run(APIFY_ACTOR, run_retry.get("id"), APIFY_TOKEN,
                                             APIFY_POLL_TIMEOUT_MS, APIFY_POLL_INTERVAL_MS)
                    run_retry = _apify_unwrap(polled)
                if (run_retry.get("status") or "").upper() != "SUCCEEDED":
                    log.error("Apify retry still failed: %s", run_retry)
                    return None
                run_obj = run_retry
            except Exception as e:
                log.error("Apify retry failed: %s", e)
                return None
        else:
            return None

    ds_id = run_obj.get("defaultDatasetId")
    kv_id = run_obj.get("defaultKeyValueStoreId")

    # 1) Try to get a direct URL from dataset/OUTPUT
    items = _apify_fetch_dataset_items(ds_id, APIFY_TOKEN) if ds_id else []
    if not items and kv_id:
        out = _apify_fetch_output_record(kv_id, APIFY_TOKEN)
        if isinstance(out, dict):
            items = [out]
        elif isinstance(out, list):
            items = out

    audio_url = _pick_audio_url_from_items(items)
    if audio_url:
        out_path = output_dir / f"{base_filename}.mp3"
        if _http_download(audio_url, out_path, timeout=(30, 300)):
            return str(out_path), {
                "extractor": "apify",
                "webpage_url": url,
                "download_attempt": "apify_fallback",
                "apify_run_id": run_id,
                "dataset_id": ds_id,
                "kv_store_id": kv_id,
            }

    # 2) If we asked for GCS upload, search the bucket directly
    if asked_gcs:
        log.info("Apify yielded no direct URL; trying GCS bucket lookup…")
        gcs_result = _gcs_find_and_download(
            bucket_name=APIFY_GCS_BUCKET,
            service_json_str=APIFY_GCS_SERVICE_JSON,
            output_dir=output_dir,
            desired_basename=base_filename,
            url=url,
        )
        if gcs_result:
            return gcs_result

    log.error("Apify: no audio URL found and GCS lookup failed.")
    return None

# ------------------------------ GCS fallback --------------------------------- #

def _gcs_find_and_download(
    bucket_name: str,
    service_json_str: str,
    output_dir: Path,
    desired_basename: str,
    url: str | None = None,
    max_age_minutes: int = 120,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Find actor output in GCS (accept .mpga etc.), prefer objects with videoId."""
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

    AUDIO_EXTS = (".mp3", ".mpga", ".m4a", ".webm", ".opus", ".ogg", ".wav")
    def _is_audio(name: str) -> bool:
        n = name.lower()
        return n.endswith(AUDIO_EXTS) or n.endswith(".mp3.mpga")

    vid = _extract_youtube_id(url or "")
    candidates: List[Any] = []

    # exact desired basename + common extensions
    for ext in AUDIO_EXTS + (".mp3.mpga",):
        try:
            b = bucket.blob(f"{desired_basename}{ext}")
            if b.exists(client):
                candidates.append(b)
        except Exception:
            pass

    # prefix with desired basename
    try:
        for b in client.list_blobs(bucket_name, prefix=desired_basename):
            if _is_audio(b.name):
                candidates.append(b)
    except Exception:
        pass

    # prefer videoId in name
    if vid:
        try:
            for b in client.list_blobs(bucket_name, prefix=vid):
                if _is_audio(b.name):
                    candidates.append(b)
        except Exception:
            pass
        try:
            for b in client.list_blobs(bucket_name):
                if _is_audio(b.name) and vid in b.name:
                    candidates.append(b)
        except Exception:
            pass

    # newest audio in recent window
    try:
        import datetime as dt
        cutoff = dt.datetime.utcnow() - dt.timedelta(minutes=max_age_minutes)
        for b in client.list_blobs(bucket_name):
            if _is_audio(b.name):
                upd = getattr(b, "updated", None)
                if not upd or upd.replace(tzinfo=None) >= cutoff:
                    candidates.append(b)
    except Exception:
        pass

    uniq = {b.name: b for b in candidates}
    if not uniq:
        log.error("GCS fallback: no audio-like objects found in '%s'.", bucket_name)
        return None

    blobs = list(uniq.values())
    blobs.sort(key=lambda b: getattr(b, "updated", None) or getattr(b, "time_created", None), reverse=True)
    best = blobs[0]
    log.info("GCS candidate: gs://%s/%s", bucket_name, best.name)

    orig_ext = Path(best.name).suffix.lower() or ".bin"
    tmp_path = output_dir / f"{desired_basename}{orig_ext}"
    out_mp3 = output_dir / f"{desired_basename}.mp3"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        best.download_to_filename(tmp_path)

        if tmp_path.suffix.lower() != ".mp3":
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

        meta = {
            "extractor": "apify_gcs",
            "webpage_url": url or "",
            "gcs_bucket": bucket_name,
            "gcs_object": best.name,
            "download_attempt": "apify_gcs_fallback",
        }
        log.info("GCS fallback success → %s (source: gs://%s/%s)", final_path, bucket_name, best.name)
        return final_path, meta
    except Exception as e:
        log.error("GCS fallback: error downloading %s: %s", best.name, e)
        return None

# ------------------------------ main entry ----------------------------------- #

def download_audio(url: str, output_dir: Path, base_filename: str, type_input: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Robust audio download with yt-dlp, Apify and GCS fallbacks.
    Returns (audio_path, metadata) or None.
    """
    # Build proxy (Apify proxy if available)
    enrich: List[str] = []
    proxy_url = (_cfg("YTDLP_PROXY_URL", "") or "").strip()
    if not proxy_url:
        ap_pw = (_cfg("APIFY_PROXY_PASSWORD", "") or "").strip()
        ap_cty = APIFY_PROXY_COUNTRY
        if ap_pw:
            proxy_url = f"http://auto:{ap_pw}@proxy.apify.com:8000/?country={ap_cty}"
    if proxy_url:
        enrich += ["--proxy", proxy_url]

    # Binaries present?
    if not YT_DLP_PATH:
        log.error("yt-dlp not found.")
        return None
    if not FFMPEG_PATH:
        log.error("ffmpeg not found.")
        return None

    # Ensure output dir exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("Failed to create output directory %s: %s", output_dir, e)
        return None

    # Cookies / UA
    temp_paths: List[str] = []
    temp_cookies_file: Optional[str] = None

    cookies_b64 = (_cfg("YTDLP_COOKIES_B64", "") or "").strip()
    cookies_file = (_cfg("YTDLP_COOKIES_FILE", "") or "").strip()
    cookies_from_browser = (_cfg("YTDLP_COOKIES_FROM_BROWSER", "") or "").strip()

    if cookies_b64:
        try:
            fd, tmp = tempfile.mkstemp(suffix=".cookies.txt")
            os.close(fd)
            Path(tmp).write_bytes(base64.b64decode(cookies_b64))
            os.chmod(tmp, 0o600)
            temp_paths.append(tmp)
            cookies_file = tmp
            log.info("Decoded cookies from YTDLP_COOKIES_B64 into: %s", tmp)
        except Exception as e:
            log.warning("Failed to decode YTDLP_COOKIES_B64: %s", e)

    if cookies_file:
        try:
            fd, tmp_copy = tempfile.mkstemp(suffix=".cookies.txt")
            os.close(fd)
            shutil.copyfile(cookies_file, tmp_copy)
            os.chmod(tmp_copy, 0o600)
            temp_paths.append(tmp_copy)
            temp_cookies_file = _ensure_utf8_netscape(tmp_copy, temp_paths)
            log.info("Using temp cookies file at: %s", temp_cookies_file)
        except Exception as e:
            log.warning("Could not copy cookies file '%s': %s", cookies_file, e)
            temp_cookies_file = None

    def _cleanup():
        for p in temp_paths:
            try:
                os.remove(p)
                log.info("Removed temp file: %s", p)
            except Exception:
                pass

    # Best-effort metadata (never aborts)
    meta_basic: Dict[str, Any] = {
        "title": "Unknown Title",
        "uploader": "Unknown Uploader",
        "upload_date": None,
        "webpage_url": url,
        "duration": None,
        "extractor": "unknown",
        "view_count": None,
        "thumbnail": None,
        "type_input": type_input,
    }
    try:
        import yt_dlp  # type: ignore
        ydl_opts: Dict[str, Any] = {"quiet": True, "no_warnings": True, "extract_flat": False}
        if temp_cookies_file:
            ydl_opts["cookiefile"] = temp_cookies_file
        elif cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = cookies_from_browser
        if YTDLP_USER_AGENT:
            ydl_opts["user_agent"] = YTDLP_USER_AGENT
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        meta_basic.update({
            "title": info.get("title") or meta_basic["title"],
            "uploader": info.get("uploader") or info.get("channel") or info.get("uploader_id") or meta_basic["uploader"],
            "upload_date": info.get("upload_date"),
            "webpage_url": info.get("webpage_url", url),
            "duration": info.get("duration"),
            "extractor": info.get("extractor_key", info.get("extractor", "unknown")),
            "view_count": info.get("view_count"),
            "thumbnail": info.get("thumbnail"),
        })
    except Exception as e:
        log.warning("yt-dlp metadata extraction failed: %s", e)

    # Build yt-dlp command base
    enrich_cli = list(enrich)
    if YTDLP_USER_AGENT:
        enrich_cli += [
            "--user-agent", YTDLP_USER_AGENT,
            "--add-header", "Accept-Language: en-US,en;q=0.9",
            "--add-header", "Referer: https://www.youtube.com/",
        ]
    if str(_cfg("YTDLP_GEO_BYPASS", "true")).lower() not in ("false", "0", "no"):
        enrich_cli += ["--geo-bypass", "--geo-bypass-country", str(_cfg("YTDLP_GEO_COUNTRY", "US"))]
    if temp_cookies_file:
        enrich_cli += ["--cookies", temp_cookies_file]
    elif cookies_from_browser:
        enrich_cli += ["--cookies-from-browser", cookies_from_browser]

    output_path_template = str(output_dir / f"{base_filename}.%(ext)s")
    final_audio_path = output_dir / f"{base_filename}.{AUDIO_FORMAT}"

    base_cmd = [
        YT_DLP_PATH, url,
        "-x", "--audio-format", AUDIO_FORMAT,
        "--no-playlist", "--no-write-info-json",
        "--progress", "--no-simulate", "--no-abort-on-error",
        "--restrict-filenames", "-o", output_path_template, "--force-ipv4",
    ] + enrich_cli

    attempts: List[Tuple[str, List[str]]] = [
        ("primary", base_cmd),
        ("tv_embedded", base_cmd + ["--extractor-args", "youtube:player_client=tv_embedded"]),
        ("web_forced", base_cmd + ["--extractor-args", "youtube:webpage_download_web=1"]),
        ("bestaudio_ladder",
         [YT_DLP_PATH, url, "-f", "251/140/bestaudio/best",
          "--no-playlist", "--no-write-info-json", "--progress", "--no-simulate",
          "--no-abort-on-error", "--restrict-filenames", "-o", output_path_template,
          "--force-ipv4"] + enrich_cli),
    ]

    log.info("Attempting to download audio from: %s", url)
    apify_tried = False
    last_err: Optional[Exception] = None

    try:
        for label, cmd in attempts:
            try:
                log.debug("[yt-dlp] Attempt '%s': %s", label, " ".join(cmd))
                cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if cp.stdout:
                    log.debug("yt-dlp stdout:\n%s", cp.stdout)
                if cp.stderr:
                    log.debug("yt-dlp stderr:\n%s", cp.stderr)

                # Expected output
                if final_audio_path.exists():
                    _cleanup()
                    return str(final_audio_path), {**meta_basic, "download_attempt": label}

                # Or any audio we can convert
                for cand in sorted(output_dir.glob(f"{base_filename}.*"),
                                   key=lambda p: p.stat().st_mtime, reverse=True):
                    if cand.suffix.lower() in [".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav"]:
                        if cand.suffix.lower() == ".mp3":
                            _cleanup()
                            return str(cand), {**meta_basic, "download_attempt": label}
                        out = _ensure_mp3(cand, final_audio_path)
                        if out:
                            _cleanup()
                            return out, {**meta_basic, "download_attempt": label}

                last_err = RuntimeError("yt-dlp completed without producing expected output.")
                log.error("Attempt '%s' completed but no usable audio file was produced.", label)

            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                log.error("yt-dlp failed (exit %s) on attempt '%s'", e.returncode, label)
                log.error("Command: %s", " ".join(e.cmd if isinstance(e.cmd, list) else [str(e.cmd)]))
                if stderr:
                    log.error("Stderr:\n%s", stderr)
                last_err = e

                if not apify_tried and _is_region_lock(stderr):
                    apify_tried = True
                    log.info("Detected region/bot/geo block. Trying Apify fallback…")
                    ap = _apify_download_audio(url, output_dir, base_filename)
                    if ap:
                        _cleanup()
                        p, m = ap
                        return p, {**meta_basic, **m}
                    else:
                        log.error("Apify fallback failed; continuing ladder.")
                continue

            except FileNotFoundError:
                log.error("yt-dlp not found at %s", YT_DLP_PATH)
                last_err = FileNotFoundError("yt-dlp not found")
                break

            except Exception as e:
                log.error("Unexpected error during '%s': %s", label, e, exc_info=True)
                last_err = e
                continue

    finally:
        _cleanup()

    log.error("All yt-dlp attempts failed.")
    if not apify_tried:
        log.info("Trying Apify fallback at end…")
        ap = _apify_download_audio(url, output_dir, base_filename)
        if ap:
            p, m = ap
            return p, {**meta_basic, **m}

    if last_err:
        log.error("Last error: %s", last_err)
    return None
