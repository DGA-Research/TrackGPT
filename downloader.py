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
import urllib.request
import json
import requests

log = logging.getLogger(__name__)


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

def _apify_download_audio(url: str, output_dir: Path, base_filename: str) -> tuple[str, dict] | None:
    token = os.getenv("APIFY_TOKEN", "")
    timeout_ms = int(os.getenv("APIFY_TIMEOUT_MS", "180000"))
    if not token:
        log.warning("APIFY_TOKEN not set; skipping Apify fallback.")
        return None

    endpoint = (
        "https://api.apify.com/v2/acts/streamers~youtube-video-downloader/"
        "run-sync-get-dataset-items?token=" + token
    )

    payload = {
        # minimal inputs for the actor; see Apify docs for more options
        "videoUrl": url,
        "convertToAudio": True,
        "audioFormat": "mp3"
    }

    try:
        log.info("Apify fallback: requesting actor run for %s", url)
        resp = requests.post(endpoint, json=payload, timeout=timeout_ms/1000)
        resp.raise_for_status()
        items = resp.json() if resp.content else []
        if not isinstance(items, list) or not items:
            log.error("Apify returned no items.")
            return None

        # The actor usually returns a downloadable URL in `audio` or inside `downloads.audio`
        audio_url = None
        it = items[0]
        audio_url = (
            it.get("audio")
            or (it.get("downloads") or {}).get("audio")
            or it.get("url")  # last-ditch if they returned direct MP3 url as 'url'
        )

        if not audio_url:
            log.error("Apify response missing audio URL: %s", json.dumps(it)[:500])
            return None

        out_path = output_dir / f"{base_filename}.mp3"
        with requests.get(audio_url, stream=True, timeout=timeout_ms/1000) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)

        if out_path.exists() and out_path.stat().st_size > 0:
            log.info("Apify fallback success → %s", out_path)
            meta = {"extractor": "apify", "webpage_url": url, "download_attempt": "apify_fallback"}
            return str(out_path), meta

        log.error("Apify fallback produced no file.")
        return None

    except requests.HTTPError as e:
        log.error("Apify HTTP error: %s %s", e.response.status_code if e.response else "?", e)
    except Exception as e:
        log.error("Apify fallback failed: %s", e, exc_info=True)
    return None


def _apify_ytdl_fallback(
    url: str,
    output_dir: Path,
    base_filename: str,
    audio_format: str,
    ffmpeg_path: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Try Apify's 'streamers/youtube-video-downloader' actor as a region-locked fallback.
    Requires APIFY_TOKEN (env or Config). Downloads best audio (or video -> audio).
    Returns (audio_path, metadata) or None.
    """
    token = os.getenv("APIFY_TOKEN", getattr(Config, "APIFY_TOKEN", "")).strip()
    if not token:
        log.warning("APIFY_TOKEN not configured; skipping Apify fallback.")
        return None

    # Allow caller to tweak timeout if needed; default 180s
    timeout_ms = int(os.getenv("APIFY_TIMEOUT_MS", getattr(Config, "APIFY_TIMEOUT_MS", 180_000)))
    # Run actor and get dataset items in one call
    endpoint = (
        "https://api.apify.com/v2/acts/streamers~youtube-video-downloader/"
        f"run-sync-get-dataset-items?token={token}&timeout={timeout_ms}"
    )

    # Minimal input: the actor accepts either `videoUrl` or `videoUrls`
    # We’ll use `videoUrls` to be forward-compatible.
    payload = {
        "videoUrls": [url],
        # You can uncomment below if you have Apify Proxy groups/geo you want to force.
        # "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
        # "maxQuality": "best",
        # "getAudioOnly": True,  # if supported by the actor; harmless if ignored
    }

    try:
        log.info("Apify fallback: requesting actor run for URL.")
        resp = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=(30, max(60, timeout_ms / 1000)),
        )
        if resp.status_code != 200:
            log.error("Apify fallback HTTP %s: %s", resp.status_code, resp.text[:4000])
            return None

        try:
            items = resp.json()
        except Exception:
            # Some Apify endpoints return NDJSON/JSONL; try a lenient parse
            text = resp.text.strip()
            if not text:
                log.error("Apify fallback returned empty body.")
                return None
            # If newline-delimited JSON, keep the first line
            first_line = text.splitlines()[0]
            try:
                items = json.loads(first_line)
            except Exception as e:
                log.error("Apify fallback: could not parse response: %s", e)
                return None

        # The actor usually returns a list; normalize to list
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list) or not items:
            log.error("Apify fallback: unexpected response structure.")
            return None

        # Find best downloadable candidate
        # Common shapes:
        # - item["downloads"] list with dicts containing "type", "format", "url"
        # - or direct fields like "audio" / "video" with URLs
        best_audio_url = None
        best_video_url = None
        item0 = items[0]

        downloads = item0.get("downloads") or []
        # Prefer explicit audio entries
        for d in downloads:
            t = (d.get("type") or "").lower()
            f = (d.get("format") or "").lower()
            u = d.get("url")
            if u and ("audio" in t or f in ("m4a", "webm", "opus", "mp3", "ogg")):
                best_audio_url = u
                break

        if not best_audio_url:
            # If no explicit audio, try to find best video instead
            # (some actors expose "video" or mp4 links with highest quality first)
            for d in downloads:
                f = (d.get("format") or "").lower()
                u = d.get("url")
                if u and (f in ("mp4", "webm") or "video" in (d.get("type") or "").lower()):
                    best_video_url = u
                    break

        # Some outputs expose flat keys
        if not best_audio_url:
            for key in ("audioUrl", "audio", "audio_link"):
                if item0.get(key):
                    best_audio_url = item0[key]
                    break

        if not best_audio_url and not best_video_url:
            # As a last resort, see if there’s a single direct `url`
            if item0.get("url"):
                best_video_url = item0["url"]

        if not best_audio_url and not best_video_url:
            log.error("Apify fallback: no downloadable URLs found in actor output.")
            return None

        # Download the file into a temp path, then ensure mp3
        tmp_in = output_dir / f"{base_filename}.apify.tmp"
        tmp_out = output_dir / f"{base_filename}.{audio_format}"

        def _http_download(src_url: str, dst_path: Path) -> bool:
            try:
                with requests.get(src_url, stream=True, timeout=(15, 120)) as r:
                    r.raise_for_status()
                    with open(dst_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)
                return True
            except Exception as e:
                log.error("Apify fallback: download failed: %s", e)
                return False

        if best_audio_url:
            log.info("Apify fallback: downloading direct audio…")
            if not _http_download(best_audio_url, tmp_out):
                return None
            # If it's already mp3, we’re done; else convert
            if tmp_out.suffix.lower() != f".{audio_format}":
                # rename to tmp_in and convert
                tmp_in2 = tmp_in.with_suffix(Path(best_audio_url).suffix or ".bin")
                try:
                    tmp_out.rename(tmp_in2)  # reuse path var for convenience
                except Exception:
                    # if rename fails, write into tmp_in2 again
                    if not _http_download(best_audio_url, tmp_in2):
                        return None
                final = _ensure_mp3(tmp_in2, tmp_out)
                if not final:
                    return None
                try:
                    if tmp_in2.exists():
                        tmp_in2.unlink()
                except Exception:
                    pass
            audio_path = str(tmp_out)
        else:
            # We got a video; download & extract audio
            log.info("Apify fallback: downloading video then extracting audio…")
            if not _http_download(best_video_url, tmp_in):
                return None
            final = _ensure_mp3(tmp_in, tmp_out)
            if not final:
                return None
            audio_path = final

        # Basic metadata (merge if actor provides more)
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
        # Cleanup temp file if exists
        try:
            if tmp_in.exists():
                tmp_in.unlink()
        except Exception:
            pass
        return (audio_path, meta)

    except requests.RequestException as e:
        log.error("Apify fallback network error: %s", e)
    except Exception as e:
        log.error("Apify fallback unexpected error: %s", e, exc_info=True)

    return None




def _ensure_utf8_netscape(cookie_path: str, temp_list: list[str]) -> str:
    """
    Ensure cookie file is UTF-8 text. If not, re-encode to UTF-8 in a temp file.
    Also detect accidental SQLite DBs.
    """
    try:
        with open(cookie_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        logging.warning(f"Cannot read cookies file '{cookie_path}': {e}")
        return cookie_path

    # Detect SQLite DBs (wrong file)
    if data.startswith(b"SQLite format 3"):
        logging.error("Provided cookies file appears to be a Chrome/SQLite DB, not a Netscape cookies.txt export. "
                      "Please export with a 'cookies.txt' extension/format.")
        return cookie_path

    # Try UTF-8 first
    try:
        data.decode('utf-8')  # ok as-is
        return cookie_path
    except UnicodeDecodeError:
        pass

    # Try UTF-8 with BOM
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        # Fallback to Latin-1 to salvage common CP1252 characters
        text = data.decode('latin-1')

    # Normalize line endings and re-write as UTF-8
    text = text.replace('\r\n', '\n')
    import tempfile, os
    fd, newp = tempfile.mkstemp(suffix=".cookies.utf8.txt")
    os.close(fd)
    with open(newp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.chmod(newp, 0o600)
    temp_list.append(newp)
    logging.info(f"Re-encoded cookies to UTF-8 at: {newp}")
    return newp

def _download_cookies_to_temp(url: str) -> Optional[str]:
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".cookies.txt")
        os.close(fd)
        with urllib.request.urlopen(url, timeout=10) as resp, open(tmp_path, "wb") as out:
            out.write(resp.read())
        os.chmod(tmp_path, 0o600)
        logging.info(f"Fetched cookies from URL into: {tmp_path}")
        return tmp_path
    except Exception as e:
        logging.warning(f"Could not fetch cookies from {url}: {e}")
        return None

cookies_url = os.getenv("YTDLP_COOKIES_URL", getattr(Config, "YTDLP_COOKIES_URL", "") if hasattr(Config, "YTDLP_COOKIES_URL") else "").strip()
if cookies_url and not orig_cookies_file:
    fetched = _download_cookies_to_temp(cookies_url)
    if fetched:
        orig_cookies_file = fetched
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
    Downloads audio from a given URL using yt-dlp with resilient fallbacks.

    Returns:
        (path_to_audio, metadata) if successful; otherwise None.
    """
    import os, tempfile, shutil, base64
    import subprocess, logging
    from typing import Optional, Dict, Any

    # Sanity: yt-dlp + ffmpeg present?
    if not YT_DLP_PATH:
        logging.error("yt-dlp executable not found. Cannot download.")
        return None
    if not FFMPEG_PATH:
        logging.error("ffmpeg not found in PATH. Cannot convert audio.")
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

    # --- Cookies: prefer Base64 in secrets/env, else file path; ensure writable path for yt-dlp ---
    # Read env/secrets (env overrides Config at runtime)
    orig_cookies_file = os.getenv("YTDLP_COOKIES_FILE", getattr(Config, "YTDLP_COOKIES_FILE", "")).strip()
    cookies_b64 = os.getenv(
        "YTDLP_COOKIES_B64",
        getattr(Config, "YTDLP_COOKIES_B64", "") if hasattr(Config, "YTDLP_COOKIES_B64") else ""
    ).strip()
    cookies_from_browser = os.getenv(
        "YTDLP_COOKIES_FROM_BROWSER",
        getattr(Config, "YTDLP_COOKIES_FROM_BROWSER", "") if hasattr(Config, "YTDLP_COOKIES_FROM_BROWSER") else ""
    ).strip()
    user_agent = os.getenv("YTDLP_USER_AGENT", getattr(Config, "YTDLP_USER_AGENT", "")).strip()

    # We'll collect any temp files we create to clean them up later
    temp_paths_to_cleanup: list[str] = []

    # If B64 is provided, materialize it to a temp file and prefer that
    if cookies_b64:
        try:
            fd, tmp_path_from_b64 = tempfile.mkstemp(suffix=".cookies.txt")
            os.close(fd)
            with open(tmp_path_from_b64, "wb") as f:
                f.write(base64.b64decode(cookies_b64))
            os.chmod(tmp_path_from_b64, 0o600)
            logging.info(f"Decoded cookies from secrets into: {tmp_path_from_b64}")
            orig_cookies_file = tmp_path_from_b64
            temp_paths_to_cleanup.append(tmp_path_from_b64)
        except Exception as e:
            logging.warning(f"Failed to decode YTDLP_COOKIES_B64: {e}")

    # If we have a cookies file path, copy it to a writable temp file (yt-dlp writes back on close)
    temp_cookies_file: Optional[str] = None
    if orig_cookies_file:
        try:
            fd, tmp_copy = tempfile.mkstemp(suffix=".cookies.txt")
            os.close(fd)
            shutil.copyfile(orig_cookies_file, tmp_copy)
            os.chmod(tmp_copy, 0o600)
            logging.info(f"Using temp cookies file at: {tmp_copy}")
            temp_cookies_file = tmp_copy
            if temp_cookies_file:
                temp_cookies_file = _ensure_utf8_netscape(temp_cookies_file, temp_paths_to_cleanup)

            temp_paths_to_cleanup.append(tmp_copy)
        except Exception as e:
            logging.warning(f"Could not create temp cookies file from '{orig_cookies_file}': {e}. Continuing without cookies.")
            temp_cookies_file = None

    # Helper to cleanup temp cookies
    def _cleanup_temp_cookies():
        for p in temp_paths_to_cleanup:
            try:
                os.remove(p)
                logging.info(f"Removed temp cookies file: {p}")
            except Exception:
                pass

    # --- Metadata Extraction (no download) ---
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
        logging.error(f"Unexpected error during metadata extraction for {url}: {e}", exc_info=True)
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
    attempts.append(("primary", base_cmd + enrich))
    attempts.append(("tv_embedded", base_cmd + enrich + ["--extractor-args", "youtube:player_client=tv_embedded"]))
    attempts.append(("web_forced", base_cmd + enrich + ["--extractor-args", "youtube:webpage_download_web=1"]))

    # bestaudio ladder; convert locally if needed
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

    try:
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
                            return (str(cand), {**metadata, "download_attempt": label})
                        out = _ensure_mp3(cand, final_audio_path)
                        if out:
                            logging.info(f"Success ({label}) + convert → {out}")
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
                return None
            except Exception as e:
                logging.error(f"Unexpected error during download (attempt '{label}'): {e}", exc_info=True)
                last_err = e
                continue
    finally:
        # Always cleanup temp cookies we created
        _cleanup_temp_cookies()

    log.error("All yt-dlp attempts failed.")
    # After your yt-dlp ladder fails:
    apify_result = _apify_download_audio(url, output_dir, base_filename)
    if apify_result:
        return apify_result
        
    if last_err:
        err_text = f"{last_err}"
        log.error("Last error: %s", err_text)

        # ── Region-lock heuristic → Apify fallback ─────────────────────────────
        geo_msgs = [
            "The uploader has not made this video available in your country",
            "This video is not available in your country",
            "is not available in your country",
        ]
        if any(m in err_text for m in geo_msgs):
            log.info("Detected region lock. Trying Apify fallback…")
            ap = _apify_ytdl_fallback(
                url=url,
                output_dir=output_dir,
                base_filename=base_filename,
                audio_format=Config.AUDIO_FORMAT,
                ffmpeg_path=FFMPEG_PATH,
            )
            if ap:
                return ap
                
    return None






