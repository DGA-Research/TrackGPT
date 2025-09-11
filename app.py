# app.py
import streamlit as st
import json
import time
from urllib.parse import urlencode
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import requests
import base64, hmac, hashlib, os
from pathlib import Path
from io import BytesIO
import csv

# --- Your project modules ---
from config import Config
import downloader
import transcriber
import analyzer
import output as output_mod

st.set_page_config(page_title="TrackGPT · YouTube OAuth", layout="wide")

CLIENT_CONFIG = {
    "web": {
        "client_id":     st.secrets["google_oauth"]["client_id"],
        "client_secret": st.secrets["google_oauth"]["client_secret"],
        "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
        "token_uri":     "https://oauth2.googleapis.com/token",
        "redirect_uris": [st.secrets["google_oauth"]["redirect_uri"]],
    }
}

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

# =========================
# Stateless signed state helpers
# =========================
STATE_SIGNING_KEY = st.secrets["google_oauth"].get(
    "state_secret",
    st.secrets["google_oauth"]["client_secret"],  # fallback for dev
)

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

def _unb64url(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

def make_state() -> str:
    payload = {"ts": int(time.time()), "nonce": _b64url(os.urandom(16))}
    body = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    sig = _b64url(hmac.new(STATE_SIGNING_KEY.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"

def verify_state(state: str, max_age_sec: int = 300) -> bool:
    try:
        body, sig = state.split(".")
        expected = _b64url(hmac.new(STATE_SIGNING_KEY.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected): return False
        payload = json.loads(_unb64url(body))
        if abs(time.time() - int(payload["ts"])) > max_age_sec: return False
        return True
    except Exception:
        return False

def _first(val):
    return val[0] if isinstance(val, list) else val

# =========================
# OAuth helpers
# =========================
def get_flow():
    return Flow.from_client_config(
        CLIENT_CONFIG,
        scopes=SCOPES,
        redirect_uri=CLIENT_CONFIG["web"]["redirect_uris"][0]
    )

def begin_auth():
    flow = get_flow()
    state_token = make_state()
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state_token,
    )
    st.query_params["oauth"] = "start"  # optional breadcrumb
    st.write("Redirecting to Google…")
    st.markdown(f"[Continue]({auth_url})")

def handle_callback():
    params = dict(st.query_params)
    state = params.get("state"); code = params.get("code")
    if not state or not code:
        st.error("Missing OAuth parameters.")
        return
    state = _first(state); code = _first(code)
    if not verify_state(state):
        st.error("State verification failed. Try again.")
        return

    flow = get_flow()
    flow.fetch_token(code=code)
    creds = flow.credentials
    st.session_state["creds"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
        "expiry": creds.expiry.timestamp() if creds.expiry else None,
    }
    st.query_params.clear()
    st.success("Signed in with Google!")

def ensure_fresh_creds():
    data = st.session_state.get("creds")
    if not data: return None
    creds = Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data["token_uri"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        scopes=data["scopes"]
    )
    if not creds.valid and creds.refresh_token:
        try:
            creds.refresh(Request())
            st.session_state["creds"]["token"] = creds.token
            st.session_state["creds"]["expiry"] = (creds.expiry.timestamp() if creds.expiry else None)
        except Exception as e:
            st.error(f"Token refresh failed: {e}")
            return None
    return creds

def yt_get(subpath, creds, params=None):
    url = f"https://www.googleapis.com/youtube/v3/{subpath}"
    headers = {"Authorization": f"Bearer {creds.token}"}
    r = requests.get(url, headers=headers, params=params or {})
    r.raise_for_status()
    return r.json()

# =========================
# Handle the OAuth callback ASAP (once)
# =========================
qs = dict(st.query_params)
if "code" in qs and "state" in qs:
    handle_callback()

# =========================
# Optional password gate AFTER callback
# =========================
PASS = st.secrets.get("password")
if PASS:
    if not st.session_state.get("pw_ok"):
        st.title("Enter password")
        pw = st.text_input("Password", type="password")
        if st.button("Unlock"):
            st.session_state["pw_ok"] = (pw == PASS)
        st.stop()

# =========================
# Google sign-in gate
# =========================
st.title("TrackGPT")
if "creds" not in st.session_state:
    st.info("Authorize YouTube read access to continue.")
    if st.button("Sign in with Google"):
        begin_auth()
    st.stop()

# =========================
# Cookie OTC helpers (FastAPI service)
# =========================
COOKIE_API_BASE = st.secrets.get("COOKIE_API_BASE", "").rstrip("/")
def _require_cookie_api():
    if not COOKIE_API_BASE:
        st.warning("COOKIE_API_BASE not set. Set it in Secrets to enable one-click cookie handoff.")
        return False
    return True

def issue_otc():
    r = requests.post(f"{COOKIE_API_BASE}/issue", timeout=10)
    r.raise_for_status()
    data = r.json()
    return data["otc"], data["expires_at"]

def claim_cookies(otc: str):
    r = requests.post(f"{COOKIE_API_BASE}/claim", json={"otc": otc}, timeout=10)
    if r.status_code == 404:
        return None  # not uploaded yet
    r.raise_for_status()
    return r.json()["cookies_b64"]

def set_cookies_env_from_session():
    """
    Just-in-time: expose session cookies (if present) to downloader via env.
    """
    if st.session_state.get("cookies_b64"):
        os.environ["YTDLP_COOKIES_B64"] = st.session_state["cookies_b64"]
    else:
        os.environ.pop("YTDLP_COOKIES_B64", None)

# =========================
# Main App UI
# =========================

# Sidebar: Account + Session Cookies
with st.sidebar:
    st.subheader("Account")
    if st.button("Sign out"):
        st.session_state.pop("creds", None)
        st.query_params.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("Session cookies")
    st.caption("Use for gated YouTube videos. Cookies are stored only for this session.")

    # One-time code flow (extension push)
    if _require_cookie_api():
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Generate code"):
                try:
                    otc, exp = issue_otc()
                    st.session_state["cookie_otc"] = otc
                    st.session_state["cookie_exp"] = exp
                except Exception as e:
                    st.error(f"Could not issue code: {e}")
        with colB:
            if st.button("Claim cookies"):
                try:
                    otc = st.session_state.get("cookie_otc")
                    if not otc:
                        st.info("Generate a code first.")
                    else:
                        cookies_b64 = claim_cookies(otc)
                        if cookies_b64 is None:
                            st.info("Not uploaded yet. In the extension, paste the code and click “Send cookies”, then try Claim again.")
                        else:
                            st.session_state["cookies_b64"] = cookies_b64
                            set_cookies_env_from_session()
                            st.success("Session cookies loaded.")
                except requests.HTTPError as e:
                    st.error(f"Claim failed: {getattr(e.response, 'text', e)}")
                except Exception as e:
                    st.error(f"Claim failed: {e}")

        if st.session_state.get("cookie_otc"):
            st.caption("One-time code (paste in the Chrome extension):")
            st.code(st.session_state["cookie_otc"], language="text")
            if COOKIE_API_BASE:
                st.text_input("Extension endpoint", value=f"{COOKIE_API_BASE}/upload", disabled=True)

    # Manual upload/paste fallback
    with st.expander("Manual cookies upload/paste"):
        upl = st.file_uploader("Upload cookies.txt (Netscape format)", type=["txt"], key="cookies_upl")
        pasted = st.text_area("…or paste cookies.txt content", key="cookies_paste", height=120)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Use uploaded/pasted"):
                try:
                    if upl:
                        content = upl.read()
                    elif pasted.strip():
                        content = pasted.encode("utf-8")
                    else:
                        content = None
                    if not content:
                        st.info("Provide a file or paste cookies first.")
                    else:
                        st.session_state["cookies_b64"] = base64.b64encode(content).decode("ascii")
                        set_cookies_env_from_session()
                        st.success("Session cookies loaded.")
                except Exception as e:
                    st.error(f"Failed to load cookies: {e}")
        with col2:
            if st.button("Clear session cookies"):
                st.session_state.pop("cookies_b64", None)
                set_cookies_env_from_session()
                st.info("Cleared.")

# ======= Workflows =======

def run_single(url: str, target_name: str, speaker_hint: str):
    # Ensure JIT cookies exposure before download
    set_cookies_env_from_session()

    outdir = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Download audio + metadata
    with st.status("Downloading audio...", expanded=True) as s:
        base_filename = "trackgpt"
        type_input = "youtube" if "youtube" in url.lower() else "web"
        dl = downloader.download_audio(url, outdir, base_filename, type_input)
        if not dl:
            s.update(label="Download failed", state="error")
            st.stop()
        audio_path, metadata = dl
        s.write(f"Saved audio: `{audio_path}`")
        s.write(f"Metadata: {json.dumps(metadata, indent=2)[:8000]}")
        s.update(label="Download complete", state="complete")

    # 2) Transcribe
    with st.status("Transcribing...", expanded=True) as s:
        try:
            transcript_text = transcriber.transcribe_file(
                audio_file_path=audio_path,
                openai_key=Config.OPENAI_API_KEY,
                assemblyai_key=Config.ASSEMBLYAI_API_KEY,
                speaker_hint=speaker_hint or None
            )
        except Exception as e:
            s.update(label="Transcription failed", state="error")
            st.error(str(e))
            st.stop()
        s.write(f"Transcript length: {len(transcript_text):,} chars")
        s.update(label="Transcription complete", state="complete")

    # 3) Analyze → bullets + highlights
    with st.status("Analyzing transcript...", expanded=True) as s:
        try:
            bullets = analyzer.extract_raw_data_from_text(
                transcript_text=transcript_text,
                target_name=target_name or "Unknown",
                metadata=metadata,
                open_ai_api=Config.OPENAI_API_KEY,
                prompt_type="format_text_bullet_prompt",
                max_bullets=100,
            )
            highlights = analyzer.extract_raw_data_from_text(
                transcript_text=transcript_text,
                target_name=target_name or "Unknown",
                metadata=metadata,
                open_ai_api=Config.OPENAI_API_KEY,
                prompt_type="format_text_highlight_prompt",
                max_bullets=50,
            )
        except Exception as e:
            s.update(label="Analysis failed", state="error")
            st.error(str(e))
            st.stop()
        s.write(f"Extracted {len(bullets)} bullets; {len(highlights)} highlights")
        s.update(label="Analysis complete", state="complete")

    # 4) Save outputs (txt + html)
    with st.status("Generating outputs...", expanded=True) as s:
        transcript_path = outdir / "transcript.txt"
        output_mod.save_text_file(transcript_text, transcript_path)

        (outdir / "results").mkdir(exist_ok=True)
        (outdir / "results" / "bullets.json").write_text(json.dumps(bullets, indent=2))
        (outdir / "results" / "highlights.json").write_text(json.dumps(highlights, indent=2))

        report_html = output_mod.generate_report_both(
            metadata=metadata,
            extracted_bullets_raw=bullets,
            extracted_highlights_raw=highlights,
            transcript_text=transcript_text,
            target_name=target_name or "Unknown",
            html_or_docx="html",
        )
        report_path = outdir / "report.html"
        report_path.write_text(report_html, encoding="utf-8")

        s.write(f"Saved: `{transcript_path.name}`, `{report_path.name}`")
        s.update(label="Outputs ready", state="complete")

    # Show results & downloads
    st.subheader("Results")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download transcript (.txt)",
            data=transcript_path.read_text(encoding="utf-8"),
            file_name=transcript_path.name,
            mime="text/plain",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download report (.html)",
            data=report_path.read_text(encoding="utf-8"),
            file_name=report_path.name,
            mime="text/html",
            use_container_width=True,
        )

    with st.expander("Metadata / JSON (debug)"):
        st.json(metadata)
        st.json({"bullets": bullets[:10], "highlights": highlights[:10]})

def run_bulk_csv(file):
    """
    Expect CSV with headers: url, target_name, speaker_hint
    """
    text = file.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    st.write(f"Loaded {len(rows)} rows")

    progress = st.progress(0.0)
    done = 0
    for row in rows:
        url = (row.get("url") or "").strip()
        target = (row.get("target_name") or "").strip()
        speaker = (row.get("speaker_hint") or "").strip()
        if not url:
            continue
        st.markdown("---")
        st.write(f"**Processing:** {url}")
        try:
            run_single(url=url, target_name=target, speaker_hint=speaker)
        except Exception as e:
            st.error(f"Row failed: {e}")
        done += 1
        progress.progress(done / max(1, len(rows)))

# Tabs for your workflows
tab1, tab2, tab3 = st.tabs(["Single video", "Bulk / CSV", "Google API test"])

with tab1:
    st.header("Single video transcript & report")
    url = st.text_input("Video URL (YouTube, etc.)")
    c1, c2 = st.columns(2)
    with c1:
        target_name = st.text_input("Target name for analysis (appears in bullets)")
    with c2:
        speaker_hint = st.text_input("Speaker hint (optional, helps diarization naming)")
    if st.button("Transcribe & analyze", type="primary", use_container_width=True, disabled=not url.strip()):
        run_single(url=url.strip(), target_name=target_name.strip(), speaker_hint=speaker_hint.strip())

with tab2:
    st.header("Batch via CSV")
    up = st.file_uploader("Upload CSV (headers: url, target_name, speaker_hint)", type=["csv"])
    if up and st.button("Process CSV", use_container_width=True):
        run_bulk_csv(up)

with tab3:
    st.header("YouTube Data API sanity check")
    auth = ensure_fresh_creds()
    if st.button("Fetch my channel info", disabled=auth is None):
        try:
            me = yt_get("channels", auth, params={"part": "snippet,statistics", "mine": "true"})
            st.json(me)
        except Exception as e:
            st.error(f"API error: {e}")
