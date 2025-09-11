# streamlit_app.py
import streamlit as st
import json
import time
from urllib.parse import urlencode
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import requests
import base64, hmac, hashlib, os

st.set_page_config(page_title="YouTube OAuth in Streamlit")

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

# Use a dedicated secret for signing if you have one; else fall back to client_secret.
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
    payload = {
        "ts": int(time.time()),
        "nonce": _b64url(os.urandom(16)),
    }
    body = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    sig = _b64url(hmac.new(STATE_SIGNING_KEY.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"
    
def verify_state(state: str, max_age_sec: int = 300) -> bool:
    try:
        body, sig = state.split(".")
        expected = _b64url(hmac.new(STATE_SIGNING_KEY.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return False
        payload = json.loads(_unb64url(body))
        if abs(time.time() - int(payload["ts"])) > max_age_sec:
            return False
        return True
    except Exception:
        return False



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
        include_granted_scopes="true",  # str or True both work
        prompt="consent",
        state=state_token,              # << stateless signed state
    )
    # Cosmetic breadcrumb in URL during outbound leg (optional)
    st.query_params["oauth"] = "start"
    st.write("Redirecting to Google…")
    st.markdown(f"[Continue]({auth_url})")
    
def _first(val):
    # st.query_params can yield str or list[str]
    return val[0] if isinstance(val, list) else val

def handle_callback():
    params = dict(st.query_params)
    state = params.get("state")
    code  = params.get("code")

    if not state or not code:
        st.error("Missing OAuth parameters.")
        return

    state = _first(state)
    code  = _first(code)

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

    # Prevent re-running the callback on refresh
    st.query_params.clear()
    st.success("Signed in with Google!")


def ensure_fresh_creds():
    data = st.session_state.get("creds")
    if not data:
        return None
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
            st.session_state["creds"]["expiry"] = (
                creds.expiry.timestamp() if creds.expiry else None
            )
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
    
# Handle the OAuth callback ASAP
qs = dict(st.query_params)
if "code" in qs and "state" in qs:
    handle_callback()



st.title("YouTube login (OAuth) in Streamlit")

# 2) Handle the OAuth callback when Google redirects back
# was: qs = st.experimental_get_query_params()
qs = dict(st.query_params)
if "code" in qs and "state" in qs:
    handle_callback()

# 3) If not signed in, show a Sign in button
if "creds" not in st.session_state:
    if st.button("Sign in with Google"):
        begin_auth()
    st.stop()

# 4) Use the token to call YouTube Data API
creds = ensure_fresh_creds()
if not creds:
    st.warning("Please sign in again.")
    if st.button("Sign in with Google"):
        begin_auth()
    st.stop()

st.success("You’re signed in!")

# Example: get the channel of the current user
if st.button("Fetch my channel info"):
    try:
        me = yt_get("channels", creds, params={"part": "snippet,statistics", "mine": "true"})
        st.json(me)
    except Exception as e:
        st.error(f"API error: {e}")

# Optional: sign out
if st.button("Sign out"):
    st.session_state.pop("creds", None)
    # was: st.experimental_set_query_params()
    st.query_params.clear()
    st.rerun()
