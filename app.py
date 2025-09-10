# streamlit_app.py
import streamlit as st
import json
import time
from urllib.parse import urlencode
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import requests

st.set_page_config(page_title="YouTube OAuth in Streamlit")

# 1) Put your Google Cloud OAuth client (type: Web) here
#    Create at https://console.cloud.google.com/apis/credentials
#    Add an authorized redirect URI: http://localhost:8501 (or your deployed URL)
CLIENT_CONFIG = {
    "web": {
        "client_id":     st.secrets["google_oauth"]["client_id"],
        "client_secret": st.secrets["google_oauth"]["client_secret"],
        "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
        "token_uri":     "https://oauth2.googleapis.com/token",
        "redirect_uris": [st.secrets["google_oauth"]["redirect_uri"]],
    }
}

# Scopes for read-only YouTube access; add more if you need to write.
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

def get_flow():
    return Flow.from_client_config(
        CLIENT_CONFIG,
        scopes=SCOPES,
        redirect_uri=CLIENT_CONFIG["web"]["redirect_uris"][0]
    )

def begin_auth():
    flow = get_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"  # ensures we can get a refresh_token the first time
    )
    st.session_state["oauth_state"] = state
    st.experimental_set_query_params(oauth="start")  # cosmetic
    st.write("Redirecting to Google…")
    st.markdown(f"[Continue]({auth_url})")

def handle_callback():
    # Google redirects back with ?state=...&code=...
    params = st.experimental_get_query_params()
    if "state" not in params or "code" not in params:
        st.error("Missing OAuth parameters.")
        return

    if params["state"][0] != st.session_state.get("oauth_state"):
        st.error("State mismatch. Try again.")
        return

    flow = get_flow()
    flow.fetch_token(code=params["code"][0])
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
    # Clear query params so refreshes don't re-run callback
    st.experimental_set_query_params()
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
    # Refresh if near expiry
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

st.title("YouTube login (OAuth) in Streamlit")

# 2) Handle the OAuth callback when Google redirects back
qs = st.experimental_get_query_params()
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
    st.experimental_set_query_params()
    st.rerun()
