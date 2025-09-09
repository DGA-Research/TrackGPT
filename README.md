# TrackGPT -- Video Research Transcriber and Analyzer

A **Streamlit web application** that automates the process of
researching and analyzing video or audio content. It downloads audio
with `yt-dlp` (with cookie, retry, and geo-bypass support), transcribes
it via the OpenAI Whisper API (with automatic chunking for large files),
extracts structured insights using the OpenAI GPT API, and generates
formatted HTML/DOCX reports.

------------------------------------------------------------------------

## Overview

The app streamlines analysis of video/audio sources in **four steps**:

1.  **Input**
    -   Upload an audio/video file\
    -   Paste a transcript\
    -   Provide a video/audio URL (e.g., YouTube, Vimeo)\
    -   *(Optional)* Upload a `cookies.txt` file to handle region-locked
        or consent-gated videos
2.  **Transcription**
    -   Converts audio into text using OpenAI's Whisper API\
    -   Automatically splits files \>25MB into overlapping chunks with
        `ffmpeg`\
    -   Recovers gracefully from failed chunk uploads
3.  **Analysis**
    -   Runs the transcript through an OpenAI GPT model (e.g.,
        GPT-4.1-mini, GPT-4o-mini)\
    -   Extracts **Headline, Speaker, Body, Source, Date** in structured
        bullets\
    -   Supports multiple report modes: Highlights, Bullets, Both, or
        Transcript-only
4.  **Report Generation**
    -   Produces HTML and DOCX reports (with citation formatting and
        speaker attribution)\
    -   Includes metadata (title, uploader, date, platform, duration)\
    -   Offers direct downloads of reports and audio files

------------------------------------------------------------------------

## Key Features

-   **Streamlit Web App:** Clean, interactive interface with
    step-by-step workflow\
-   **yt-dlp Hardening:**
    -   Cookie file support (`cookies.txt`, browser export, Base64, or
        URL download)\
    -   Configurable retries, geo-bypass, and user-agent override\
-   **Metadata Extraction:** Captures video title, uploader, upload
    date, platform, duration, and more\
-   **Robust Transcription:** Automatic chunking for large files, retry
    handling for API errors\
-   **Flexible Reports:** Choose Highlights, Bullets, Both, or
    Transcript-only\
-   **Download Options:** Export HTML, DOCX (via `html2docx`), and audio
    files\
-   **Configurable:** Secrets and defaults loaded from `st.secrets` or
    environment variables via `.env`\
-   **Secure:** API keys managed via Streamlit secrets or `.env`; cookie
    files stored with restricted permissions and cleaned up after use

------------------------------------------------------------------------

## Requirements

-   **Python:** 3.8+\
-   **API Keys:**
    -   `OPENAI_API_KEY` (required)\
    -   `ASSEMBLYAI_API_KEY` (required for current pipeline)\
-   **External Tools:**
    -   `yt-dlp` -- audio/video download and metadata extraction\
    -   `ffmpeg` + `ffprobe` -- required for audio conversion and
        chunking\
-   **Python Packages:**
    -   `streamlit`, `openai`, `python-dotenv`, `yt-dlp`, `html2docx`,
        `tenacity`

------------------------------------------------------------------------

## Setup

1.  **Clone the repository:**

    ``` bash
    git clone https://github.com/sh-patterson/TrackGPT-Audio
    cd TrackGPT-Audio
    ```

2.  **Create and activate virtual environment:**

    ``` bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**

    ``` bash
    pip install -r requirements.txt
    ```

4.  **Install external tools:**

    -   [yt-dlp](https://github.com/yt-dlp/yt-dlp#installation)\
    -   [ffmpeg & ffprobe](https://ffmpeg.org/download.html)\
        Make sure all are in your system PATH.

5.  **Configure API keys:**

    -   In `.env` (for local dev):

        ``` dotenv
        OPENAI_API_KEY=your-openai-key
        ASSEMBLYAI_API_KEY=your-assemblyai-key
        ```

    -   Or via **Streamlit Cloud Secrets** when deploying online.

6.  **Optional yt-dlp settings (in `.env`):**

    ``` dotenv
    YTDLP_COOKIES_FILE=cookies.txt
    YTDLP_GEO_BYPASS=true
    YTDLP_GEO_COUNTRY=US
    YTDLP_RETRIES=3
    ```

------------------------------------------------------------------------

## Yt-dlp Cookie Handling

Some videos (age-restricted, region-locked, or behind a consent wall)
require authenticated cookies to download properly. TrackGPT supports
cookies via file, environment, or Streamlit secrets.

### Manual Cookie Update via Base64

1.  Install the [**Get cookies.txt
    locally**](https://chrome.google.com/webstore/detail/get-cookiestxt-local-exp/naepdomgkenhinolocfifgehidddafch)
    browser extension.\

2.  Visit **youtube.com** and export cookies to a file (e.g.,
    `www.youtube.com_cookies.txt`).\

3.  Convert the cookie file to Base64 text:

    ``` bash
    base64 -w0 "www.youtube.com_cookies.txt" > "www.youtube.com_cookies.b64"
    ```

4.  Copy the text contents of `www.youtube.com_cookies.b64`.\

5.  In Streamlit secrets, add:

    ``` dotenv
    YTDLP_COOKIES_B64="IyBY2FwZSBIVFZSBGaWxlCiMgaHR0cDovL2N1cDov..."
    ```

6.  Re-run TrackGPT. If a log-in error appears, repeat steps 1--3 to
    refresh your cookies.

------------------------------------------------------------------------

## Usage

Launch the Streamlit app:

``` bash
streamlit run app.py
```

Navigate to the provided local URL. The workflow is:

1.  **Step 1:** Choose input (upload file, paste transcript, or URL) and
    select report type\
2.  **Step 2:** Review and edit transcript & speaker labels\
3.  **Step 3:** Generate the chosen report (highlights, bullets, both,
    or transcript only)\
4.  **Step 4:** Download reports/audio and optionally restart

------------------------------------------------------------------------

## Example

Generate a combined **Highlights + Bullets report** for a YouTube video:

1.  Run `streamlit run app.py`\

2.  Paste the URL:

        https://www.youtube.com/watch?v=rDexVZY3yYE

3.  Enter target name: `Kanye West`\

4.  Select **Generate Highlights and Bullets**\

5.  After processing, download the DOCX or HTML report with citations
    and transcript.
