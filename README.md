# TrackGPT Research Pipeline

TrackGPT combines a Streamlit front end, a downloading/transcription toolchain, and LLM-powered analysis prompts to turn long-form video or audio sources into structured tracking reports. The app can ingest hosted URLs, local files, or bulk ZIP archives, then transcribe via AssemblyAI, enrich speaker labels with OpenAI, and finally extract highlights/bullet points plus a formatted transcript.

## Highlights
- End-to-end workflow from video URL to HTML/DOCX report with transcripts, highlights, and bullet points.
- Automatic metadata capture (`yt-dlp`), optional cookies upload for signed-in/region-locked content, and resilient retries.
- AssemblyAI transcript generation with timestamped speaker diarization; OpenAI models add speaker names and run prompt-engineered extraction.
- Choice of report modes (highlights, bullets, both, or transcript-only) plus download buttons for audio, HTML, DOCX.
- Bulk transcription mode in both UI and CLI to process folders or ZIP archives beyond Streamlit upload limits.

## Streamlit Hosted App Link:
https://trackgpt-b9fpyc9vm5dlimhcs9nr2q.streamlit.app/

## Architecture & Modules
| Component | Responsibility |
| --- | --- |
| `app.py` | Streamlit UX, password gating, session-state workflow (input -> transcript editing -> report generation -> downloads). |
| `config.py` | Loads API keys from Streamlit secrets or env vars and validates they exist before anything else runs. |
| `downloader.py` | Wraps `yt-dlp` + `ffmpeg` to fetch audio, normalize metadata, and honor optional cookies. |
| `transcriber.py` | Sends audio to AssemblyAI for diarized transcripts, chunking long utterances and optionally using OpenAI to append human-readable speaker names. |
| `analyzer.py` + `prompts.py` | Prompt-engineered OpenAI calls for two report styles (highlights vs. structured bullets) with `tenacity`-based retries. |
| `output.py` | Builds HTML and DOCX-ready markup, strict title casing, and saves transcript/analysis files. |
| `bulk_transcribe_cli.py` | Headless entry point mirroring the UI bulk ZIP workflow. |

## Repository Layout
```text
TrackGPT-main/
+-- app.py                  # Streamlit application
+-- analyzer.py             # Highlight/bullet extraction
+-- bulk_transcribe_cli.py  # Offline transcription helper
+-- config.py               # Configuration loader & validation
+-- config.toml             # Streamlit server tweaks (max upload size, watched dirs)
+-- downloader.py           # yt-dlp + ffmpeg wrapper
+-- output.py               # Report generation helpers
+-- prompts.py              # Prompt templates used by analyzer
+-- transcriber.py          # AssemblyAI + OpenAI speaker labeling
+-- requirements.txt        # Python dependencies
+-- secrets.toml            # Sample Streamlit secrets file (replace with your values)
+-- output/                 # Created at runtime; holds transcripts & reports
```

## Requirements
### Services & API Keys
- `OPENAI_API_KEY` - used for speaker labeling and highlight/bullet extraction.
- `ASSEMBLYAI_API_KEY` - used for diarized transcription.
- Optional: YouTube cookies file (`YTDLP_COOKIES_FILE`) for age-restricted/region-locked downloads.

### Python
- Python 3.9+ (Streamlit + AssemblyAI SDK both support 3.9-3.12).
- Install dependencies: `pip install -r requirements.txt` (Streamlit itself is expected to come from your environment).

### System Binaries
- [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) in `PATH`.
- [`ffmpeg` and `ffprobe`](https://ffmpeg.org/download.html) in `PATH` (audio extraction + duration probing).

## Configuration
### Streamlit Secrets
Create `.streamlit/secrets.toml` (or update the provided `secrets.toml` before deploying) with:
```toml
OPENAI_API_KEY = "sk-..."
ASSEMBLYAI_API_KEY = "..."
password = "choose-a-ui-password"
```
### CLI / Local Scripts
For CLI workflows, place the same keys in a `.env` file (loaded via `python-dotenv`):
```dotenv
OPENAI_API_KEY=sk-...
ASSEMBLYAI_API_KEY=...
WHISPER_MODEL=whisper-1           # optional override
ANALYSIS_MODEL=gpt-4o-mini        # optional override
DEFAULT_OUTPUT_DIR=output         # default already
AUDIO_FORMAT=mp3
```
The config layer validates keys at import time; launching `streamlit run app.py` without the keys exits immediately.

## Running the Streamlit Workflow
1. **Install dependencies** and launch: `streamlit run app.py`. (The provided `config.toml` already raises `maxUploadSize` to 400 MB.)
2. **Authenticate** - the app is password protected via `st.secrets["password"]`.
3. **Step 1 - Input Source**
   - Provide a video/audio URL, upload an mp3/m4a/mp4, or paste an existing transcript.
   - Optionally upload a `cookies.txt` (via the "Get cookies.txt" browser extension) for signed-in YouTube access. The file is stored with 0600 perms and its path exported via `YTDLP_COOKIES_FILE`.
   - Toggle "Upload ZIP" to enter **bulk mode**: each supported file in the archive is transcribed, but highlights/bullets are skipped-results show per-file transcripts plus a downloadable ZIP.
   - Supply optional metadata (title, air date, source station, headline, logo URLs, notes) that feeds the final report header.
   - Choose report type: `Highlights`, `Bullets`, `Both`, or `Transcript Only`.
4. **Step 2 - Review & Edit Transcript**
   - Listen to the auto-downloaded audio, edit the transcript text directly, and adjust speaker labels. The UI enforces that labels (`Speaker A`, `Speaker B`, .) stay intact while letting you change the display names.
5. **Step 3 - Generate Report**
   - Depending on report type, the app calls `analyzer.extract_raw_data_from_text` with either the highlight or bullet prompt template (or both) and then formats the output via `output.py` helpers.
   - `html2docx` converts the HTML to a DOCX for word-processor delivery. HTML, DOCX, and the source transcript land in `output/<target>_<timestamp>_report.*`.
6. **Step 4 - Download Results**
   - Buttons provide HTML, DOCX (when conversion succeeds), and the original audio (`.mp3`) when available. Restart clears state without logging you out.

## Bulk Transcription CLI
Use the bundled helper to bypass Streamlit upload limits entirely:
```bash
python bulk_transcribe_cli.py --input path/to/folder-or-zip \
    --target "Target Name" \
    --output-dir output/bulk_$(date +%Y%m%d)
```
Key flags:
- `--input` accepts a single audio file, a directory, or a ZIP archive (extensions: mp3/m4a/mp4/wav/aac/flac/ogg/webm).
- `--target` seeds the speaker-labeling hint used by `transcriber.py`.
- `--no-zip` skips creating an aggregate `transcripts.zip`.
- `--openai-key` / `--assemblyai-key` override the `.env` values per run.
The CLI stages files in a temp folder, runs `transcribe_file` for each, writes `<stem>.txt` into the output directory, and (optionally) bundles them into `transcripts.zip`.

## Output Artifacts
- `output/<target>_<timestamp>_report.html` - Share-ready HTML (metadata header, highlights/bullets, transcript).
- `output/<target>_<timestamp>_report.docx` - Word export of the same content.
- `output/<stem>.txt` - Plain-text transcripts saved both from the UI and CLI.
- Optional `transcripts.zip` - Bulk CLI aggregate.
- Uploaded cookies are written to `cookies.txt` in the repo root and reused by `yt-dlp` for the session.

## Troubleshooting & Tips
- **`ffmpeg` / `yt-dlp` not found:** Confirm both commands run from your shell; the downloader exits early otherwise.
- **Import errors at startup:** `config.Config` validates keys immediately, so missing API keys manifest as `ConfigError` before Streamlit renders.
- **Large files:** Use the built-in chunking (automatic) or compress locally. The UI surfaces a link to a browser-based compressor for >600 MB uploads.
- **Rate limits:** `tenacity` retries OpenAI calls (up to 6 attempts for bullet extraction). Persistent failures show inline errors; rerun after waiting.
- **DOCX export failures:** The report still saves as HTML. Check console logs for `html2docx` errors; often caused by unsupported HTML tags.
- **Resetting state:** Use the "Restart" button; it clears everything except the password flag, ensuring secrets aren't re-entered repeatedly.

With this structure you can confidently describe, run, and extend TrackGPT's ingestion -> transcription -> analysis pipeline.
