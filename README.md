# Video Research Transcriber and Analyzer

A Python tool designed to automate the process of researching video content. It downloads audio using `yt-dlp`, transcribes it via the OpenAI Whisper API (automatically handling large files via chunking), extracts structured factual statements about a specified target using the OpenAI GPT API, and generates a comprehensive HTML report.

## Overview

This script streamlines the analysis of video/audio sources by performing the following steps:

1.  **Download:** Fetches audio and associated metadata from a given URL (e.g., YouTube, Vimeo) using `yt-dlp`.
2.  **Transcribe:** Converts the downloaded audio into text using OpenAI's Whisper API. **Crucially, it automatically detects files exceeding the Whisper API's ~25MB size limit and splits them into smaller, overlapping chunks using `ffmpeg` before transcription.** Failed chunks do not stop the process; partial transcripts are combined.
3.  **Extract:** Analyzes the full transcript using an OpenAI GPT model (e.g., GPT-4o Mini) with a carefully engineered prompt to identify and extract key factual statements, claims, or commitments related to a specific target person/entity. This step captures structured data (**Headline, Speaker, Body, Source, Date**) via text delimiters parsed by the script.
4.  **Report:** Generates a self-contained, formatted HTML report containing the video metadata, extracted bullet points (with title-cased headlines, verbatim body quotes, source/date citations, and links), and the full transcript for verification.

## Key Features

*   **Automated Pipeline:** Full workflow from URL to structured HTML report.
*   **Metadata Extraction:** Captures video title, uploader, date, duration, etc., using `yt-dlp`.
*   **Robust Transcription:** Leverages OpenAI's Whisper API and **automatically handles large audio files (>24MB) via `ffmpeg`-based chunking** with configurable overlap, ensuring complete transcription without hitting API limits. Handles individual chunk failures gracefully.
*   **Structured Analysis:** Extracts key information about a target into structured bullet points (**Headline, Speaker, Body, Source, Date**) using an OpenAI GPT model and sophisticated prompt engineering.
*   **Comprehensive Reporting:** Generates a clean, readable HTML report with a dynamic title, metadata, formatted & cited bullet points, and the full transcript.
*   **Robust Error Handling:** Includes retries for GPT API calls (`tenacity`), handles chunk transcription failures, and implements retries for temporary file cleanup on Windows.
*   **Flexible Usage:** Supports skipping specific steps (download, transcription, extraction) if intermediate files exist.
*   **Configurable:** Settings managed via a `.env` file (API keys, models, output format, default overlap).
*   **Command-Line Interface:** Easy to use via CLI arguments.

## Requirements

*   **Python:** Version 3.8 or higher.
*   **OpenAI API Key:** Required for transcription and analysis. Obtain from [OpenAI Platform](https://platform.openai.com/).
*   **External Tools:**
    *   `yt-dlp`: Command-line tool for downloading video/audio. ([Installation](https://github.com/yt-dlp/yt-dlp#installation))
    *   `ffmpeg` **and** `ffprobe`: Command-line tools for audio processing. **Crucial for both audio format conversion (by `yt-dlp`) and the automatic audio chunking functionality in the transcriber.** ([Download](https://ffmpeg.org/download.html))
    *   All three (`yt-dlp`, `ffmpeg`, `ffprobe`) must be installed and accessible in your system's PATH.
*   **Python Packages:** Listed in `requirements.txt` (if you create one). Key dependencies include:
    *   `openai`: For interacting with OpenAI APIs.
    *   `python-dotenv`: For managing environment variables.
    *   `yt-dlp`: Python wrapper (used for metadata extraction).
    *   `tenacity`: For robust GPT API call retries.
    *   *(Note: `pydub` is no longer required if using the `ffmpeg`-based chunking in the latest `transcriber.py`)*

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/sh-patterson/TrackGPT-Audio
    cd TrackGPT-Audio
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    # Ensure you have a requirements.txt file reflecting the dependencies listed above, or install manually:
    pip install openai python-dotenv yt-dlp tenacity
    ```
    *(Create/update a `requirements.txt` file for easier setup: `pip freeze > requirements.txt`)*

4.  **Install `yt-dlp`, `ffmpeg`, and `ffprobe`:**
    Follow the installation instructions linked in the [Requirements](#requirements) section to ensure these are installed and available in your system's PATH. **Verify `ffmpeg` and `ffprobe` installation, as they are essential for the chunking feature.**

5.  **Configure OpenAI API Key (Required):**
    *   Create a file named `.env` in the project root directory.
    *   Add your OpenAI API key to the `.env` file:
        ```dotenv
        OPENAI_API_KEY=your-actual-api-key-here
        ```
    *   **Security:** Never commit your `.env` file to version control. Add `.env` to your `.gitignore` file.

6.  **Optional Configuration (in `.env`):**
    You can override default settings by adding these variables to your `.env` file:
    ```dotenv
    # Model for transcription (via OpenAI API)
    WHISPER_MODEL=whisper-1

    # Model for analysis/extraction (via OpenAI API)
    ANALYSIS_MODEL=gpt-4o-mini # Or your preferred model like gpt-4-turbo

    # Directory for output files
    DEFAULT_OUTPUT_DIR=output

    # Audio format for downloads (requires ffmpeg)
    AUDIO_FORMAT=mp3

    # --- Transcriber Specific (Optional) ---
    # Overlap in seconds for audio chunking (default is 2 if not set)
    # DEFAULT_OVERLAP_SECONDS=2
    ```

## Usage

Run the script from your terminal using the following structure:

```bash
python main.py "<VIDEO_URL>" "<TARGET_NAME>" [OPTIONS]
```

**Arguments:**

*   `<VIDEO_URL>`: The URL of the video/audio source (enclose in quotes if it contains special characters).
*   `<TARGET_NAME>`: The name of the person or entity to focus the analysis on (enclose in quotes if it contains spaces).

**Options:**

*   `-o DIRECTORY`, `--output_dir DIRECTORY`: Specify the directory for output files (default: `output`).
*   `--skip_download`: Skip downloading audio (requires the expected audio file to exist in the output directory).
*   `--skip_transcription`: Skip audio transcription (requires the expected transcript `.txt` file to exist).
*   `--skip_extraction`: Skip bullet point extraction using the GPT model.
*   `-v`, `--verbose`: Enable DEBUG level logging for more detailed output.

**Example:**

```bash
python main.py "https://www.youtube.com/watch?v=rDexVZY3yYE" "Kanye West" -o ./results --verbose
```

This command will download the audio from the specified YouTube URL, save intermediate files and the final report to the `./results` directory, focus the analysis on "Kanye West", and provide detailed logging output. If the audio file is large, it will automatically be chunked during transcription.
