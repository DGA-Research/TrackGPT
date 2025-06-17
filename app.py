# app.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import re

# Import your existing modules directly (no argparse)
from config import Config
from downloader import download_audio
from transcriber import transcribe_file
from analyzer import extract_raw_bullet_data_from_text
from output import generate_html_report, save_text_file

# UI layout
st.set_page_config(page_title="TrackGPT", layout="centered")
st.title("TrackGPT: Tracking Report Tool")
url = "https://docs.google.com/document/d/1SR45h_w20Vn1-KrCRfAfkf2E2-aDvH-mXu8S2eA4630/edit?usp=sharing"
st.markdown("Questions? Check out the [TrackGPT Instructions](%s)" % url)

# Restart button
if "processing_done" not in st.session_state:
    st.session_state["processing_done"] = False
if st.button("🛑 Stop Running"):
    st.session_state.clear()

# Set up API keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Access the API key
ASSEMBLYAI_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]  # Access the API key

# Input
transcript_input = False
uploaded_file = False
download_button = st.checkbox("Enter my own mp3 or m4a file")
if download_button:
    uploaded_file = st.file_uploader("Upload an mp3 file", type=["mp3"], key="video_file")
    uploaded_file = st.file_uploader("Upload an m4a file", type=["m4a"], key="video_file2")
transcript_button = st.checkbox("Enter my own transcript file")
if transcript_button:
    transcript_input = st.text_area("Copy and paste transcript here", key="transcript_input")
video_url = st.text_input("Enter a video or audio URL. See [Supported Sources](%s)" % url)
target_name = st.text_input("Target Name*")
run_btn = st.button("Generate Tracking Report with Highlights")
run_bullets = st.button("Generate Tracking Report with Bullets")

if run_btn and transcript_input and target_name:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name)
    base_filename = f"{safe_name}_{timestamp}"
    output_dir = Path(Config.DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / f"{base_filename}_transcript.txt"
    report_path = output_dir / f"{base_filename}_report.html"

    metadata = {
                    'title': f"Existing file:",
                    'uploader': "Unknown (Download Skipped)",
                    'upload_date': None,
                    'webpage_url': "N/A",
                    'extractor': "Local file",
                }
    
    # Highlights
    with st.spinner("Writing Highlights..."):
        try:
            type = "format_text_highlight_prompt"
            bullets = extract_raw_bullet_data_from_text(transcript_input, target_name, metadata, OPENAI_API_KEY, type)
        except Exception as e:
            bullets = []
            st.warning("Bullet extraction failed.")
            
    # Report
    with st.spinner("Formatting Tracking Report..."):
        try:
            html = generate_html_report(metadata, bullets, transcript_input, target_name)
            save_text_file(html, report_path)
        except Exception as e:
            st.error(f"Failed to generate report: {e}")
            st.stop()

    # Output
    st.success("✅ Analysis complete!")
    st.download_button("📄 Download HTML Report", html, file_name=report_path.name)
    st.markdown(html, unsafe_allow_html=True)



elif run_btn and uploaded_file and target_name:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name)
    base_filename = f"{safe_name}_{timestamp}"
    output_dir = Path(Config.DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = uploaded_file
    transcript_path = output_dir / f"{base_filename}_transcript.txt"
    report_path = output_dir / f"{base_filename}_report.html"

    metadata = {
                    'title': f"Existing file: {audio_path.name}",
                    'uploader': "Unknown (Download Skipped)",
                    'upload_date': None,
                    'webpage_url': "N/A",
                    'extractor': "Local file",
                }
    
    # Transcribe
    with st.spinner("Transcribing..."):
        try:
            transcript = transcribe_file(uploaded_file, OPENAI_API_KEY, ASSEMBLYAI_API_KEY)
            save_text_file(transcript, transcript_path)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

        # Format transcript for HTML
        transcript = re.sub(r'(\[\d+:\d+:\d+\.\d+\] Speaker [A-Z])', r'</p><p>\1', transcript)
        transcript = '<p>' + transcript.strip() + '</p>'
            
    # Highlights
    with st.spinner("Writing Highlights..."):
        try:
            type = "format_text_highlight_prompt"
            bullets = extract_raw_bullet_data_from_text(transcript, target_name, metadata, OPENAI_API_KEY, type)
        except Exception as e:
            bullets = []
            st.warning("Bullet extraction failed.")
            
    # Report
    with st.spinner("Formatting Tracking Report..."):
        try:
            html = generate_html_report(metadata, bullets, transcript, target_name)
            save_text_file(html, report_path)
        except Exception as e:
            st.error(f"Failed to generate report: {e}")
            st.stop()

    # Output
    st.success("✅ Analysis complete!")
    st.download_button("📄 Download HTML Report", html, file_name=report_path.name)
    st.markdown(html, unsafe_allow_html=True)

elif run_btn and video_url and target_name:
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name)
    base_filename = f"{safe_name}_{timestamp}"
    output_dir = Path(Config.DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"
    transcript_path = output_dir / f"{base_filename}_transcript.txt"
    report_path = output_dir / f"{base_filename}_report.html"

    with st.spinner("Downloading..."):
        # Download
        try:
            audio_str, metadata = download_audio(video_url, output_dir, base_filename)
        except Exception as e:
                st.error(f"Download failed: {e}")
                if st.button("🔄 Start Over: Invalid Link. Try Again or Upload File as MP3"):
                    st.session_state.clear()
                    str.stop()

    with st.spinner("Transcribing..."):
        # Transcribe
        try:
            transcript = transcribe_file(audio_str, OPENAI_API_KEY, ASSEMBLYAI_API_KEY)
            save_text_file(transcript, transcript_path)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

        # Format transcript for HTML
        transcript = re.sub(r'(\[\d+:\d+:\d+\.\d+\] Speaker [A-Z])', r'</p><p>\1', transcript)
        transcript = '<p>' + transcript.strip() + '</p>'

    with st.spinner("Writing Highlights..."):
        # Analyze
        try:
            type = "format_text_highlight_prompt"
            bullets = extract_raw_bullet_data_from_text(transcript, target_name, metadata, OPENAI_API_KEY, type)
        except Exception as e:
            bullets = []
            st.warning("Bullet extraction failed.")

    with st.spinner("Formatting Tracking Report..."):
        # Report
        try:
            html = generate_html_report(metadata, bullets, transcript, target_name)
            save_text_file(html, report_path)
        except Exception as e:
                st.error(f"Failed to generate report: {e}")
                st.stop()

    # Output
    st.success("✅ Analysis complete!")

    # Save HTML and MP3 to session state to survive re-runs
    st.session_state["html_report"] = html
    try:
        with open(audio_path, "rb") as f:
            mp3_bytes = f.read()
        st.session_state["mp3_data"] = mp3_bytes
    except Exception as e:
        st.warning(f"Could not prepare MP3 download: {e}")
        st.session_state["mp3_data"] = None

    # Show HTML report before any buttons (so it doesn’t disappear)
    st.markdown(st.session_state["html_report"], unsafe_allow_html=True)

    # Download buttons (after rendering the report)
    st.download_button(
        "📄 Download HTML Report",
        data=st.session_state["html_report"],
        file_name=report_path.name,
        mime="text/html"
    )

    if st.session_state["mp3_data"]:
        st.download_button(
            "🎵 Download MP3 File",
            data=st.session_state["mp3_data"],
            file_name=audio_path.name,
            mime="audio/mpeg"
        )

elif run_bullets and transcript_input and target_name:
     # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name)
    base_filename = f"{safe_name}_{timestamp}"
    output_dir = Path(Config.DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"
    transcript_path = output_dir / f"{base_filename}_transcript.txt"
    report_path = output_dir / f"{base_filename}_report.html"

    with st.spinner("Downloading..."):
        # Download
        try:
            audio_str, metadata = download_audio(video_url, output_dir, base_filename)
        except Exception as e:
                st.error(f"Download failed: {e}")
                if st.button("🔄 Start Over: Invalid Link. Try Again or Upload File as MP3"):
                    st.session_state.clear()
                    str.stop()

    with st.spinner("Transcribing..."):
        # Transcribe
        try:
            transcript = transcribe_file(audio_str, OPENAI_API_KEY, ASSEMBLYAI_API_KEY)
            save_text_file(transcript, transcript_path)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

        # Format transcript for HTML
        transcript = re.sub(r'(\[\d+:\d+:\d+\.\d+\] Speaker [A-Z])', r'</p><p>\1', transcript)
        transcript = '<p>' + transcript.strip() + '</p>'

    with st.spinner("Writing Bullets..."):
        # Analyze
        type = "format_text_bullet_prompt"
        try:
            bullets = extract_raw_bullet_data_from_text(transcript_input, target_name, metadata, OPENAI_API_KEY, type)
        except Exception as e:
            bullets = []
            st.warning("Bullet extraction failed.")

    with st.spinner("Formatting Tracking Report..."):
        # Report
        try:
            html = generate_html_report(metadata, bullets, transcript, target_name)
            save_text_file(html, report_path)
        except Exception as e:
                st.error(f"Failed to generate report: {e}")
                st.stop()

    # Output
    st.success("✅ Analysis complete!")

    # Save HTML and MP3 to session state to survive re-runs
    st.session_state["html_report"] = html
    try:
        with open(audio_path, "rb") as f:
            mp3_bytes = f.read()
        st.session_state["mp3_data"] = mp3_bytes
    except Exception as e:
        st.warning(f"Could not prepare MP3 download: {e}")
        st.session_state["mp3_data"] = None

    # Show HTML report before any buttons (so it doesn’t disappear)
    st.markdown(st.session_state["html_report"], unsafe_allow_html=True)

    # Download buttons (after rendering the report)
    st.download_button(
        "📄 Download HTML Report",
        data=st.session_state["html_report"],
        file_name=report_path.name,
        mime="text/html"
    )

    if st.session_state["mp3_data"]:
        st.download_button(
            "🎵 Download MP3 File",
            data=st.session_state["mp3_data"],
            file_name=audio_path.name,
            mime="audio/mpeg"
        )
