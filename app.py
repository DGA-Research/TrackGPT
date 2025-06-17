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
st.title("🎧 TrackGPT: Video Transcriber & Analyzer")

# Set up API keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Access the API key
ASSEMBLYAI_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]  # Access the API key

# Input
download_button = st.checkbox("Enter my own mp3 file")
if download_button:
    uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])
transcript_button = st.checkbox("Enter my own transcript file")
if transcript_button:
    transcript_input = st.text_area("Copy and paste transcript here")
video_url = st.text_input("Enter a YouTube URL or local video path")
target_name = st.text_input("Enter target name (person or entity)")
run_btn = st.button("Run Analysis")

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
            bullets = extract_raw_bullet_data_from_text(transcript_input, target_name, metadata, OPENAI_API_KEY)
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



if run_btn and uploaded_file and target_name:
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
            
    # Highlights
    with st.spinner("Writing Highlights..."):
        try:
            bullets = extract_raw_bullet_data_from_text(transcript, target_name, metadata, OPENAI_API_KEY)
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

if run_btn and video_url and target_name:
    with st.spinner("Processing..."):

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
                st.stop()

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
                bullets = extract_raw_bullet_data_from_text(transcript, target_name, metadata, OPENAI_API_KEY)
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
        st.download_button("📄 Download HTML Report", html, file_name=report_path.name)
        st.markdown(html, unsafe_allow_html=True)
