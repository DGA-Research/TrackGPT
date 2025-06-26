# app.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import re
import hmac

def check_password():
    """Returns `True` if the user entered the correct password."""

    if st.session_state.get("password_correct", False):
        return True

    with st.form("password_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
        if submitted:
            if hmac.compare_digest(password, st.secrets["password"]):
                st.session_state["password_correct"] = True
                return True
            else:
                st.session_state["password_correct"] = False

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")

    return False

# Now use it like this
if check_password():
    # Import your existing modules directly (no argparse)
    from config import Config
    from downloader import download_audio
    from transcriber import transcribe_file
    from analyzer import extract_raw_bullet_data_from_text
    from output import generate_html_report, save_text_file
    from output import generate_html_report_bullets
    
    # UI layout
    st.set_page_config(page_title="TrackGPT", layout="centered")
    st.title("TrackGPT: Tracking Report Tool")
    url = "https://docs.google.com/document/d/1SR45h_w20Vn1-KrCRfAfkf2E2-aDvH-mXu8S2eA4630/edit?usp=sharing"
    st.markdown("Questions? Check out the [TrackGPT Instructions](%s)" % url)
    
    # Restart button
    if "processing_done" not in st.session_state:
        st.session_state["processing_done"] = False
    if st.button("🔄 Restart"):
        st.session_state.clear()
    
    # Set up API keys
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Access the API key
    ASSEMBLYAI_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]  # Access the API key
    
    # Input
    date_input = False
    type_input = False
    transcript_input = False
    uploaded_file = False
    download_button = st.checkbox("Enter my own mp3, m4a or mp4 file")
    if download_button:
        compress_url = "https://www.freeconvert.com/video-compressor"
        st.markdown(":blue-background[File over 600mb? Compress [here](%s) and then upload!]" % compress_url)
        uploaded_file_mp3 = st.file_uploader("Upload an mp3 file", type=["mp3"], key="video_file")
        uploaded_file_m4a = st.file_uploader("Upload an m4a file", type=["m4a"], key="video_file2")
        uploaded_file_mp4 = st.file_uploader("Upload an mp4 file", type=["mp4"], key="video_file3")
        # enter date
        date_input = st.text_input("Enter upload date: (MM/DD/YYYY)")
    
        uploaded_file = uploaded_file_mp3 or uploaded_file_m4a or uploaded_file_mp4
        
    transcript_button = st.checkbox("Enter my own transcript file")
    if transcript_button:
        transcript_input = st.text_area("Copy and paste transcript here", key="transcript_input")
    video_url = st.text_input("Enter a video or audio URL. See [Supported Sources](%s)" % url)
    type_input = st.text_input("Enter file type (VIDEO or AUDIO):")
    target_name = st.text_input("Target Name*")
    run_highlights = st.button("Generate Tracking Report with Highlights")
    run_bullets = st.button("Generate Tracking Report with Bullets")
    
    
    if ((run_highlights or run_bullets) and (transcript_input or not transcript_button) and (uploaded_file or video_url or transcript_input)):
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
                        'upload_date': date_input,
                        'webpage_url': "N/A",
                        'extractor': "Local file",
                        'type_input': type_input
                    }
        
        # Download Step
        with st.spinner("Downloading..."):
            # download if user entered a url
            if video_url:
                try:
                    audio_str, metadata = download_audio(video_url, output_dir, base_filename, type_input)
                    audio_path = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"
                except Exception as e:
                        st.error(f"Download failed: {e}")
                        if st.button("🔄 Start Over: Invalid Link. Try Again or Upload File as MP3"):
                            st.session_state.clear()
                            str.stop()
            # if user uploaded a file, download it and set it to the audio path
            elif uploaded_file:
                audio_str = uploaded_file
                audio_path = uploaded_file
            else:
                audio_path = "none"
    
        # Transcribe Step
        with st.spinner("Transcribing..."):
            # transcribe if user does not upload their own file
            if not transcript_input:
                try:
                    transcript = transcribe_file(audio_str, OPENAI_API_KEY, ASSEMBLYAI_API_KEY)
                    save_text_file(transcript, transcript_path)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    st.stop()
            # use input transcription
            elif transcript_input:
                try:
                    transcript = transcript_input
                    save_text_file(transcript, transcript_path)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    st.stop()
            # Format transcript for HTML
            transcript = re.sub(r'(\[\d+:\d+:\d+\.\d+\] Speaker [A-Z])', r'</p><p>\1', transcript)
            transcript = '<p>' + transcript.strip() + '</p>'
            
        # Highlight/Bullet and Report Step
        if run_highlights:
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
        elif run_bullets:
            with st.spinner("Writing Bullets..."):
            # Analyze
                type = "format_text_bullet_prompt"
                try:
                    bullets = extract_raw_bullet_data_from_text(transcript, target_name, metadata, OPENAI_API_KEY, type)
                except Exception as e:
                    bullets = []
                    st.warning("Bullet extraction failed.")
    
            with st.spinner("Formatting Tracking Report..."):
                # Report
                try:
                    html = generate_html_report_bullets(metadata, bullets, transcript, target_name)
                    save_text_file(html, report_path)
                except Exception as e:
                        st.error(f"Failed to generate report: {e}")
                        st.stop()
    
        # Output
        st.success("✅ Analysis complete!")
    
        # Save HTML and MP3 to session state to survive re-runs
        st.session_state["html_report"] = html
        if video_url:
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
    
        if video_url:
            st.download_button(
                "🎵 Download MP3 File",
                data=st.session_state["mp3_data"],
                file_name=audio_path.name,
                mime="audio/mpeg"
            )
    
    # Display saved results after report generation
    elif "html_report" in st.session_state:
        st.markdown(st.session_state["html_report"], unsafe_allow_html=True)
    
        st.download_button(
            "📄 Download HTML Report",
            data=st.session_state["html_report"],
            file_name="report.html",
            mime="text/html"
        )
    
        if "mp3_data" in st.session_state and st.session_state["mp3_data"]:
            st.download_button(
                "🎵 Download MP3 File",
                data=st.session_state["mp3_data"],
                file_name="audio.mp3",
                mime="audio/mpeg"
            )
