import streamlit as st
from pathlib import Path
from datetime import datetime
import re
import hmac
import io
from html2docx import html2docx

from urllib.parse import urlsplit

import os

import logging
log = logging.getLogger(__name__)

def is_youtube(u: str) -> bool:
    try:
        host = urlsplit(u).netloc.lower()
        return ("youtube.com" in host) or ("youtu.be" in host) or ("youtube-nocookie.com" in host)
    except Exception:
        return False

level = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, level, logging.INFO),
    format="%(levelname)s:%(name)s:%(message)s"
)

st.set_page_config(page_title="TrackGPT", layout="centered")

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
        st.error("Incorrect Password")

    return False

if check_password():
    # Import functions
    from config import Config
    import downloader as downloader_module

    def download_audio_no_apify(
        url: str,
        output_dir: Path,
        base_filename: str,
        type_input,
    ):
        """Wrapper around downloader.download_audio that skips Apify fallbacks."""
        original_apify = getattr(downloader_module, "_apify_download_audio", None)
        original_ytdl = getattr(downloader_module, "_apify_ytdl_fallback", None)

        def _disabled(*args, **kwargs):
            log.info("Apify fallback disabled for this run.")
            return None

        try:
            if original_apify is not None:
                downloader_module._apify_download_audio = _disabled
            if original_ytdl is not None:
                downloader_module._apify_ytdl_fallback = _disabled
            return downloader_module.download_audio(
                url,
                output_dir,
                base_filename,
                type_input,
            )
        finally:
            if original_apify is not None:
                downloader_module._apify_download_audio = original_apify
            if original_ytdl is not None:
                downloader_module._apify_ytdl_fallback = original_ytdl

    from transcriber import transcribe_file
    from analyzer import extract_raw_data_from_text
    from output import generate_report_highlights, save_text_file, generate_report_bullets, generate_report_both
    
    # UI layout
    st.title("TrackGPT: Tracking Report Tool")
    url = "https://docs.google.com/document/d/1SR45h_w20Vn1-KrCRfAfkf2E2-aDvH-mXu8S2eA4630/edit?usp=sharing"
    st.markdown("Questions? Check out the [TrackGPT Instructions](%s)" % url)
    
    st.markdown(
        "Optional: provide YouTube cookies for sign-in/consent/region-locked videos.\n\n"
        "1. Install the [Get cookies.txt](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) Chrome extension.\n"
        "2. Open youtube.com in Chrome while signed in.\n"
        "3. Use the extension to export cookies.txt and upload it below."
    )
    cookies_file = st.file_uploader("Upload cookies.txt", type=["txt"])
    if cookies_file is not None:
        cookies_path = Path("cookies.txt").absolute()
        cookies_path.write_bytes(cookies_file.read())
        os.chmod(cookies_path, 0o600)
        os.environ["YTDLP_COOKIES_FILE"] = str(cookies_path)
        st.success(f"Cookies loaded: {cookies_path}")

    # Initialize session state
    if "step" not in st.session_state:
        st.session_state.step = "input"
    if "report_type" not in st.session_state:
        st.session_state.report_type = None
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "metadata" not in st.session_state:
        st.session_state.metadata = {}
    if "target_name" not in st.session_state:
        st.session_state.target_name = ""
    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None
    if "transcript_docx" not in st.session_state:
        st.session_state.transcript_docx = ""
    
    # Restart button
    if st.button("Restart"):
        for key in list(st.session_state.keys()):
            if key != "password_correct":
                del st.session_state[key]
        st.session_state.step = "input"
        st.rerun()
    
    # Set up API keys
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    ASSEMBLYAI_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]
    
    # STEP 1: INPUT
    if st.session_state.step == "input":
        st.header("Step 1: Input Source")
        
        # Set input options to false
        type_input = False
        transcript_input = False
        uploaded_file = False
        
        download_button = st.checkbox("Enter my own mp3, m4a or mp4 file")
        if download_button:
            # Provide link to web compressor
            compress_url = "https://www.freeconvert.com/video-compressor"
            st.markdown(":blue-background[File over 600mb? Compress [here](%s) and then upload!]" % compress_url)
            # Options for file upload
            uploaded_file_mp3 = st.file_uploader("Upload an mp3 file", type=["mp3"], key="video_file")
            uploaded_file_m4a = st.file_uploader("Upload an m4a file", type=["m4a"], key="video_file2")
            uploaded_file_mp4 = st.file_uploader("Upload an mp4 file", type=["mp4"], key="video_file3")
            uploaded_file = uploaded_file_mp3 or uploaded_file_m4a or uploaded_file_mp4
            
        transcript_button = st.checkbox("Enter my own transcript file")
        if transcript_button:
            transcript_input = st.text_area("Copy and paste transcript here", key="transcript_input")
            
        video_url = st.text_input("Enter a video or audio URL. See [Supported Sources](%s)" % url)

        # Enter file type (only relevant for bullets)
        type_input = st.selectbox("Enter file type:", ["AUDIO", "VIDEO"])
        
        # Optional metadata
        title_box = st.checkbox("Enter Title: (optional)")
        title = st.text_input("Enter Title:") if title_box else "Existing file:"
        
        uploader_box = st.checkbox("Enter Uploader/Channel: (optional)")
        uploader = st.text_input("Enter Uploader/Channel:") if uploader_box else "Unknown (Download Skipped)"
        
        upload_date_box = st.checkbox("Enter Upload Date: (optional)")
        upload_date = st.text_input("Enter Upload Date:") if upload_date_box else "Unknown"
        
        platform_box = st.checkbox("Enter Platform: (optional)")
        platform = st.text_input("Enter Platform:") if platform_box else "Local file"
        
        target_name = st.text_input("Target Name*")
        
        # Select which type of report
        st.subheader("Select Report Type:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Generate with Highlights"):
                st.session_state.report_type = "highlights"
        with col2:
            if st.button("Generate with Bullets"):
                st.session_state.report_type = "bullets"
        with col3:
            if st.button("Transcript Only"):
                st.session_state.report_type = "transcript_only"
        with col1:
            if st.button("Generate Highlights and Bullets"):
                st.session_state.report_type = "both"

        # Validate inputs and proceed
        if st.session_state.report_type and target_name and (transcript_input or uploaded_file or video_url):
            # Store data in session state
            st.session_state.target_name = target_name
            st.session_state.metadata = {
                'title': title,
                'uploader': uploader,
                'upload_date': upload_date,
                'webpage_url': "N/A",
                'extractor': platform,
                'type_input': type_input
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c if c.isalnum() else "_" for c in target_name)
            base_filename = f"{safe_name}_{timestamp}"
            output_dir = Path(Config.DEFAULT_OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process the input
            with st.spinner("Processing input..."):
                try:
                    # --- Handle audio source selection ---
                    audio_path = None

                    # A) URL input
                    if video_url:
                            download_result = download_audio_no_apify(
                                video_url,
                                output_dir,
                                base_filename,
                                type_input,
                            )
                            if download_result:
                                audio_path_str, metadata_update = download_result
                                st.session_state.metadata.update(metadata_update or {})
                                st.session_state.metadata["webpage_url"] = video_url
                                if not st.session_state.metadata.get("extractor"):
                                    st.session_state.metadata["extractor"] = "youtube" if is_youtube(video_url) else "generic"
                                st.session_state.audio_path = audio_path_str
                                audio_path = audio_path_str
                            else:
                                st.error("Processing failed: unable to download audio from the provided URL. Upload a file or provide a transcript instead.")
                                st.stop()

                    # B) Uploaded file
                    elif uploaded_file:
                        # Save uploaded file to the output dir with a safe name
                        upload_ext = Path(uploaded_file.name).suffix.lower() or ".mp3"
                        dest = output_dir / f"{base_filename}{upload_ext}"
                        with open(dest, "wb") as f:
                            f.write(uploaded_file.read())
                        st.session_state.audio_path = str(dest)
                        audio_path = str(dest)

                    # --- Transcript handling ---
                    if transcript_input:
                        transcript = transcript_input
                    else:
                        audio_path = audio_path or st.session_state.get("audio_path")  # add this line (optional but helpful)
                        if not audio_path:
                            raise ValueError("No audio source available to transcribe.")
                        transcript = transcribe_file(audio_path, OPENAI_API_KEY, ASSEMBLYAI_API_KEY, target_name)
                    # --- Format transcript HTML ---
                    transcript = re.sub(r'(\[\d+:\d+:\d+\] Speaker [A-Z])', r'</p><p>\1', transcript)
                    transcript = '<p>' + transcript.strip() + '</p>'

                    # --- Extract speaker labels for editor ---
                    pattern = r'\[[\d:.]+\]\s+(Speaker\s+[A-Z])\s+\(([^)]+)\):'
                    matches = re.findall(pattern, transcript)

                    unique_speakers = set()
                    unique_speakers_edit = set()

                    for speaker_id, name in matches:
                        unique_speakers.add(f"{speaker_id} ({name})")
                        unique_speakers_edit.add(f"{speaker_id}: {name}")

                    speaker_list = sorted(unique_speakers)
                    speaker_list_text = sorted(unique_speakers_edit)

                    st.session_state.speaker_list = speaker_list
                    st.session_state.speaker_list_text = speaker_list_text
                    st.session_state.transcript = transcript
                    st.session_state.step = "edit_transcript"
                    st.rerun()

                except Exception as e:
                    st.error(f"Processing failed: {e}")

        
        elif st.session_state.report_type and not target_name:
            st.error("Please enter a Target Name")
        elif st.session_state.report_type and not (transcript_input or uploaded_file or video_url):
            st.error("Please provide a transcript, upload a file, or enter a URL")
    
    # STEP 2: EDIT TRANSCRIPT
    elif st.session_state.step == "edit_transcript":
        st.header("Step 2: Review and Edit Transcript")
        
        # Show audio player if available
        if st.session_state.audio_path:
            st.audio(st.session_state.audio_path)
        
        # Show current report type
        st.info(f"Report Type: {st.session_state.report_type.title()}")
        
        # Edit Transcript Step for User
        edited_transcript = st.text_area(
            "Edit Transcript:",
            value=st.session_state.transcript.replace('<p>', '').replace('</p>', '\n\n'),
            height=400
        )

        # Confirm Speaker Step for User
        speaker_text = ""
        counter = 0
        # Format speaker edit text output
        for speaker in st.session_state.speaker_list_text:
            if counter == 0:
                speaker_text = speaker
            else:
                speaker_text = speaker_text + "\n" + speaker
            counter += 1

        st.markdown("To edit a speaker, change the name only and do not delete the label. See [Instructions](%s) for more details." % url)

        edited_speaker = st.text_area(
            "Edit Speakers:",
            value=speaker_text,
            height=100
        )

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Report"):
                # Update transcript with edits
                transcript = re.sub(r'(\[\d+:\d+:\d+\] Speaker [A-Z])', r'</p><p>\1', edited_transcript)
                transcript = '<p>' + transcript.strip() + '</p>'

                # st.session_state.speaker_list = ["Speaker A (Troy)", "Speaker B (Karrin Taylor Robson)"]
                
                pattern = r'Speaker [A-Z]:\s+([\w\s]+?)(?=Speaker [A-Z]:|$)'
                
                # Make list of edited speakers
                matches = re.findall(pattern, edited_speaker)
                # st.write("edited_speakers: " + edited_speaker)
                
                # Extract unique speakers while preserving order
                unique_speakers = []
                seen = set()
                for name in matches:
                    if name not in seen:
                        unique_speakers.append(name)
                        seen.add(name)
                
                speaker_list_edited = unique_speakers
                print("speaker_list_edited:", speaker_list_edited)
                print("st.session_state.speaker_list:", st.session_state.speaker_list)
                
                # Ensure both lists have the same length
                if len(st.session_state.speaker_list) != len(speaker_list_edited):
                    st.write("Issue with changing speakers. Please manually change the output.")
                    st.session_state.transcript = transcript
                    st.session_state.step = "generate_report"
                    st.rerun()
                else:
                    # Replace speaker labels in transcript
                    print("original speakers", st.session_state.speaker_list)
                    print("edited speakers", speaker_list_edited)
                    for original_speaker, edited_speaker in zip(st.session_state.speaker_list, speaker_list_edited):
                        transcript = transcript.replace(original_speaker, edited_speaker.strip())
                        # st.write(f"Replaced '{original_speaker}' with '{edited_speaker}'")
                        # st.write(transcript)

                    transcript_docx = re.sub(r'<p>', '<br><br>', transcript)
            
                    st.session_state.transcript_docx = transcript_docx
                    st.session_state.transcript = transcript
                    st.session_state.step = "generate_report"
                    st.rerun()
    
    # STEP 3: GENERATE REPORT
    elif st.session_state.step == "generate_report":
        st.header("Step 3: Generating Report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in st.session_state.target_name)
        base_filename = f"{safe_name}_{timestamp}"
        output_dir = Path(Config.DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / f"{base_filename}_report.html"
        docx_path = output_dir / f"{base_filename}_report.docx"
        
        try:
            # Call higlight step from analyzer.py
            if st.session_state.report_type == "highlights":
                with st.spinner("Writing Highlights..."):
                    bullets = extract_raw_data_from_text(
                        st.session_state.transcript, 
                        st.session_state.target_name, 
                        st.session_state.metadata, 
                        OPENAI_API_KEY, 
                        "format_text_highlight_prompt"
                    )

                # Format report with function from output.py
                with st.spinner("Formatting Report..."):
                    html = generate_report_highlights(
                        st.session_state.metadata, 
                        bullets, 
                        st.session_state.transcript, 
                        st.session_state.target_name,
                        "html"
                    )
                    docx = generate_report_highlights(
                        st.session_state.metadata, 
                        bullets, 
                        st.session_state.transcript_docx, 
                        st.session_state.target_name,
                        "docx"
                    )

            # Call bullet step from analyzer.py
            elif st.session_state.report_type == "bullets":
                with st.spinner("Writing Bullets..."):
                    bullets = extract_raw_data_from_text(
                        st.session_state.transcript, 
                        st.session_state.target_name, 
                        st.session_state.metadata, 
                        OPENAI_API_KEY, 
                        "format_text_bullet_prompt"
                    )
                # Format report with function from output.py
                with st.spinner("Formatting Report..."):
                    html = generate_report_bullets(
                        st.session_state.metadata, 
                        bullets, 
                        st.session_state.transcript, 
                        st.session_state.target_name,
                        "html"
                    )
                    docx = generate_report_bullets(
                        st.session_state.metadata, 
                        bullets, 
                        st.session_state.transcript_docx, 
                        st.session_state.target_name,
                        "docx"
                    )
            
            elif st.session_state.report_type == "both":
                # Call bullet step from analyzer.py
                with st.spinner("Writing Bullets..."):
                    bullets = extract_raw_data_from_text(
                        st.session_state.transcript, 
                        st.session_state.target_name, 
                        st.session_state.metadata, 
                        OPENAI_API_KEY, 
                        "format_text_bullet_prompt"
                    )
                # Call highlight step from analyzer.py
                with st.spinner("Writing Highlights..."):
                        highlights = extract_raw_data_from_text(
                            st.session_state.transcript, 
                            st.session_state.target_name, 
                            st.session_state.metadata, 
                            OPENAI_API_KEY, 
                            "format_text_highlight_prompt"
                        )
                # Format report with both bullets and highlights from output.py
                with st.spinner("Formatting Report..."):
                        html = generate_report_both(
                            st.session_state.metadata, 
                            bullets, 
                            highlights,
                            st.session_state.transcript, 
                            st.session_state.target_name,
                            "html"
                        )
                        docx = generate_report_both(
                            st.session_state.metadata, 
                            bullets, 
                            highlights,
                            st.session_state.transcript_docx, 
                            st.session_state.target_name,
                            "docx"
                        )
               
                
            else:  # transcript_only
                with st.spinner("Formatting Transcript..."):
                    html = f"<h2>{st.session_state.target_name} Transcript</h2>" + st.session_state.transcript
                    docx = f"<h2>{st.session_state.target_name} Transcript</h2>" + st.session_state.transcript_docx
            
            # Store results in session_state
            st.session_state.html_report = html
            save_text_file(html, html_path)

            st.session_state.docx_report = docx
            try:
                docx_document = html2docx(
                    st.session_state.docx_report,
                    title=f"{st.session_state.target_name} Report"
                )
                docx_buffer = io.BytesIO()
                docx_document.save(docx_buffer)
                docx_bytes = docx_buffer.getvalue()
                docx_path.write_bytes(docx_bytes)
                st.session_state.docx_bytes = docx_bytes
                st.session_state.docx_path = str(docx_path)
            except Exception as docx_error:
                st.session_state.docx_bytes = None
                st.session_state.docx_path = None
                st.warning(f"DOCX export failed: {docx_error}")

            # Prepare audio download if available
            if st.session_state.audio_path and isinstance(st.session_state.audio_path, str):
                try:
                    with open(st.session_state.audio_path, "rb") as f:
                        st.session_state.mp3_data = f.read()
                except:
                    st.session_state.mp3_data = None
            
            st.session_state.step = "show_results"
            st.rerun()
            
        except Exception as e:
            st.error(f"Report generation failed: {e}")
            if st.button("Back to Edit Transcript"):
                st.session_state.step = "edit_transcript"
                st.rerun()
    
    # STEP 4: SHOW RESULTS
    elif st.session_state.step == "show_results":
        st.success("Report complete!")
        
        # Show the report
        st.markdown(st.session_state.html_report, unsafe_allow_html=True)
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "Download HTML Report",
                data=st.session_state.html_report,
                file_name=f"{st.session_state.target_name}_report.html",
                mime="text/html"
            )
        
        with col2:
            if st.session_state.report_type in ['highlights', 'bullets', 'both', 'transcript_only']:
                docx_bytes = st.session_state.get("docx_bytes")
                if docx_bytes:
                    st.download_button(
                        label="Download DOCX",
                        data=docx_bytes,
                        file_name=f"{st.session_state.target_name}_report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                else:
                    st.caption("DOCX download is unavailable for this run.")

        with col3:
            if hasattr(st.session_state, 'mp3_data') and st.session_state.mp3_data:
                st.download_button(
                    "Download Audio File",
                    data=st.session_state.mp3_data,
                    file_name=f"{st.session_state.target_name}_audio.mp3",
                    mime="audio/mpeg"
                )
        
        # Option to start over
        if st.button("Create Another Report"):
            for key in list(st.session_state.keys()):
                if key != "password_correct":
                    del st.session_state[key]
            st.session_state.step = "input"
            st.rerun()

