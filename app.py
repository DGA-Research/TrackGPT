import streamlit as st
from pathlib import Path
from datetime import datetime
import re
import hmac
from docx import Document
import io
from html2docx import html2docx

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
    from output import generate_html_report_bullets, generate_docx_report
    
    # UI layout
    st.set_page_config(page_title="TrackGPT", layout="centered")
    st.title("TrackGPT: Tracking Report Tool")
    url = "https://docs.google.com/document/d/1SR45h_w20Vn1-KrCRfAfkf2E2-aDvH-mXu8S2eA4630/edit?usp=sharing"
    st.markdown("Questions? Check out the [TrackGPT Instructions](%s)" % url)
    
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
    
    # Restart button
    if st.button("🔄 Restart"):
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
        
        # Input options
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
            date_input = st.text_input("Enter upload date: (MM/DD/YYYY)")
            uploaded_file = uploaded_file_mp3 or uploaded_file_m4a or uploaded_file_mp4
            
        transcript_button = st.checkbox("Enter my own transcript file")
        if transcript_button:
            transcript_input = st.text_area("Copy and paste transcript here", key="transcript_input")
            
        video_url = st.text_input("Enter a video or audio URL. See [Supported Sources](%s)" % url)
        
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
        
        # Report type selection
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
                    # Handle different input types
                    if video_url:
                        audio_str, metadata_update = download_audio(video_url, output_dir, base_filename, type_input)
                        st.session_state.metadata.update(metadata_update)
                        audio_path = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"
                        st.session_state.audio_path = str(audio_path)
                    elif uploaded_file:
                        audio_str = uploaded_file
                        st.session_state.audio_path = uploaded_file
                    else:
                        audio_str = None
                        st.session_state.audio_path = None
                    
                    # Get transcript
                    if transcript_input:
                        transcript = transcript_input
                    else:
                        transcript = transcribe_file(audio_str, OPENAI_API_KEY, ASSEMBLYAI_API_KEY, target_name)
                    
                    # Format transcript for HTML
                    transcript = re.sub(r'(\[\d+:\d+:\d+\.\d+\] Speaker [A-Z])', r'</p><p>\1', transcript)
                    transcript = '<p>' + transcript.strip() + '</p>'
                   
                    pattern = r'\[[\d:.]+\]\s+(Speaker\s+[A-Z])\s+\(([^)]+)\):'
    
                    # Find all matches
                    matches = re.findall(pattern, transcript)
                    
                    # Use set to automatically handle duplicates
                    unique_speakers = set()
                    
                    # Format each speaker and add to set (duplicates automatically ignored)
                    for speaker_id, name in matches:
                        formatted_speaker = f"{speaker_id}: {name}"
                        unique_speakers.add(formatted_speaker)
                    
                    # Convert to sorted list for consistent output
                    speaker_list = sorted(list(unique_speakers))
                
                    st.session_state.speaker_list = speaker_list
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
        
        # Edit transcript
        edited_transcript = st.text_area(
            "Edit Transcript:",
            value=st.session_state.transcript.replace('<p>', '').replace('</p>', '\n\n'),
            height=400
        )

        # Confirm Speaker
        speaker_text = ""
        for speaker in st.session_state.speaker_list:
            speaker_text = speaker + '\n' + speaker_text

        edited_speaker = st.text_area(
            "Edit Speakers:",
            value=speaker_text,
            height=100
        )

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Report →"):
                # Update transcript with edits
                transcript = re.sub(r'(\[\d+:\d+:\d+\.\d+\] Speaker [A-Z])', r'</p><p>\1', edited_transcript)
                transcript = '<p>' + transcript.strip() + '</p>'
                # edit transcript with new speakers
                pattern = r'(Speaker\s+[A-Z])\s+\(([^)]+)\):'
                # Make list of edited speakers
                matches = re.findall(pattern, edited_speaker)
                st.write("edited_speakers" + edited_speaker)
                unique_speakers = set()
                for speaker_id, name in matches:
                    unique_speakers.add(name)
                speaker_list_edited = sorted(list(unique_speakers))
                counter = 0
                print("speaker_list_edited", speaker_list_edited)
                print("st.session_state.speaker_list", st.session_state.speaker_list)
                for counter, item in enumerate(st.session_state.speaker_list):
                    escaped_item = re.escape(item)
                    transcript = re.sub(escaped_item, speaker_list_edited[counter], transcript)
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
        report_path = output_dir / f"{base_filename}_report.html"
        
        try:
            if st.session_state.report_type == "highlights":
                with st.spinner("Writing Highlights..."):
                    bullets = extract_raw_bullet_data_from_text(
                        st.session_state.transcript, 
                        st.session_state.target_name, 
                        st.session_state.metadata, 
                        OPENAI_API_KEY, 
                        "format_text_highlight_prompt"
                    )
                
                with st.spinner("Formatting Report..."):
                    html = generate_html_report(
                        st.session_state.metadata, 
                        bullets, 
                        st.session_state.transcript, 
                        st.session_state.target_name
                    )
                    
            elif st.session_state.report_type == "bullets":
                with st.spinner("Writing Bullets..."):
                    bullets = extract_raw_bullet_data_from_text(
                        st.session_state.transcript, 
                        st.session_state.target_name, 
                        st.session_state.metadata, 
                        OPENAI_API_KEY, 
                        "format_text_bullet_prompt"
                    )
                
                with st.spinner("Formatting Report..."):
                    html = generate_html_report_bullets(
                        st.session_state.metadata, 
                        bullets, 
                        st.session_state.transcript, 
                        st.session_state.target_name
                    )
                    
            else:  # transcript_only
                with st.spinner("Formatting Transcript..."):
                    html = f"<h2>{st.session_state.target_name} Transcript</h2>" + st.session_state.transcript
            
            # Store results
            st.session_state.html_report = html
            save_text_file(html, report_path)
            
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
            if st.button("← Back to Edit Transcript"):
                st.session_state.step = "edit_transcript"
                st.rerun()
    
    # STEP 4: SHOW RESULTS
    elif st.session_state.step == "show_results":
        st.success("✅ Analysis complete!")
        
        # Show the report
        st.markdown(st.session_state.html_report, unsafe_allow_html=True)
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📄 Download HTML Report",
                data=st.session_state.html_report,
                file_name=f"{st.session_state.target_name}_report.html",
                mime="text/html"
            )
        
        with col2:
            if st.session_state.report_type in ['highlights', 'bullets']:
                docx_buffer = html2docx(st.session_state.html_report, title="Converted Document")
                st.download_button(
                    label="📝 Download DOCX",
                    data=docx_buffer.getvalue(),
                    file_name=f"{st.session_state.target_name}_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        
        with col3:
            if hasattr(st.session_state, 'mp3_data') and st.session_state.mp3_data:
                st.download_button(
                    "🎵 Download Audio File",
                    data=st.session_state.mp3_data,
                    file_name=f"{st.session_state.target_name}_audio.mp3",
                    mime="audio/mpeg"
                )
        
        # Option to start over
        if st.button("🔄 Create Another Report"):
            for key in list(st.session_state.keys()):
                if key != "password_correct":
                    del st.session_state[key]
            st.session_state.step = "input"
            st.rerun()
