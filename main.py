"""
Main script for audio processing pipeline:
1. Downloads audio from URL using yt-dlp
2. Transcribes audio using OpenAI Whisper API
3. Extracts key bullet points from transcript
4. Generates HTML report with analysis

Usage: python main.py "URL" "Target Name" [options]
"""
import argparse
import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import re

# --- Initialize logging ---
# Moved logging setup to the top to capture logs from module imports if needed
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Set logger name for main script
log = logging.getLogger(__name__)

# --- Import project modules AFTER logging is set up ---
try:
    from config import Config # Already validated on import
    from downloader import download_audio
    from transcriber import transcribe_file
    # from analyzer import analyze_transcript # Keep if needed
    from analyzer import extract_raw_bullet_data_from_text # Import the V3 extractor
    # from output import format_report # Keep if needed
    from output import generate_html_report, save_text_file # Import HTML report generator
except ImportError as e:
    log.critical(f"Failed to import necessary module: {e}. Please ensure all files exist and dependencies are installed.", exc_info=True)
    sys.exit(1)
except Exception as e:
    log.critical(f"An unexpected error occurred during initial imports: {e}", exc_info=True)
    sys.exit(1)


# --- Main Orchestration ---
def main() -> None:
    """
    Main entry point for the video research pipeline.
    
    Orchestrates the full processing pipeline:
    1. Downloads audio from URL (unless skipped)
    2. Transcribes audio to text (unless skipped)
    3. Extracts key bullet points (unless skipped)
    4. Generates HTML report with analysis
    
    Args:
        None (uses command line arguments via argparse)
    
    Returns:
        None (exits with status code)
    
    Exit Codes:
        0: Success
        1: Partial success (some steps failed)
        2: Critical failure
    
    Error Handling:
        - Each stage has independent error handling
        - Critical errors stop the pipeline
        - Non-critical errors allow continuation with warnings
        - All errors are logged with context
    """
    log.info("--- Starting Process ---")

    parser = argparse.ArgumentParser(
        description="""Video Research Transcriber and Analyzer
        
        Processes video/audio content through a 4-stage pipeline:
        1. Download: Uses yt-dlp to fetch audio from URL
        2. Transcription: Converts audio to text using OpenAI Whisper
        3. Analysis: Extracts key points about target using GPT
        4. Reporting: Generates formatted HTML research report
        
        Each stage can be skipped if files exist from previous runs.
        """,
        epilog="""Examples:
        Basic usage:
          python main.py \"https://youtube.com/watch?v=123\" \"John Doe\"
        
        Skip download and transcription:
          python main.py \"existing.mp3\" \"Jane Smith\" --skip_download --skip_transcription
        
        Custom output directory:
          python main.py \"https://youtube.com/watch?v=123\" \"John Doe\" -o results
        """
    )
    parser.add_argument(
        "url",
        help="URL of the video/audio source (use quotes if needed)."
    )
    parser.add_argument(
        "target_name",
        help="Name of the research target (person/entity)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=Config.DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {Config.DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download step (requires existing audio file)."
    )
    parser.add_argument(
        "--skip_transcription",
        action="store_true",
        help="Skip transcription step (requires existing transcript file)."
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip bullet point extraction step."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging for more detailed output."
    )


    args = parser.parse_args()

    # --- Setup Logging Level ---
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.info("Verbose logging enabled.")

    # --- Prepare Paths ---
    output_dir = Path(args.output_dir).resolve() # Use absolute path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize target name for filename
    safe_target_name = "".join(c if c.isalnum() else "_" for c in args.target_name)
    base_filename = f"{safe_target_name}_{timestamp}"

    # edit this out to not use pre-downloaded stuff

    # audio_path = output_dir / f"Karrin_Taylor_Robson_20250610_112024.mp3"
    # transcript_path = output_dir / "KTR.txt"
    # analysis_path = output_dir / f"{base_filename}_analysis.txt"

    # this is the original code
    audio_path = output_dir / f"{base_filename}.{Config.AUDIO_FORMAT}"

    transcript_path = output_dir / f"{base_filename}_transcript.txt"
    analysis_path = output_dir / f"{base_filename}_analysis.txt"

    log.info(f"Processing URL: {args.url}")
    log.info(f"Target Name: {args.target_name}")
    log.info(f"Output Directory: {output_dir}")
    log.info(f"Audio File: {audio_path.name}")
    log.info(f"Transcript File: {transcript_path.name}")
    log.info(f"Analysis File: {analysis_path.name}")
    log.info(f"Skip Download: {args.skip_download}")
    log.info(f"Skip Transcription: {args.skip_transcription}")
    log.info(f"Skip Extraction: {args.skip_extraction}")

    # --- Execution Pipeline ---
    # Initialize pipeline state variables
    # Pipeline state variables
    current_audio_path_str: str | None = None  # Path to downloaded/input audio file
    video_metadata: Dict[str, Any] | None = None  # Metadata from video source
    transcript_text: str | None = None  # Full transcript text
    extracted_bullets_raw: Optional[List[Dict[str, Any]]] = None  # Raw bullet point data
    exit_code = 0  # Process exit status (0=success)

    try:
        # --- Step 1: Download Audio & Get Metadata ---
        if not args.skip_download:
            log.info("--- Step 1: Downloading Audio & Metadata ---")
            # Expect a tuple: (path_str, metadata_dict) or None
            download_result: Optional[Tuple[str, Dict[str, Any]]] = download_audio(args.url, output_dir, base_filename)
            if download_result:
                current_audio_path_str, video_metadata = download_result # Unpack the tuple
                log.info(f"Audio downloaded to: {current_audio_path_str}")
                log.info(f"Metadata obtained: Title='{video_metadata.get('title', 'N/A')}'")
            else:
                log.error("Audio download or metadata extraction failed. Cannot proceed.")
                raise RuntimeError("Audio download/metadata failed.")
        else:
            # Keep the existing skip_download logic, but ensure video_metadata is created
            log.info("--- Step 1: Skipping Audio Download ---")
            if audio_path.is_file():
                log.info(f"Using existing audio file: {audio_path}")
                current_audio_path_str = str(audio_path)
                # Create minimal metadata when skipping download
                video_metadata = {
                    'title': f"Existing file: {audio_path.name}",
                    'uploader': "Unknown (Download Skipped)",
                    'upload_date': None,
                    'webpage_url': "N/A",
                    'extractor': "Local file",
                }
            else:
                log.error(f"Skip download requested, but audio file not found: {audio_path}")
                raise FileNotFoundError(f"Required audio file missing for skipped download: {audio_path}")

        # --- Step 2: Transcribe Audio ---
        if not args.skip_transcription:
            log.info("--- Step 2: Transcribing Audio ---")
            if not current_audio_path_str:
                log.error("Cannot transcribe because audio path is not available.")
                raise RuntimeError("Audio path missing for transcription.")
            
            # Transcription process flow:
            # 1. Validate audio file exists and is readable
            # 2. Send to OpenAI Whisper API for transcription
            # 3. Handle API response and errors
            # 4. Save transcript to file

            transcript_text = transcribe_file(current_audio_path_str)
            if transcript_text:
                if not save_text_file(transcript_text, transcript_path):
                    log.warning("Failed to save transcript file, but continuing analysis.")
            else:
                log.error("Transcription failed. Cannot proceed with analysis.")
                raise RuntimeError("Transcription failed.")
        else:
            log.info("--- Step 2: Skipping Transcription ---")
            if transcript_path.is_file():
                log.info(f"Loading existing transcript from: {transcript_path}")
                try:
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript_text = f.read()
                        # added with chat
                        transcript_text = re.sub(r'(\[\d+:\d+:\d+\.\d+\] Speaker [A-Z])', r'</p><p>\1', transcript_text)
                        transcript_text = '<p>' + transcript_text.strip() + '</p>'

                    if not transcript_text or not transcript_text.strip():
                         log.warning(f"Loaded transcript file is empty: {transcript_path}")
                         # Decide whether to proceed or fail based on empty transcript
                         # raise RuntimeError("Cannot analyze empty transcript.") # Option to fail
                except Exception as e:
                    log.error(f"Failed to read existing transcript file {transcript_path}: {e}", exc_info=True)
                    raise # Re-raise file read error
            else:
                log.error(f"Skip transcription requested, but transcript file not found: {transcript_path}")
                raise FileNotFoundError(f"Required transcript file missing for skipped transcription: {transcript_path}")

        # --- Step 3: Extract Bullet Points ---
        extracted_bullets_raw: Optional[List[Dict[str, Any]]] = None # Store raw dicts
        if not args.skip_extraction:
            log.info("--- Step 3: Extracting Bullet Points ---")
            if transcript_text and video_metadata:
                try:
                    extracted_bullets_raw = extract_raw_bullet_data_from_text(transcript_text, args.target_name, video_metadata) # Call V3
                    if extracted_bullets_raw is None:
                         log.error("V3 Bullet extraction function failed critically.")
                         extracted_bullets_raw = []
                         exit_code = 1
                    else:
                         log.info(f"Extracted {len(extracted_bullets_raw)} raw bullet data points.")
                except Exception as e:
                    log.error(f"V3 Bullet extraction step failed: {e}", exc_info=True)
                    extracted_bullets_raw = []
                    exit_code = 1
            else:
                log.error("Cannot extract V3 bullets: Transcript or Metadata missing.")
                extracted_bullets_raw = []
        else:
             log.info("--- Step 3: Skipping Bullet Extraction ---")
             extracted_bullets_raw = []

        # --- Step 4: Format and Save Report (V3) ---
        log.info("--- Step 4: Formatting and Saving Report (V3) ---")
        report_path = output_dir / f"{base_filename}_report.html"
        log.info(f"Output report will be saved to: {report_path}")
        
        # Report generation workflow:
        # 1. Validate required inputs (metadata, bullets, transcript)
        # 2. Format HTML structure with:
        #    - Header with target name and timestamp
        #    - Metadata section
        #    - Bullet points with citations
        #    - Full transcript
        # 3. Apply responsive CSS styling
        # 4. Save to output file with error handling

        if video_metadata is None: video_metadata = {}; exit_code = 1; log.error("Metadata missing for report.")
        if transcript_text is None: transcript_text = "Transcript unavailable."; log.warning("Transcript missing for report.")

        try:
            # Call the V3 formatter, passing the RAW bullet data list
            final_report_content = generate_html_report(
                 video_metadata, extracted_bullets_raw or [], transcript_text, args.target_name
            )
        except Exception as e:
            log.error(f"Failed to format V3 report content: {e}", exc_info=True)
            final_report_content = f"Error formatting report: {e}" # Fallback
            exit_code = 1

        # Save the final report using save_text_file (no change needed here)
        if not save_text_file(final_report_content, report_path):
            log.error("Failed to save the final report file.")
            exit_code = 1
        else:
            log.info(f"Successfully saved report to: {report_path}")


    except FileNotFoundError as e:
        # Critical error - missing required files
        log.error(f"File not found: {e}")
        exit_code = 2
    except RuntimeError as e:
        # Pipeline execution error
        log.error(f"Process halted due to error: {e}")
        exit_code = 2
    except Exception as e:
        # Unexpected error - log full traceback
        log.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        traceback.print_exc()
        exit_code = 2
    finally:
        log.info("--- Process Finished ---")
        if exit_code == 0:
            log.info("Process completed successfully.")
            if transcript_path.exists():
                log.info(f"Transcript saved to: {transcript_path}")
            if report_path.exists():
                log.info(f"Report saved to: {report_path}")
        else:
            log.warning(f"Process finished with errors (exit code {exit_code}). Check logs above.")

        sys.exit(exit_code)


if __name__ == "__main__":
    main()
