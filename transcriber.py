import os
import math
import subprocess
import logging
import openai
import time
import sys
from pathlib import Path
import tempfile
import assemblyai as aai
import logging
import json
from typing import Optional, List, Dict, Any
from config import Config
import re

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE_LIMIT = 24 * 1024 * 1024  # 24 MB
DEFAULT_OVERLAP_SECONDS = 2

def format_timestamp(ms):
    total_seconds = ms / 1000
    hours   = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def transcribe_file(audio_file_path, openai_key, assemblyai_key, speaker):
    aai.settings.api_key=assemblyai_key # replace with your actual key

    audio_file = audio_file_path

    config = aai.TranscriptionConfig(
        speaker_labels=True,
    )

    transcript = aai.Transcriber().transcribe(audio_file, config)

    lines = []

    for utterance in transcript.utterances:
        duration = utterance.end - utterance.start
        
        # If utterance is 30 seconds or less, keep as is
        if duration <= 30000:  # 30 seconds in milliseconds
            timestamp = format_timestamp(utterance.start)
            lines.append(f"[{timestamp}] Speaker {utterance.speaker}: {utterance.text}")
        else:
            # Break up long utterances into 30-second chunks
            text = utterance.text
            words = text.split()
            total_words = len(words)
            
            # Calculate words per millisecond
            words_per_ms = total_words / duration
            
            # Calculate how many words fit in 30 seconds
            words_per_30_sec = int(words_per_ms * 30000)
            
            # Split text into chunks
            chunk_start_time = utterance.start
            
            for i in range(0, total_words, words_per_30_sec):
                chunk_words = words[i:i + words_per_30_sec]
                chunk_text = " ".join(chunk_words)
                
                timestamp = format_timestamp(chunk_start_time)
                lines.append(f"[{timestamp}] Speaker {utterance.speaker}: {chunk_text}")
                
                # Update start time for next chunk
                chunk_start_time += 30000  # Add 30 seconds
    
    lines1 = "\n".join(lines)

    
    # --- Single-speaker fast path: preserve timestamps exactly, skip GPT ---
    try:
        spk_ids = {u.speaker for u in transcript.utterances or []}
    except Exception:
        spk_ids = set()

    if len(spk_ids) == 1 and lines1.strip():
        display_name = (speaker or "Unknown").strip()
        # Append a guessed display name right after the 'Speaker X' token, keep everything else identical
        import re
        fast_pat = re.compile(r"^(\[\d{1,2}:\d{2}:\d{2}\]\s+Speaker\s+\S+)(:)", re.M)
        out_lines = []
        for line in lines1.splitlines():
            m = fast_pat.match(line)
            if m:
                out_lines.append(f"{m.group(1)} ({display_name}){m.group(2)}{line[m.end():]}")
            else:
                out_lines.append(line)
        return "\n".join(out_lines)
    
    # Set your OpenAI API key
    client = openai.OpenAI(
        api_key=openai_key)

    # Input your transcript
    transcript = lines1

    # System prompt for speaker labeling
    system_prompt = f"""
    You must preserve the input transcript EXACTLY:
    - Do NOT change or remove timestamps like [H:MM:SS] or [HH:MM:SS].
    - Do NOT merge, split, reorder, or wrap lines.
    - Do NOT remove blank lines.
    - Do NOT alter anything after the colon.

    Task: ONLY append a guessed human-readable name in parentheses immediately after the 'Speaker X' tag on each line.

    Example:
      Input:  [00:00:03] Speaker A: Thank you for coming.
      Output: [00:00:03] Speaker A (Jane Doe): Thank you for coming.

    Rules:
    - Keep the exact token after "Speaker " unchanged (e.g., if input has Speaker 0 or Speaker A, don’t rename it).
    - If unsure, use (Unknown).
    - Be consistent for the same Speaker across the whole file.
    - Consider the spelling of {speaker}.
    """.strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ],
        temperature=0.2
    )

    print("RETURNING")
    print("LABELED TRANSCRIPT BY CHAT", response.choices[0].message.content)
    # return the labeled transcript
    return(response.choices[0].message.content)



def _clean_hint_name(hint: str | None) -> str | None:
    """Pick a simple display name from speaker_hint like 'Donald Trump; Charles Payne'."""
    if not hint:
        return None
    parts = re.split(r"[;,/]| and | with | vs ", hint, flags=re.IGNORECASE)
    for p in parts:
        name = p.strip()
        if name:
            return re.sub(r"\s+", " ", name)
    return None


def _transcribe_large_file(audio_path: str, model: str, overlap_seconds: int, file_size: int) -> str:
    """Handle transcription of large audio files by splitting into chunks."""
    try:
        # Get total duration using ffprobe
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ]
        total_duration = float(subprocess.check_output(duration_cmd).decode().strip())
        logger.info(f"Total audio duration: {total_duration} seconds")
    except Exception as e:
        logger.error(f"Failed to get audio duration: {str(e)}")
        raise RuntimeError(f"Failed to get audio duration: {str(e)}")
    
    # Calculate chunk parameters
    avg_bitrate = (file_size * 8) / total_duration
    max_chunk_duration = (CHUNK_SIZE_LIMIT * 8) / avg_bitrate * 0.95
    effective_chunk_duration = max(max_chunk_duration - overlap_seconds, 0.1)
    num_chunks = math.ceil(total_duration / effective_chunk_duration)
    
    logger.info(f"Processing {num_chunks} chunks with {overlap_seconds}s overlap")
    logger.debug(f"Max chunk duration: {max_chunk_duration}s, effective: {effective_chunk_duration}s")
    
    transcripts = []
    temp_files_to_delete = []
    
    try:
        for i in range(num_chunks):
            start_time = i * effective_chunk_duration
            if start_time >= total_duration:
                break
                
            end_time = min(total_duration, start_time + max_chunk_duration)
            chunk_path = _create_chunk_file(audio_path, start_time, end_time, i)
            temp_files_to_delete.append(chunk_path)
            
            # Verify chunk size
            chunk_size = os.path.getsize(chunk_path)
            logger.debug(f"Chunk {i+1}: {start_time:.2f}-{end_time:.2f}s, size: {chunk_size} bytes")
            if chunk_size > CHUNK_SIZE_LIMIT:
                logger.warning(f"Chunk {i+1} exceeds size limit: {chunk_size} bytes")
            
            # Transcribe chunk
            try:
                logger.info(f"Transcribing chunk {i+1}/{num_chunks}")
                with open(chunk_path, "rb") as chunk_file:
                    response = openai.audio.transcriptions.create(
                        model=model,
                        file=chunk_file
                    )
                transcripts.append(response.text)
                logger.debug(f"Chunk {i+1} transcription successful, length: {len(response.text)}")
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i+1}: {str(e)}")
                continue
                
    finally:
        _cleanup_temp_files(temp_files_to_delete)
    
    if not transcripts:
        raise RuntimeError("Transcription failed: no chunks could be transcribed successfully")
    
    logger.info(f"Transcription completed with {len(transcripts)}/{num_chunks} successful chunks")
    return " ".join(transcripts)

def _create_chunk_file(audio_path: str, start_time: float, end_time: float, index: int) -> Path:
    """Create a temporary chunk file using ffmpeg."""
    audio_path = Path(audio_path)
    chunk_path = Path(tempfile.gettempdir()) / f"{audio_path.stem}_part{index+1}{audio_path.suffix}"
    
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', str(audio_path),
            '-to', str(end_time),
            '-c', 'copy',
            str(chunk_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create chunk {index+1}: {str(e)}")
        raise RuntimeError(f"Failed to create audio chunk: {str(e)}")
    
    return chunk_path

def _cleanup_temp_files(file_paths: List[Path]):
    """Clean up temporary files with retry mechanism."""
    if not file_paths:
        return
        
    logger.info(f"Starting cleanup of {len(file_paths)} temporary files")
    failed_deletions = 0
    
    for temp_path in file_paths:
        if not temp_path.exists():
            continue
            
        max_retries = 3
        deleted = False
        
        for attempt in range(max_retries):
            try:
                temp_path.unlink()
                logger.debug(f"Deleted temporary file: {temp_path}")
                deleted = True
                break
            except PermissionError as e:
                if sys.platform == "win32" and attempt < max_retries - 1:
                    wait_time = 0.5 * (attempt + 1)
                    logger.warning(
                        f"PermissionError deleting {temp_path}, retry {attempt+1}/{max_retries} in {wait_time}s"
                    )
                    time.sleep(wait_time)
                    continue
                logger.error(f"Failed to delete {temp_path}: {str(e)}")
                failed_deletions += 1
                break
            except FileNotFoundError:
                logger.debug(f"File already deleted: {temp_path}")
                deleted = True
                break
            except Exception as e:
                logger.error(f"Failed to delete {temp_path}: {str(e)}")
                failed_deletions += 1
                break
    
    if failed_deletions:
        logger.error(f"Failed to delete {failed_deletions} temporary files")
    else:
        logger.info("All temporary files cleaned up successfully")




