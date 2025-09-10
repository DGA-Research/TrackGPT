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

def format_timestamp(ms: int | float) -> str:
    total_seconds = int(round(ms / 1000.0))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def _bucket_words_to_segments(words: list[dict], bucket_ms: int = 8000) -> list[dict]:
    """
    When there are no utterances/paragraphs, turn word-level timings into
    ~8s segments with start/end and text. words: [{'start':ms,'end':ms,'text':str}, ...]
    """
    if not words:
        return []
    segs, cur, cur_start = [], [], words[0]["start"]
    for w in words:
        cur.append(w["text"])
        if (w["end"] - cur_start) >= bucket_ms:
            segs.append({"start": cur_start, "end": w["end"], "text": " ".join(cur)})
            cur, cur_start = [], w["end"]
    if cur:
        segs.append({"start": cur_start, "end": words[-1]["end"], "text": " ".join(cur)})
    return segs
    
def transcribe_file(audio_file_path: str, openai_key: str, assemblyai_key: str, speaker_hint: str | None) -> str:
    """
    Returns a timecoded transcript as plain text:
        [HH:MM:SS] Speaker N: text

    - Always includes timestamps and spacing.
    - Only calls GPT to guess names when multiple speakers exist.
    """
    # 1) AssemblyAI setup with robust timing outputs
    aai.settings.api_key = assemblyai_key
    config = aai.TranscriptionConfig(
        speaker_labels=True,     # diarization
        paragraphs=True,         # paragraph timing
        punctuate=True,
        disfluencies=True,
        enable_words=True        # word timings (fallback)
    )

    # 2) Transcribe
    transcript = aai.Transcriber().transcribe(audio_file_path, config)

    # 3) Build consistent segments with start/end/text + speaker_id
    segments: list[dict] = []   # [{'start':ms,'end':ms,'text':str,'spk':int}, ...]
    unique_speakers = set()

    # Prefer utterances (speaker-labelled)
    if transcript.utterances:
        for u in transcript.utterances:
            unique_speakers.add(u.speaker)
            dur = u.end - u.start
            if dur <= 30000:  # <= 30s: keep as one segment
                segments.append({"start": u.start, "end": u.end, "text": (u.text or "").strip(), "spk": u.speaker})
            else:
                # break long utterances into ~30s chunks based on word rate
                words = (u.text or "").split()
                if not words:
                    continue
                wpm = max(1, len(words)) / max(1, dur)  # words per ms
                step = max(1, int(wpm * 30000))         # words per 30s
                t0 = u.start
                for i in range(0, len(words), step):
                    chunk = " ".join(words[i:i+step]).strip()
                    if not chunk:
                        continue
                    segments.append({"start": t0, "end": min(u.end, t0 + 30000), "text": chunk, "spk": u.speaker})
                    t0 += 30000
    else:
        # No utterances? Try paragraphs with timings
        paras = (getattr(transcript, "paragraphs", None) or {}).get("paragraphs", []) if hasattr(transcript, "paragraphs") else []
        if paras:
            # No speaker IDs here; assume single speaker '0'
            unique_speakers = {0}
            for p in paras:
                txt = (p.get("text") or "").strip()
                if not txt:
                    continue
                segments.append({"start": p.get("start", 0), "end": p.get("end", 0), "text": txt, "spk": 0})
        else:
            # Last resort: bucket words into segments
            words = getattr(transcript, "words", None) or []
            segments_raw = _bucket_words_to_segments(
                [{"start": w.start, "end": w.end, "text": w.text} for w in words]
            )
            unique_speakers = {0}
            for s in segments_raw:
                segments.append({"start": s["start"], "end": s["end"], "text": s["text"], "spk": 0})

    # 4) Render baseline timecoded transcript (Speaker N) — this is the canonical output
    lines: list[str] = []
    for seg in segments:
        if not seg["text"]:
            continue
        ts = format_timestamp(seg["start"])
        spk = seg["spk"]
        lines.append(f"[{ts}] Speaker {spk}: {seg['text']}")
        lines.append("")  # blank separator

    base_text = "\n".join(lines).strip()

    # 5) If only one speaker, skip GPT — keep timestamps/spacing intact
    if len(unique_speakers) <= 1:
        return base_text

    # 6) Multi-speaker: run a *strictly preserving* GPT pass to add **names in parentheses** only
    client = openai.OpenAI(api_key=openai_key)

    system_prompt = f"""
You are a transcription assistant. You must preserve the input EXACTLY:
- Do NOT change or remove timestamps like [HH:MM:SS].
- Do NOT merge, split, reorder, or wrap lines.
- Do NOT remove blank lines.
- Do NOT alter any text after the colon.

Task: only append a guessed human-readable name in parentheses immediately after the "Speaker X" tag.

Examples:
  Input line:  [00:00:03] Speaker 1: Thank you for coming.
  Output line: [00:00:03] Speaker 1 (Donald Trump): Thank you for coming.

Rules:
- If unsure, write (Unknown).
- If you see a likely name, use it once consistently for that Speaker.
- Keep "Speaker X" numbering the same as input.
- Do NOT add new speakers.
- Do NOT add or remove any other characters.

You may consider the hint: {speaker_hint or 'None'}.
    """.strip()

    # The user message is the whole base_text; the model must only add names in parentheses.
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_text}
        ],
        temperature=0.0
    )
    # In case the model deviates, fall back to baseline
    out = (resp.choices[0].message.content or "").strip()
    if not out or "[" not in out or "Speaker " not in out:
        return base_text
    return out


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





