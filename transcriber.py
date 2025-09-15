import os
import math
import subprocess
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
import string

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE_LIMIT = 24 * 1024 * 1024  # 24 MB
DEFAULT_OVERLAP_SECONDS = 2


def _clean_hint_name(hint: str | None) -> str | None:
    """Pick a simple display name from speaker_hint like 'Donald Trump; Charles Payne'."""
    if not hint:
        return None
    # take first non-empty token split on ; , / ' and ' etc.
    parts = re.split(r"[;,/]| and | with | vs ", hint, flags=re.IGNORECASE)
    for p in parts:
        name = p.strip()
        if name:
            # collapse inner whitespace
            return re.sub(r"\s+", " ", name)
    return None



def format_timestamp(ms: int | float) -> str:
    total_seconds = int(round(ms / 1000.0))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def _bucket_words_to_segments(words, bucket_ms: int = 8000) -> list[dict]:
    """
    When only word timings exist, group them into ~8s segments with start/end/text.
    'words' items have .start, .end, .text
    """
    words = words or []
    if not words:
        return []
    segs, cur, cur_start = [], [], words[0].start
    for w in words:
        cur.append(w.text)
        if (w.end - cur_start) >= bucket_ms:
            segs.append({"start": cur_start, "end": w.end, "text": " ".join(cur)})
            cur, cur_start = [], w.end
    if cur:
        segs.append({"start": cur_start, "end": words[-1].end, "text": " ".join(cur)})
    return segs


def transcribe_file(audio_file_path: str, openai_key: str, assemblyai_key: str, speaker_hint: str | None) -> str:
    """
    Returns a timecoded transcript as plain text:
        [HH:MM:SS] Speaker N: text

    - Always includes timestamps & spacing
    - Uses diarization from AssemblyAI
    - If multiple speakers: asks GPT to append name guesses in parentheses
      while STRICTLY preserving timestamps, speaker numbering, and line breaks
    """
    # ---- AssemblyAI transcription ----
    aai.settings.api_key = assemblyai_key
    config = aai.TranscriptionConfig(
        speaker_labels=True,   # diarization
        punctuate=True,
        disfluencies=True,
        # AssemblyAI SDK returns words by default in recent versions
    )
    transcript = aai.Transcriber().transcribe(audio_file_path, config)

    # ---- Build segments [{'start','end','text','spk'}] ----
    segments: list[dict] = []
    unique_speakers: set[int] = set()

    # Preferred path: utterances (have speaker id + timings)
    if getattr(transcript, "utterances", None):
        for u in transcript.utterances:
            txt = (u.text or "").strip()
            if not txt:
                continue
            unique_speakers.add(u.speaker)
            dur = u.end - u.start
            if dur <= 30000:  # <= 30s keep intact
                segments.append({"start": u.start, "end": u.end, "text": txt, "spk": u.speaker})
            else:
                # break long utterances into ~30s chunks by density
                words = txt.split()
                if not words:
                    continue
                w_per_ms = max(1, len(words)) / max(1, dur)
                step = max(1, int(w_per_ms * 30000))  # words per ~30s
                t0 = u.start
                for i in range(0, len(words), step):
                    chunk = " ".join(words[i:i+step]).strip()
                    if chunk:
                        segments.append({"start": t0, "end": min(u.end, t0 + 30000), "text": chunk, "spk": u.speaker})
                        t0 += 30000

    # Fallback: paragraphs (if SDK exposes them)
    if not segments:
        paras = []
        try:
            resp = transcript.get_paragraphs()
            paras = getattr(resp, "paragraphs", []) or []
        except Exception:
            paras = []
        if paras:
            unique_speakers = {0}
            for p in paras:
                txt = (p.text or "").strip()
                if not txt:
                    continue
                segments.append({"start": p.start, "end": p.end, "text": txt, "spk": 0})

    # Last resort: bucket words (~8s)
    if not segments:
        words = getattr(transcript, "words", None) or []
        segs = _bucket_words_to_segments(words, bucket_ms=8000)
        unique_speakers = {0}
        for s in segs:
            segments.append({"start": s["start"], "end": s["end"], "text": s["text"], "spk": 0})

    # If still nothing, return raw text to avoid crashing
    if not segments:
        logging.error("No segments produced from utterances, paragraphs, or words.")
        return (transcript.text or "").strip()

    # ---- Normalize speaker ids to letter labels (A, B, C, ...) ----
    # Build a stable mapping so the same numeric diarization id always becomes the same letter
    speaker_list = sorted(list(unique_speakers), key=lambda x: (isinstance(x, str), x))
    label_map = {}
    for i, spk in enumerate(speaker_list):
        label_map[spk] = string.ascii_uppercase[i] if i < 26 else f"S{i+1}"  # A..Z then S27, S28...

    # Also rewrite segments to carry the label alongside the raw id
    for s in segments:
        s["spk_label"] = label_map.get(s["spk"], str(s["spk"]))
    
    # ---- Render baseline (canonical) ----
    lines: list[str] = []
    for seg in segments:
        text = (seg["text"] or "").strip()
        if not text:
            continue
        ts = format_timestamp(seg["start"])
        spk_label = seg["spk_label"]          # <-- use letter label
        lines.append(f"[{ts}] Speaker {spk_label}: {text}")
        lines.append("")  # blank line between segments
    base_text = "\n".join(lines).strip()



    # First non-blank line (often the host intro)
    first_talking_line = next((ln for ln in base_text.splitlines() if ln.strip()), "")

    looks_like_host_intro = bool(re.search(
        r"\b(Joining us now|We('?| a)re joined by|Welcome back|Joining me|Now to|We have with us)\b",
        first_talking_line,
        re.I,
    ))

    # Names explicitly mentioned in the intro line (likely guests, not the current speaker)
    mentioned_in_intro = re.findall(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
        first_talking_line,
    )
    # crude filter of non-names you saw in practice; tailor as needed
    blocklist = {"fox", "news", "turning", "point", "usa", "founder", "host", "commentator"}
    mentioned_in_intro = [
        n for n in mentioned_in_intro
        if len(n.split()) <= 3 and n.lower() not in blocklist
    ]

    # Names addressed like "Well, Liz," in later lines (helps pick the host)
    addressed_names = re.findall(r"\bWell,\s+([A-Z][a-zA-Z]+)\b", base_text)
    addressed_names = list(dict.fromkeys(addressed_names))  # dedupe

    # ---- If single speaker, append a deterministic name so UI can parse ----
    if len(unique_speakers) == 1:
        single_label = segments[0]["spk_label"]  # e.g., "A" or "1"
        display_name = _clean_hint_name(speaker_hint) or "Unknown"

        # Transform every "Speaker {label}:" to "Speaker {label} (Name):"
        # Keep timestamps, text, and blank lines exactly as-is.
        out_lines = []
        pat = re.compile(rf"^(\[\d{{2}}:\d{{2}}:\d{{2}}\]\s+Speaker\s+{re.escape(single_label)})(:)")
        for line in base_text.splitlines():
            if not line.strip():
                out_lines.append(line)
                continue
            m = pat.match(line)
            if m:
                out_lines.append(f"{m.group(1)} ({display_name}){m.group(2)}{line[m.end():]}")
            else:
                # Non-dialog lines (shouldn't occur, but be robust)
                out_lines.append(line)
        return "\n".join(out_lines)
        

    # ---- Build tiny "voice profiles" per speaker to help accurate naming ----
    by_label: dict[str, list[str]] = {}
    for seg in segments:
        lab = seg["spk_label"]
        by_label.setdefault(lab, [])
        if len(by_label[lab]) < 2 and len(seg["text"].split()) >= 6:
            ts = format_timestamp(seg["start"])
            by_label[lab].append(f"[{ts}] {seg['text']}")

    profiles = []
    for lab in sorted(by_label.keys()):
        sample_text = "\n".join(by_label[lab]) if by_label[lab] else "(no sample)"
        profiles.append(f"Speaker {lab} samples:\n{sample_text}")
    profile_block = "\n\n".join(profiles) if profiles else "(no samples)"



    # pull capitalized tokens followed by comma (e.g., "Liz,")
    comma_names = re.findall(r"\b([A-Z][a-zA-Z]+),", base_text)

    auto_candidates = mentioned_in_intro + addressed_names + comma_names

    # merge with user-provided candidates
    if speaker_hint:
        for tok in re.split(r"[;,/]| and | vs | with ", speaker_hint, flags=re.I):
            tok = tok.strip()
            if tok:
                auto_candidates.append(tok)

    # dedupe & trim
    candidates = list(dict.fromkeys(auto_candidates))[:8]
    candidate_block = ", ".join(candidates) if candidates else "None"
    role_hints = []
    if looks_like_host_intro:
        role_hints.append(
            "If the first line is an intro (e.g., 'Joining us now', 'We’re joined by'), "
            "that line is the HOST speaking. Any person named in that line is a different speaker (the guest)."
        )
    if mentioned_in_intro:
        role_hints.append(
            "Names explicitly mentioned in the intro line (likely guests): "
            + ", ".join(mentioned_in_intro) + "."
        )
    if addressed_names:
        role_hints.append(
            "If a line begins with 'Well, NAME,', NAME is being addressed (the other speaker), "
            "not the current line's speaker."
        )

    role_hint_block = "\n".join(role_hints) if role_hints else "No extra role hints."

    logging.debug("candidates=%s", candidate_block)
    logging.debug("role_hints=%s", role_hint_block)

    # ---- Multi-speaker: GPT name guessing with STRICT preservation ----
    client = openai.OpenAI(api_key=openai_key)

    system_prompt = f"""
You must preserve the input transcript EXACTLY:
- Do NOT change or remove timestamps like [HH:MM:SS].
- Do NOT merge, split, reorder, or wrap lines.
- Do NOT remove blank lines.
- Do NOT alter anything after the colon.

Task: ONLY append a guessed human-readable name in parentheses immediately after the 'Speaker X' tag on each line.

Example:
  Input:  [00:00:03] Speaker 1: Thank you for coming.
  Output: [00:00:03] Speaker 1 (Jane Doe): Thank you for coming.

Rules:
- Keep the exact token after "Speaker " unchanged (e.g., if input has Speaker A, do not change it to Speaker 1).
- If unsure, use (Unknown).
- Be consistent for the same Speaker across the whole file.
- Use these candidates if relevant: {candidate_block}

Additional disambiguation hints (follow strictly):
{role_hint_block}

Use these short samples to inform your guesses (do not copy these into the output):
-----
{profile_block}
-----
""".strip()
    logging.debug("GPT candidate_block=%r", candidate_block)
    logging.debug("GPT role_hint_block=%r", role_hint_block)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_text}
            ],
            temperature=0.0
        )
        out = (resp.choices[0].message.content or "").strip()

        # ---- Safety validation with allowed relabeling (bijection mapping) ----
        base_lines = base_text.splitlines()
        out_lines = out.splitlines()
        if len(base_lines) != len(out_lines):
            logging.warning("Name guessing changed line count; returning baseline.")
            return base_text

        ts_pat = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+Speaker\s+(\S+):\s*$")
        ts_pat_named = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+Speaker\s+(\S+)\s*(\([^)]+\))?\s*:\s*$")


        label_map_in2out: dict[str, str] = {}
        label_map_out2in: dict[str, str] = {}

        for i, (a, b) in enumerate(zip(base_lines, out_lines)):
            if not a.strip():   # blank line must remain blank
                if b.strip() != "":
                    logging.warning(f"Line {i+1}: expected blank line; returning baseline.")
                    return base_text
                continue

            ma = ts_pat.match(a)
            mb = ts_pat_named.match(b)
            if not ma or not mb:
                logging.warning(f"Line {i+1}: timestamp/speaker tag mismatch; returning baseline.")
                return base_text

            # timestamps must be identical
            if ma.groups()[:3] != mb.groups()[:3]:
                logging.warning(f"Line {i+1}: timestamp changed; returning baseline.")
                return base_text

            in_label = ma.group(4)   # baseline Speaker token (e.g., A)
            out_label = mb.group(4)  # output Speaker token (e.g., B)

            # Establish or verify bijection mapping
            if in_label not in label_map_in2out:
                if out_label in label_map_out2in and label_map_out2in[out_label] != in_label:
                    logging.warning(f"Line {i+1}: non-bijective relabeling; returning baseline.")
                    return base_text
                label_map_in2out[in_label] = out_label
                label_map_out2in[out_label] = in_label
            else:
                if label_map_in2out[in_label] != out_label:
                    logging.warning(f"Line {i+1}: inconsistent relabeling; returning baseline.")
                    return base_text

        # All good — keep GPT output with appended names (and possibly swapped labels)
        return out or base_text

    except Exception as e:
        logging.warning(f"Name guessing step failed; returning baseline. Error: {e}")
        return base_text





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












