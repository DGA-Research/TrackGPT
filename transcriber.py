# transcriber.py (tidied)
from __future__ import annotations

import logging
import re
from typing import List
import assemblyai as aai
import openai

logger = logging.getLogger(__name__)

def format_timestamp(ms: int) -> str:
    total_seconds = ms / 1000
    hours   = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def transcribe_file(audio_file_path: str, openai_key: str, assemblyai_key: str, speaker_hint: str) -> str:
    """
    1) Use AssemblyAI with speaker_labels to get utterances.
    2) Break any single utterance >30s into ~30s chunks (safe step size).
    3) If there’s only 1 speaker, append the hint to each line locally (fast path).
    4) Otherwise, ask OpenAI only to append a human-readable name after 'Speaker X'
       while preserving every character of the transcript (timestamps, order, etc.).
    """
    # --- AAI transcription ---
    aai.settings.api_key = assemblyai_key
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcript = aai.Transcriber().transcribe(audio_file_path, config)

    lines: List[str] = []
    for utt in transcript.utterances:
        duration_ms = max(utt.end - utt.start, 1)  # guard against zero/negative
        if duration_ms <= 30_000:
            timestamp = format_timestamp(utt.start)
            lines.append(f"[{timestamp}] Speaker {utt.speaker}: {utt.text}")
            continue

        # Split long utterances into ~30s chunks without step=0
        words = utt.text.split()
        total_words = len(words)
        words_per_ms = (total_words / duration_ms) if duration_ms > 0 else total_words
        words_per_30s = max(1, int(words_per_ms * 30_000))  # <-- clamp to >= 1

        chunk_start = utt.start
        for i in range(0, total_words, words_per_30s):
            chunk_text = " ".join(words[i:i + words_per_30s]).strip()
            if not chunk_text:
                break
            timestamp = format_timestamp(chunk_start)
            lines.append(f"[{timestamp}] Speaker {utt.speaker}: {chunk_text}")
            chunk_start += 30_000

    raw = "\n".join(lines)

    # --- Single-speaker fast path: append hint locally ---
    try:
        spk_ids = {u.speaker for u in (transcript.utterances or [])}
    except Exception:
        spk_ids = set()

    if len(spk_ids) == 1 and raw.strip():
        display = (speaker_hint or "Unknown").strip()
        # append " (Name)" immediately after 'Speaker X'
        pat = re.compile(r"^(\[\d{1,2}:\d{2}:\d{2}\]\s+Speaker\s+\S+)(:)", re.M)
        out_lines = []
        for line in raw.splitlines():
            m = pat.match(line)
            if m:
                out_lines.append(f"{m.group(1)} ({display}){m.group(2)}{line[m.end():]}")
            else:
                out_lines.append(line)
        return "\n".join(out_lines)

    # --- Multi-speaker: ask OpenAI to append readable names only ---
    client = openai.OpenAI(api_key=openai_key)
    system_prompt = (
        "You must preserve the input transcript EXACTLY:\n"
        "- Do NOT change or remove timestamps like [H:MM:SS] or [HH:MM:SS].\n"
        "- Do NOT merge, split, reorder, or wrap lines.\n"
        "- Do NOT remove blank lines.\n"
        "- Do NOT alter anything after the colon.\n\n"
        "Task: ONLY append a guessed human-readable name in parentheses immediately "
        "after the 'Speaker X' tag on each line.\n\n"
        "Example:\n"
        "  Input:  [00:00:03] Speaker A: Thank you for coming.\n"
        "  Output: [00:00:03] Speaker A (Jane Doe): Thank you for coming.\n\n"
        "Rules:\n"
        "- Keep the exact token after 'Speaker ' unchanged (e.g., 'Speaker 0' stays 'Speaker 0').\n"
        "- If unsure, use (Unknown).\n"
        "- Be consistent for the same Speaker across the whole file.\n"
        f"- Consider the spelling of {speaker_hint}.\n"
    ).strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
