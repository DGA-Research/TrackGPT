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
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours}:{minutes:02}:{seconds:02}"  # Always show HH:MM:SS

def transcribe_file(audio_file_path, openai_key, assemblyai_key, speaker):
    aai.settings.api_key = assemblyai_key  # replace with your actual key

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
            words_per_ms = total_words / duration if duration else 0

            # Calculate how many words fit in 30 seconds
            words_per_30_sec = int(words_per_ms * 30000) if words_per_ms > 0 else total_words

            # Split text into chunks
            chunk_start_time = utterance.start

            for i in range(0, total_words, max(1, words_per_30_sec)):
                chunk_words = words[i:i + max(1, words_per_30_sec)]
                chunk_text = " ".join(chunk_words)

                timestamp = format_timestamp(chunk_start_time)
                lines.append(f"[{timestamp}] Speaker {utterance.speaker}: {chunk_text}")

                # Update start time for next chunk
                chunk_start_time += 30000  # Add 30 seconds

    transcript_text = "\n".join(lines)

    # Set your OpenAI API key
    client = openai.OpenAI(api_key=openai_key)

    # Input your transcript
    # (Keep the variable name distinct from the AssemblyAI transcript object)
    system_prompt = f"""
        You are a transcription assistant. Given a monologue-style transcript of a conversation or interview, your task is to assign speaker labels (e.g., A, B, C...) and make a guess who is talking (e.g. Speaker A (Barack Obama):). Place the speaker labels before each line as clearly as possible.

        If there are multiple uknown speakers, differentiate them: Speaker A, Speaker B, etc. 
            - Correct Example: Speaker A (Unknown A), Speaker B (Mary) Speaker C (Unknown B)
            - Incorrect Example: Speaker A (Unknown), Speaker B (Mary), Speaker C (Unknown)
        
        Don't delete anything, just add guesses. Consider the spelling of {speaker}.

        Only add labels — DO NOT rephrase or summarize anything.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": transcript_text},
        ],
        temperature=0.2,
    )

    print("RETURNING")
    print("LABELED TRANSCRIPT BY CHAT", response.choices[0].message.content)
    # return the labeled transcript
    return response.choices[0].message.content

def _transcribe_large_file(audio_path: str, model: str, overlap_seconds: int, file_size: int) -> str:
    """Handle transcription of large audio files by splitting into chunks."""
    ...
