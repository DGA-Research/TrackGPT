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
from typing import List
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
    parts = re.split(r"[;,/]| and | with | vs ", hint, flags=re.IGNORECASE)
    for p in parts:
        name = p.strip()
        if name:
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


# --- Name harvesting helpers (flexible, avoids generic tokens) ---

# Basic proper-name token: Smith, O'Neil, Mary-Jane
NAME = r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?"
# Intro full names must have at least two tokens (e.g., "Tomi Lahren", "Charles Payne")
INTRO_FULLNAME = rf"{NAME}\s+{NAME}(?:\s+{NAME})?"

# Stoplist of generic capitalized tokens common in TV intros / filler
BLOCKLIST = {
    "fox", "news", "turning", "point", "usa", "founder", "host",
    "contributor", "commentator",
    "joining", "welcome", "now", "tonight", "breaking", "live",
    "us", "me", "back", "thanks", "thank", "you", "okay", "ok",
    "alright", "right", "so", "look", "listen",
}

def _is_ok_name(s: str) -> bool:
    w = s.strip().lower()
    if w in BLOCKLIST:
        return False
    toks = [t.lower() for t in s.split()]
    if all(t in BLOCKLIST for t in toks):
        return False
    return True


# Scrub obviously wrong assigned names like "(Joining)" -> "(Unknown)"
BAD_NAME_RE = re.compile(
    r"^(?:"
    r"joining|welcome|now|tonight|breaking|live|thanks|thank you|thank|you|"
    r"okay|ok|alright|right|back|so|look|listen|us|me"
    r")$",
    re.IGNORECASE,
)

ADDRESS_RE = re.compile(r"\b(?:Well|Look|Listen|So),\s+([A-Z][a-zA-Z]+)\b")

def _collect_addressed_names_by_label(base_text: str) -> dict[str, list[str]]:
    """
    Parse the baseline transcript to find addressed names like 'Well, Liz,' and
    attribute them to the speaker label of that line.
    Returns: { label -> [name1, name2, ...] }
    """
    by_label: dict[str, list[str]] = {}
    line_pat = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+Speaker\s+(\S+):\s*(.*)$")
    for line in base_text.splitlines():
        m = line_pat.match(line)
        if not m:
            continue
        label = m.group(4)
        text  = m.group(5)
        for nm in ADDRESS_RE.findall(text):
            if _is_ok_name(nm):
                by_label.setdefault(label, []).append(nm)
    return by_label


def _force_other_label_name_when_addressed(base_text: str, out_text: str) -> str:
    """
    If exactly two speaker labels exist overall, and one label frequently addresses a single
    name (e.g., 'Well, Liz,'), assign that name to the OTHER label consistently.

    We only overwrite '(Unknown)' or scrubbed bad guesses for that other label.
    """
    # Find all labels present
    label_pat = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s+Speaker\s+(\S+):", re.M)
    labels = list(dict.fromkeys(label_pat.findall(base_text)))  # in order of appearance
    if len(labels) != 2:
        return out_text

    a, b = labels[0], labels[1]

    addressed = _collect_addressed_names_by_label(base_text)
    # Pick the dominant addressed name per label (if any)
    def top_name(names: list[str]) -> str | None:
        if not names:
            return None
        # frequency, then first appearance
        counts = {}
        order = []
        for n in names:
            if n not in counts:
                counts[n] = 0
                order.append(n)
            counts[n] += 1
        best = max(order, key=lambda n: (counts[n], -order.index(n)))
        return best

    name_addr_by_label = {lab: top_name(nms) for lab, nms in addressed.items() if nms}

    # If one label repeatedly addresses a single name N, map N to the OTHER label
    # Only act if we have a single confident addressed name
    # Case 1: B addresses N -> N is A
    # Case 2: A addresses N -> N is B
    forced: dict[str, str] = {}
    if b in name_addr_by_label and name_addr_by_label[b]:
        forced[a] = name_addr_by_label[b]
    if a in name_addr_by_label and name_addr_by_label[a]:
        # If both sides address someone, bail unless they agree (avoid conflicts)
        candidate = name_addr_by_label[a]
        if a in forced or b in forced:
            # Both sides set something; only keep if consistent
            if forced.get(a) and forced[a] != candidate:
                return out_text  # conflict -> do nothing
        else:
            force


def _scrub_bad_assigned_names(text: str) -> str:
    # Replace "(Joining)"-style names with "(Unknown)" but preserve everything else EXACTLY.
    pattern = re.compile(r"^(\[\d{2}:\d{2}:\d{2}\]\s+Speaker\s+\S+\s*\()([^)]+)(\):)", re.M)

    def repl(m: re.Match) -> str:
        name = m.group(2).strip()
        if BAD_NAME_RE.match(name):
            return f"{m.group(1)}Unknown{m.group(3)}"
        return m.group(0)

    return pattern.sub(repl, text)


def _postfix_single_unknown_with_single_candidate(out_text: str, candidates: list[str]) -> str:
    """
    If exactly one speaker label has '(Unknown)' and exactly one candidate name is unused,
    replace that Unknown with the candidate. Keeps everything else identical.
    """
    line_pat = re.compile(
        r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+Speaker\s+(\S+)\s*\(([^)]+)\):",
        re.M,
    )
    used_names = set()
    labels_with_unknown = set()

    for m in line_pat.finditer(out_text):
        label = m.group(4)
        name = m.group(5).strip()
        if name.lower() == "unknown":
            labels_with_unknown.add(label)
        else:
            used_names.add(name.lower())

    if len(labels_with_unknown) != 1:
        return out_text

    # pick a candidate that isn't already used and not blocked
    viable = [c for c in candidates if _is_ok_name(c) and c.lower() not in used_names]
    if len(viable) != 1:
        return out_text

    label_to_fix = next(iter(labels_with_unknown))
    chosen = viable[0]

    # Replace only "(Unknown)" for this specific label
    fix_pat = re.compile(
        rf"^(\[\d{{2}}:\d{{2}}:\d{{2}}\]\s+Speaker\s+{re.escape(label_to_fix)}\s*\()Unknown(\):)",
        re.M,
    )
    return fix_pat.sub(rf"\1{chosen}\2", out_text)

_ADDR_PAT = re.compile(
    r"\b(?:Well|Yes|No|Right|Thanks|Thank you|Look|Listen|Okay|Ok|So|Alright),\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
)

_NAME_TAG_PAT = re.compile(
    r'^(\[\d{2}:\d{2}:\d{2}\]\s+Speaker\s+(\S+))(\s*\(([^)]+)\))?(:)(.*)$'
)

def _collect_addressed_names(text: str) -> List[str]:
    """Return a de-duped list of names that appear as being addressed."""
    names = _ADDR_PAT.findall(text)
    # keep order, dedupe
    seen = set(); out = []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out

def _fill_missing_name_adaptively(base_text: str, out_text: str, candidates: List[str]) -> str:
    """
    If exactly two labels are present and only one has a confident name,
    infer the other from addressed-name frequency and intro mentions.
    Only fills when unambiguous; otherwise leaves output unchanged.
    """
    # label -> assigned name (from out_text), and all labels
    label_to_name: dict[str, str] = {}
    labels: set[str] = set()
    lines = out_text.splitlines()

    for ln in lines:
        m = _NAME_TAG_PAT.match(ln)
        if not m:
            continue
        label = m.group(2)
        labels.add(label)
        name = (m.group(4) or "").strip()
        # treat explicit names other than "Unknown" as confident
        if name and name.lower() != "unknown":
            label_to_name[label] = name

    # need exactly two labels
    if len(labels) != 2:
        return out_text

    # exactly one confidently named label?
    unnamed = [lab for lab in labels if lab not in label_to_name]
    if len(unnamed) != 1:
        return out_text
    named_label = next(iter(set(labels) - set(unnamed)))
    other_label = unnamed[0]

    # Build a frequency map of addressed names on lines spoken by the OTHER label.
    # Use OUT lines (same line structure) to know which label is speaking.
    freq: dict[str, int] = {}
    for ln in lines:
        m = _NAME_TAG_PAT.match(ln)
        if not m:
            continue
        label = m.group(2)
        if label != other_label:
            continue
        for name in _ADDR_PAT.findall(ln):
            freq[name] = freq.get(name, 0) + 1

    # restrict to candidates if you like
    cand_set = set(candidates or [])
    if cand_set:
        freq = {n: c for n, c in freq.items() if n in cand_set}

    if not freq:
        return out_text

    # choose the most frequent addressed name, but avoid picking the already-named person
    already_named = {label_to_name[named_label]}
    best_name, best_count = None, 0
    for n, c in freq.items():
        if n in already_named:
            continue
        if c > best_count:
            best_name, best_count = n, c

    if not best_name:
        return out_text

    # Rewrite lines for the other_label that are missing or say (Unknown) -> add (best_name)
    new_lines = []
    for ln in lines:
        m = _NAME_TAG_PAT.match(ln)
        if not m:
            new_lines.append(ln); continue
        pre, label, paren, name, colon, rest = m.groups()
        if label == other_label:
            if not name or name.lower() == "unknown":
                ln = f"{pre} ({best_name}){colon}{rest}"
        new_lines.append(ln)

    return "\n".join(new_lines)




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
        speaker_labels=True,
        punctuate=True,
        disfluencies=True,
    )
    transcript = aai.Transcriber().transcribe(audio_file_path, config)

    # ---- Build segments [{'start','end','text','spk'}] ----
    segments: list[dict] = []
    unique_speakers: set[int] = set()

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

    if not segments:
        words = getattr(transcript, "words", None) or []
        segs = _bucket_words_to_segments(words, bucket_ms=8000)
        unique_speakers = {0}
        for s in segs:
            segments.append({"start": s["start"], "end": s["end"], "text": s["text"], "spk": 0})

    if not segments:
        logger.error("No segments produced from utterances, paragraphs, or words.")
        return (transcript.text or "").strip()

    # ---- Normalize speaker ids by first appearance to A, B, C... ----
    label_map = {}
    next_letter_idx = 0
    for s in segments:
        spk = s["spk"]
        if spk not in label_map:
            label_map[spk] = string.ascii_uppercase[next_letter_idx] if next_letter_idx < 26 else f"S{next_letter_idx+1}"
            next_letter_idx += 1
        s["spk_label"] = label_map[spk]

    # ---- Render baseline (canonical) ----
    lines: list[str] = []
    for seg in segments:
        text = (seg["text"] or "").strip()
        if not text:
            continue
        ts = format_timestamp(seg["start"])
        spk_label = seg["spk_label"]
        lines.append(f"[{ts}] Speaker {spk_label}: {text}")
        lines.append("")  # blank line between segments
    base_text = "\n".join(lines).strip()

    # ---- Harvest lightweight cues for names/roles ----
    first_talking_line = next((ln for ln in base_text.splitlines() if ln.strip()), "")

    looks_like_host_intro = bool(re.search(
        r"\b(Joining us now|We('?| a)re joined by|Welcome back|Joining me|Now to|We have with us)\b",
        first_talking_line,
        re.I,
    ))

    # Require >=2 tokens for intro names to avoid "Joining"
    mentioned_in_intro = re.findall(rf"\b({INTRO_FULLNAME})\b", first_talking_line)
    mentioned_in_intro = [n for n in mentioned_in_intro if _is_ok_name(n)]

    # Names addressed like "Well, Liz," anywhere in the body
    addressed_names = re.findall(r"\bWell,\s+([A-Z][a-zA-Z]+)\b", base_text)
    addressed_names = [n for n in dict.fromkeys(addressed_names) if _is_ok_name(n)]  # dedupe + filter

    # Capitalized tokens followed by comma (e.g., "Liz,")
    comma_names = re.findall(r"\b([A-Z][a-zA-Z]+),", base_text)
    comma_names = [n for n in dict.fromkeys(comma_names) if _is_ok_name(n)]

    # Merge auto candidates + user hint
    auto_candidates = mentioned_in_intro + addressed_names + comma_names
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

    logger.debug("candidates=%s", candidate_block)
    logger.debug("role_hints=%s", role_hint_block)

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
{ "\n\n".join([
    f"Speaker {lab} samples:\n" + "\n".join(
        [f"[{format_timestamp(seg['start'])}] {seg['text']}" for seg in segments if seg['spk_label'] == lab][:2]
    ) or "(no sample)"
    for lab in sorted(set(s['spk_label'] for s in segments))
]) }
-----
""".strip()

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

        # 1) Scrub obviously wrong names like "(Joining)"
        out = _scrub_bad_assigned_names(out)

        # NEW: infer the other speaker's name from addressed cues like "Well, Liz,"
        out = _force_other_label_name_when_addressed(base_text, out)

        # Keep your existing fallback too (only triggers in the simple 1-unknown/1-candidate case)
        out = _postfix_single_unknown_with_single_candidate(out, candidates)


        # Debug snapshots
        logger.debug("---- BASE (first 6 lines) ----\n%s", "\n".join(base_text.splitlines()[:6]))
        logger.debug("---- OUT  (first 6 lines) ----\n%s", "\n".join(out.splitlines()[:6]))
        logger.debug("Candidates: %s", candidate_block)
        logger.debug("Role hints:\n%s", role_hint_block)

        # ---- Safety validation with allowed relabeling (bijection mapping) ----
        base_lines = base_text.splitlines()
        out_lines  = out.splitlines()
        if len(base_lines) != len(out_lines):
            logger.warning("Name guessing changed line count; returning baseline.")
            return base_text

        ts_pat       = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+Speaker\s+(\S+):")
        ts_pat_named = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+Speaker\s+(\S+)\s*(\([^)]+\))?:")

        label_map_in2out: dict[str, str] = {}
        label_map_out2in: dict[str, str] = {}

        for i, (a, b) in enumerate(zip(base_lines, out_lines), 1):
            if not a.strip():   # blank line must remain blank
                if b.strip() != "":
                    logger.warning("Line %d: expected blank line; returning baseline.", i)
                    return base_text
                continue

            ma = ts_pat.match(a)
            mb = ts_pat_named.match(b)
            if not ma or not mb:
                logger.warning("Line %d: timestamp/speaker tag mismatch.\nBASE:%r\nOUT :%r", i, a, b)
                return base_text

            # timestamps must be identical
            if ma.groups()[:3] != mb.groups()[:3]:
                logger.warning("Line %d: timestamp changed.\nBASE:%r\nOUT :%r", i, a, b)
                return base_text

            in_label  = ma.group(4)
            out_label = mb.group(4)

            # bijection check
            prev_out = label_map_in2out.get(in_label)
            prev_in  = label_map_out2in.get(out_label)
            if prev_out and prev_out != out_label:
                logger.warning("Line %d: inconsistent relabeling %r->%r (saw %r).", i, in_label, out_label, prev_out)
                return base_text
            if prev_in and prev_in != in_label:
                logger.warning("Line %d: non-bijective relabeling %r <- %r (saw %r).", i, out_label, in_label, prev_in)
                return base_text

            label_map_in2out[in_label] = out_label
            label_map_out2in[out_label] = in_label


        # Build the candidate list you already compute:
        candidates_list = [c.strip() for c in (candidate_block.split(",") if candidate_block else []) if c.strip()]

        # Adaptively fill missing name for the other label (if unambiguous)
        out = _fill_missing_name_adaptively(base_text, out, candidates_list)

        return out or base_text

    except Exception as e:
        logger.warning("Name guessing step failed; returning baseline. Error: %s", e)
        return base_text


# ------------- Large-file chunking utilities (unchanged) ----------------

def _transcribe_large_file(audio_path: str, model: str, overlap_seconds: int, file_size: int) -> str:
    """Handle transcription of large audio files by splitting into chunks."""
    try:
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ]
        total_duration = float(subprocess.check_output(duration_cmd).decode().strip())
        logger.info(f"Total audio duration: {total_duration} seconds")
    except Exception as e:
        logger.error(f"Failed to get audio duration: {str(e)}")
        raise RuntimeError(f"Failed to get audio duration: {str(e)}")

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

            chunk_size = os.path.getsize(chunk_path)
            logger.debug(f"Chunk {i+1}: {start_time:.2f}-{end_time:.2f}s, size: {chunk_size} bytes")
            if chunk_size > CHUNK_SIZE_LIMIT:
                logger.warning(f"Chunk {i+1} exceeds size limit: {chunk_size} bytes")

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


