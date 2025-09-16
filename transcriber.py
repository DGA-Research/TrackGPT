# transcriber.py
import os
import re
import string
import logging
from typing import Optional, List, Dict, Any

import assemblyai as aai
import openai

logger = logging.getLogger("transcriber")

# -------------------------
# Small helpers
# -------------------------

def format_timestamp(ms: int | float) -> str:
    total_seconds = int(round(ms / 1000.0))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def _clean_hint_name(hint: str | None) -> str | None:
    """Pick a simple display name from a semicolon/comma/and-separated hint string."""
    if not hint:
        return None
    parts = re.split(r"[;,/]| and | with | vs ", hint, flags=re.IGNORECASE)
    for p in parts:
        name = p.strip()
        if name:
            return re.sub(r"\s+", " ", name)
    return None

def _bucket_words_to_segments(words, bucket_ms: int = 8000) -> list[dict]:
    """When only word timings exist, group them into ~8s segments with start/end/text."""
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


# -------------------------
# Heuristic fallback naming (transcriber2-style)
# -------------------------

_NAME_BLOCKLIST = {
    "fox", "news", "turning", "point", "usa", "founder", "host",
    "commentator", "press", "secretary", "governor", "president",
    "senator", "vice", "speaker", "majority", "leader"
}

_INTRO_TRIGGERS = [
    r"Joining us now",
    r"We(?:'| a)re joined by",
    r"Welcome back",
    r"Joining me",
    r"Now to",
    r"We have with us",
]

def _extract_proper_names(text: str) -> List[str]:
    """
    Extract candidate Proper Names (1–3 tokens, capitalized) from a line of text,
    filtering obvious non-name nouns and multiword orgs.
    """
    # Capture up to 3 capitalized tokens; allow hyphens / apostrophes within tokens
    raw_names = re.findall(r"\b([A-Z][A-Za-z'-\-]+(?:\s+[A-Z][A-Za-z'-\-]+){0,2})\b", text)
    out = []
    for n in raw_names:
        parts = n.split()
        if 1 <= len(parts) <= 3:
            norm = n.strip()
            if norm and norm.lower() not in _NAME_BLOCKLIST:
                out.append(norm)
    # de-dupe keep order
    seen = set()
    dedup = []
    for n in out:
        if n not in seen:
            seen.add(n)
            dedup.append(n)
    return dedup

def _looks_like_intro(line: str) -> bool:
    return any(re.search(trig, line, flags=re.I) for trig in _INTRO_TRIGGERS)

def _split_lines_preserve(s: str) -> List[str]:
    return s.splitlines()

def _speaker_tag_regex(label: str) -> re.Pattern:
    return re.compile(rf"^(\[\d{{2}}:\d{{2}}:\d{{2}}\]\s+Speaker\s+{re.escape(label)})(\s*(\([^)]+\))?)\s*:", re.UNICODE)

def _parse_dialogue_base(base_text: str) -> List[Dict[str, str]]:
    """
    Parse rendered baseline into a list of dicts:
    {'raw': line, 'label': 'A', 'timestamp': '[00:00:03]', 'body': 'text...'} for speaker lines,
    or {'raw': '', 'blank': True} for blank lines.
    """
    out = []
    pat = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+Speaker\s+(\S+):\s*(.*)$", re.UNICODE)
    for line in _split_lines_preserve(base_text):
        if not line.strip():
            out.append({'raw': line, 'blank': True})
            continue
        m = pat.match(line)
        if not m:
            out.append({'raw': line, 'other': True})
            continue
        ts = f"[{m.group(1)}:{m.group(2)}:{m.group(3)}]"
        out.append({
            'raw': line,
            'timestamp': ts,
            'label': m.group(4),
            'body': m.group(5),
        })
    return out

def _apply_name_map(base_text: str, name_map: Dict[str, str]) -> str:
    """
    Insert '(Name)' after Speaker label if not already present; preserve everything else.
    """
    out_lines = []
    for line in _split_lines_preserve(base_text):
        if not line.strip():
            out_lines.append(line)
            continue
        # detect "[..] Speaker X ..." with optional existing "(...)" already there
        m = re.match(r"^(\[\d{2}:\d{2}:\d{2}\]\s+Speaker\s+(\S+))(\s*\([^)]+\))?\s*:(.*)$", line)
        if not m:
            out_lines.append(line)
            continue
        head = m.group(1)
        label = m.group(2)
        paren = m.group(3) or ""
        tail = m.group(4)
        maybe_name = name_map.get(label)
        if maybe_name:
            # replace or insert
            out_lines.append(f"{head} ({maybe_name}):{tail}")
        else:
            # keep as-is
            out_lines.append(line)
    return "\n".join(out_lines)

def _heuristic_name_fallback(base_text: str, speaker_hint: Optional[str]) -> str:
    """
    Heuristic pass:
      - If two speakers, infer 'guest' from host intro (names in first line that looks like intro)
      - Infer 'host' from addressed tokens like 'Well, Liz,' within the guest's turn
      - Use speaker_hint tokens as candidates, but never hard-code specific names
    """
    lines = _parse_dialogue_base(base_text)
    # Collect speaker labels seen in order
    labels = [l['label'] for l in lines if 'label' in l]
    unique_labels = []
    for lab in labels:
        if lab not in unique_labels:
            unique_labels.append(lab)

    if len(unique_labels) == 0:
        return base_text

    # Build candidate names from:
    #  - first speaking line (intro-like)
    #  - any addressed forms like "Well, Name" or "^Name," at start
    #  - speaker_hint tokens
    first_speaker_label: Optional[str] = next((l['label'] for l in lines if 'label' in l), None)
    first_line_text = next((l['body'] for l in lines if 'label' in l), "")

    candidates: List[str] = []
    intro_names = []
    if _looks_like_intro(first_line_text):
        intro_names = _extract_proper_names(first_line_text)
        candidates += intro_names

    addressed_names = []
    # capture "Well, Liz," or "Liz," at line start or after small discourse marker
    addr_pat = re.compile(r"\b(?:Well|Right|No|Yes|Now|So|Uh|Um)?\s*,?\s*([A-Z][a-zA-Z'-\-]+)\s*,")
    for l in lines:
        if 'label' not in l:
            continue
        for m in addr_pat.finditer(l['body']):
            nm = m.group(1)
            if nm and nm.lower() not in _NAME_BLOCKLIST:
                addressed_names.append(nm)

    if speaker_hint:
        for tok in re.split(r"[;,/]| and | with | vs ", speaker_hint, flags=re.I):
            tok = tok.strip()
            if tok:
                candidates.append(tok)

    # dedupe
    def _dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    candidates = _dedupe(candidates)
    addressed_names = _dedupe(addressed_names)

    # If exactly two speakers, map:
    #  - If intro has a full name like "Tomi Lahren", assume that's the GUEST (not first speaker)
    #  - If we see "Well, Liz," then 'Liz' is likely the other speaker being addressed
    name_map: Dict[str, str] = {}

    if len(unique_labels) == 2:
        a, b = unique_labels[0], unique_labels[1]
        # Guess guest_full from intro
        guest_full: Optional[str] = None
        for nm in intro_names:
            # prefer 2- or 3-token full names in intro for guest
            if len(nm.split()) >= 2:
                guest_full = nm
                break
        if not guest_full and intro_names:
            guest_full = intro_names[0]

        if guest_full:
            # If first line (intro) is by 'a', put guest on the OTHER label
            # i.e., host speaks first → guest is the other speaker
            host_label = first_speaker_label or a
            guest_label = b if host_label == a else a
            name_map[guest_label] = guest_full

        # Guess host_name from addressed_names if it doesn't match guest_full
        host_name: Optional[str] = None
        for nm in addressed_names:
            if not guest_full or nm not in guest_full:
                host_name = nm
                break

        if host_name and (first_speaker_label in (a, b)):
            name_map[first_speaker_label] = host_name

    # As a last resort: if still nothing and only one label needs a name, use hint
    if speaker_hint:
        hint_name = _clean_hint_name(speaker_hint)
        if hint_name:
            # Assign hint to whichever label remains unnamed and is not clearly an intro speaker
            for lab in unique_labels:
                if lab not in name_map:
                    name_map[lab] = hint_name
                    break

    # Apply only where name is missing/unknown
    return _apply_name_map(base_text, name_map)


# -------------------------
# Main function
# -------------------------

def transcribe_file(audio_file_path: str, openai_key: str, assemblyai_key: str, speaker_hint: Optional[str]) -> str:
    """
    Returns a timecoded transcript as plain text:
        [HH:MM:SS] Speaker X (Optional Name): text

    - Uses AssemblyAI diarization for segments
    - Renders canonical baseline: timestamps + Speaker letters (A, B, C, …)
    - Tries GPT name guessing (strict preservation)
    - If GPT fails or leaves Unknowns, falls back to a heuristic (transcriber2-style)
      to identify likely host/guest names (no hard-coded identities).
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
    segments: List[Dict[str, Any]] = []
    unique_speakers: set[int] = set()

    if getattr(transcript, "utterances", None):
        for u in transcript.utterances:
            txt = (u.text or "").strip()
            if not txt:
                continue
            unique_speakers.add(u.speaker)
            dur = u.end - u.start
            if dur <= 30000:
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

    # ---- Normalize speakers to A, B, C... (stable by numeric id sort) ----
    speaker_list = sorted(list(unique_speakers), key=lambda x: (isinstance(x, str), x))
    label_map = {}
    for i, spk in enumerate(speaker_list):
        label_map[spk] = string.ascii_uppercase[i] if i < 26 else f"S{i+1}"

    for s in segments:
        s["spk_label"] = label_map.get(s["spk"], str(s["spk"]))

    # ---- Render baseline ----
    lines: List[str] = []
    for seg in segments:
        text = (seg["text"] or "").strip()
        if not text:
            continue
        ts = format_timestamp(seg["start"])
        spk_label = seg["spk_label"]
        lines.append(f"[{ts}] Speaker {spk_label}: {text}")
        lines.append("")
    base_text = "\n".join(lines).strip()

    # ---- If single speaker, optionally append deterministic name and return ----
    if len(speaker_list) == 1:
        single_label = segments[0]["spk_label"]
        display_name = _clean_hint_name(speaker_hint) or "Unknown"
        pat = re.compile(rf"^(\[\d{{2}}:\d{{2}}:\d{{2}}\]\s+Speaker\s+{re.escape(single_label)})(:)")
        out_lines = []
        for line in base_text.splitlines():
            if not line.strip():
                out_lines.append(line)
                continue
            m = pat.match(line)
            if m:
                out_lines.append(f"{m.group(1)} ({display_name}){m.group(2)}{line[m.end():]}")
            else:
                out_lines.append(line)
        return "\n".join(out_lines)

    # -------------------------
    # GPT pass (strict)
    # -------------------------
    # Prepare candidate hints for GPT (but we’ll validate strictly)
    # Extract a few short samples per label for GPT context
    by_label: Dict[str, List[str]] = {}
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

    # Build candidate name list (intro/addresses + hints) just for GPT to consider
    first_line = next((ln for ln in base_text.splitlines() if ln.strip()), "")
    intro_names_for_gpt = _extract_proper_names(first_line) if _looks_like_intro(first_line) else []
    addressed_for_gpt = re.findall(r"\b([A-Z][a-zA-Z'-\-]+),", base_text)
    auto_candidates = intro_names_for_gpt + addressed_for_gpt

    if speaker_hint:
        for tok in re.split(r"[;,/]| and | vs | with ", speaker_hint, flags=re.I):
            tok = tok.strip()
            if tok:
                auto_candidates.append(tok)

    # de-dupe
    seen = set()
    candidates = []
    for n in auto_candidates:
        if n and n not in seen and n.lower() not in _NAME_BLOCKLIST:
            seen.add(n)
            candidates.append(n)
    candidates = candidates[:8]
    candidate_block = ", ".join(candidates) if candidates else "None"

    # Role hints
    role_hints = []
    if _looks_like_intro(first_line):
        role_hints.append(
            "If the first line is an intro (e.g., 'Joining us now'), that line is the HOST speaking. "
            "Any person named in that line is a different speaker (the guest)."
        )
    if intro_names_for_gpt:
        role_hints.append("Names explicitly mentioned in the intro line (likely guests): " + ", ".join(intro_names_for_gpt) + ".")
    if addressed_for_gpt:
        role_hints.append("If a line begins with 'Well, NAME,' then NAME is being addressed (the other speaker).")

    role_hint_block = "\n".join(role_hints) if role_hints else "No extra role hints."

    system_prompt = f"""
You must preserve the input transcript EXACTLY:
- Do NOT change or remove timestamps like [HH:MM:SS].
- Do NOT merge, split, reorder, or wrap lines.
- Do NOT remove blank lines.
- Do NOT alter anything after the colon.

Task: ONLY append a guessed human-readable name in parentheses immediately after the 'Speaker X' tag on each line.

Example:
  Input:  [00:00:03] Speaker A: Thank you for coming.
  Output: [00:00:03] Speaker A (Jane Doe): Thank you for coming.

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

    out_gpt = None
    try:
        client = openai.OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_text}
            ],
            temperature=0.0
        )
        out_gpt = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Name guessing step failed; will use heuristic fallback. Error: %s", e)
        out_gpt = None

    # -------------------------
    # Validate GPT output; if bad, run heuristic fallback
    # -------------------------
    def _valid_and_complete(gpt_text: str) -> bool:
        if not gpt_text:
            return False
        base_lines = base_text.splitlines()
        out_lines = gpt_text.splitlines()
        if len(base_lines) != len(out_lines):
            return False
        # check timestamps & labels preserved; ensure not all Unknown
        ts_pat       = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+Speaker\s+(\S+):")
        ts_pat_named = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+Speaker\s+(\S+)\s*(\([^)]+\))?:")
        saw_named = False
        for a, b in zip(base_lines, out_lines):
            if not a.strip():
                if b.strip() != "":
                    return False
                continue
            ma = ts_pat.match(a)
            mb = ts_pat_named.match(b)
            if not ma or not mb:
                return False
            if ma.groups()[:3] != mb.groups()[:3]:
                return False
            # consider it "named" if there is a paren and not '(Unknown)'
            if mb.group(5):
                if "(Unknown)" not in mb.group(5):
                    saw_named = True
        return saw_named

    if out_gpt and _valid_and_complete(out_gpt):
        return out_gpt

    # Heuristic fallback overlay (transcriber2-style)
    logger.info("Falling back to heuristic name assignment.")
    out_heur = _heuristic_name_fallback(base_text, speaker_hint)
    return out_heur
