# --- START OF FILE prompts.py ---

from typing import Dict, Any

TEXT_BULLET_PROMPT_TEMPLATE = """# ROLE: Meticulous Communications Analyst & Information Extractor

# GOAL: Extract specific and impactful political statements, claims, or contrasts involving **{target_name}**. Focus on content with potential public resonance, policy significance, reputational impact, or campaign relevance.

# --- CONTEXTUAL METADATA (Use for source/date identification) ---
# Source Title: {video_title}
# Source Provider: {video_platform} # Default source provider
# Provider Channel/Uploader: {video_uploader} # Use as potential source if needed
# Upload Date (YYYYMMDD): {video_upload_date} # Default date
# Source URL: {video_url} # Default URL
# --- END METADATA ---

# --- TRANSCRIPT TEXT ---
{transcript_text}
# --- END TRANSCRIPT ---

# ================== INSTRUCTIONS ==================

1. **Extract up to {max_bullets} bullet points** based only on the transcript above.
2. **Each bullet should reflect a single, standalone factual or rhetorical claim** made by or directly about {target_name}.
3. Use the speaker’s own words where impactful, but summarize concisely if needed.
4. Capture statements that:
   - Indicate political, moral, or emotional framing.
   - Mention opponents by name or refer to policy battles.
   - Include memorable phrases, strong opinions, personal experiences, or rallying messages.
5. **Omit** introductory banter, jokes, or off-topic interviewer chatter **unless** it involves a meaningful claim or public statement.
6. **Limit** Limit the bullet points to only the necessary information. This should be a concise list of highlights.


# ================== OUTPUT FORMAT (PLAIN TEXT ONLY) ==================

# For EACH bullet point extracted, output EXACTLY the following block structure, using "@@DELIM@@" as the separator:

*** BULLET START ***
**Headline:** [Concise PAST TENSE Summary (Accurately Attributed Actor) + Period. Do not include any quotations from the transcript. The output should be a clean list of highlights ready for insertion into a “Highlights” section of a tracking report.]
@@DELIM@@
*** BULLET END ***

# Repeat the entire "*** BULLET START ***" to "*** BULLET END ***" block for each bullet.
# Put a single blank line between each "*** BULLET END ***" and the next "*** BULLET START ***".

# == CRITICAL: DO NOT ==
#   *   DO NOT output JSON.
#   *   DO NOT add any text before the first "*** BULLET START ***" (unless it's "@@NO BULLETS FOUND@@").
#   *   DO NOT add any text after the final "*** BULLET END ***".
#   *   DO NOT use markdown formatting (like **). Use the exact delimiters shown.
#   *   DO NOT apply Title Case formatting to the **Headline:** output. Python code will handle capitalization later.
#   *   DO NOT add surrounding double quotes or prefixes like "According to..." to the **Body:** output field.

# Begin Extraction:
"""

def format_text_bullet_prompt(
    transcript_text: str,
    target_name: str,
    metadata: Dict[str, Any],
    max_bullets: int = 15
) -> str:
    """Formats the Text Bullet Extraction prompt."""
    import logging # Ensure logging is imported

    if not transcript_text or not transcript_text.strip():
        logging.warning("Formatting Text Bullet prompt with empty transcript text.")

    if not metadata:
         logging.warning("Formatting Text Bullet prompt with missing metadata. Using defaults.")
         metadata = {}

    title = metadata.get('title', 'Unknown Title')
    uploader = metadata.get('uploader', 'Unknown Uploader')
    upload_date = metadata.get('upload_date') or "Date Unknown"
    platform = metadata.get('extractor', 'Unknown Platform')
    url = metadata.get('webpage_url', '#')
    platform_display = "YouTube" if str(platform).lower() == "youtube" else str(platform)

    logging.debug(f"Formatting Text Bullet prompt: Title='{title}', Uploader='{uploader}', Date='{upload_date}', Platform='{platform_display}', URL='{url}'")

    try:
        return TEXT_BULLET_PROMPT_TEMPLATE.format(
            target_name=target_name,
            transcript_text=transcript_text,
            video_title=title,
            video_uploader=uploader,
            video_upload_date=upload_date,
            video_platform=platform_display,
            video_url=url,
            max_bullets=max_bullets
        )
    except KeyError as e:
        logging.error(f"Missing key in Text Bullet prompt formatting: {e}")
        raise ValueError(f"Failed to format Text Bullet prompt due to missing key: {e}")
# --- END OF FILE prompts.py ---