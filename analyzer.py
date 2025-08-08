import logging
import sys
import json
from typing import Optional, List, Dict, Any
from config import Config
from prompts import format_text_bullet_prompt, format_text_highlight_prompt

# --- Dependency Checks ---
try:
    import openai
    # Explicitly import the required classes and exceptions from the openai library
    from openai import OpenAI, APIError, AuthenticationError, RateLimitError
    from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
except ImportError:
    # You might want to refine this error message to pinpoint which library failed
    # For example:
    # except ImportError as e:
    #     print(f"ERROR: Required library import failed: {e}. ", file=sys.stderr)
    #     print("Ensure 'openai' and 'tenacity' are installed: pip install openai tenacity", file=sys.stderr)
    print(
        "ERROR: Required libraries ('openai', 'tenacity') not found or failed to import. "
        "Install using: pip install openai tenacity",
        file=sys.stderr
    )
    sys.exit(1)

# --- Helper for Tenacity Retry Logging ---
def _log_retry_attempt(retry_state):
    """
    Logs details of a retry attempt for API calls using tenacity.

    Args:
        retry_state: The state object provided by tenacity.
    """
    exc = retry_state.outcome.exception()
    wait_time = getattr(retry_state.next_action, 'sleep', 0)
    logging.warning(
        f"API Error encountered: {exc}. Retrying attempt "
        f"{retry_state.attempt_number} after {wait_time:.2f} seconds..."
    )

# --- Core Function ---
@retry(
    wait=wait_random_exponential(min=3, max=30), # Wait 3-30 seconds between retries
    stop=stop_after_attempt(4), # Retry up to 3 times (4 attempts total)
    retry=retry_if_exception_type((APIError, RateLimitError)), # Retry on API errors & rate limits
    after=_log_retry_attempt, # Log on retries
    reraise=True # Re-raise the exception if all retries fail
)
# Retry Behavior Documentation:
# - Exponential backoff between retries (3-30s)
# - Maximum of 4 total attempts (3 retries)
# - Only retries on API errors and rate limits
# - Logs each retry attempt via _log_retry_attempt
# - Re-raises final exception if all retries fail
# - Designed to handle temporary API issues gracefully
def legacy_analyze_transcript(
    transcript_text: str,
    target_name: str,
    open_ai_key
) -> Optional[str]:
    """
    Analyzes the transcript using an LLM (Large Language Model) based on a defined prompt.

    This function sends the transcript and target name to the OpenAI API
    to generate a plain text analysis. It includes retry logic for handling
    transient API errors and rate limits.

    Args:
        transcript_text: The full text of the transcript to be analyzed.
        target_name: The name of the person or entity that the analysis should focus on.

    Returns:
        A plain text string containing the analysis results, or None if the
        analysis fails after all retry attempts or due to critical errors
        like authentication failure.
    """
    logging.info(f"Starting analysis for target: {target_name} using model: {Config.ANALYSIS_MODEL}")

    if not transcript_text or not transcript_text.strip():
        logging.error("Cannot analyze an empty transcript.")
        return None
    if not target_name or not target_name.strip():
        logging.error("Target name cannot be empty for analysis.")
        return None

    try:

        client = OpenAI(api_key=open_ai_key)
        
        def format_analysis_prompt(transcript_text: str, target_name: str) -> str:
            """Formats the analysis prompt for LLM processing."""
            return f"""Analyze the following transcript for key statements made by or about {target_name}:

Transcript:
{transcript_text}

Instructions:
1. Identify all significant statements, claims, or commitments made by {target_name}
2. Note any factual assertions made by others about {target_name}
3. Highlight any potentially controversial or newsworthy statements
4. Provide a concise summary of the key points"""

        # Format the prompt using the transcript and target name
        prompt = format_analysis_prompt(transcript_text, target_name)

        logging.debug("Sending analysis prompt to LLM...")
        # Consider logging the prompt content here for detailed debugging if needed

        # Call the OpenAI Chat Completions API
        response = client.chat.completions.create(
            model=Config.ANALYSIS_MODEL,
            messages=[
                # A system message can be added here to guide the AI's behavior
                # {"role": "system", "content": "You are a helpful research assistant focused on accuracy and direct evidence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # Lower temperature for more factual, less creative output
            # max_tokens=... # Optional: set max_tokens to control output length and cost
        )

        # Extract the analysis content from the API response
        analysis_content = response.choices[0].message.content
        logging.info("Analysis received from LLM.")
        # Uncomment the line below to log the raw LLM output for debugging
        # logging.debug(f"Raw LLM Analysis Output:\n{analysis_content}")

        # Check if the analysis content is empty
        if not analysis_content or not analysis_content.strip():
             logging.warning("LLM returned empty analysis content.")
             return None # Return None or a default message if no content is received

        return analysis_content.strip()

    # Specific error handling for OpenAI API errors
    except AuthenticationError:
        logging.error("OpenAI Authentication Failed during analysis. Check API key.")
        logging.debug("Re-raising AuthenticationError") # Log before re-raising
        raise # Re-raise the exception to be caught by the caller (e.g., main)
    except RateLimitError:
        logging.error("OpenAI Rate Limit Exceeded during analysis after retries.")
        logging.debug("Re-raising RateLimitError") # Log before re-raising
        raise # Re-raise the exception
    except APIError as e:
        logging.error(f"OpenAI API error occurred during analysis after retries: {e}")
        logging.debug("Re-raising APIError") # Log before re-raising
        raise # Re-raise the exception
    except Exception as e:
        # Catch any other unexpected exceptions
        logging.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
        logging.debug("Re-raising unexpected exception") # Log before re-raising
        raise # Re-raise the exception to be caught by the caller

@retry(
    wait=wait_random_exponential(min=5, max=60),  # Longer wait times for bullet extraction
    stop=stop_after_attempt(6),  # More attempts allowed for bullet extraction
    retry=retry_if_exception_type(RateLimitError), # Only retry on rate limits
    after=_log_retry_attempt,  # Log each retry attempt
    reraise=True  # Re-raise if all retries fail
)
# Retry Behavior Documentation:
# - Longer exponential backoff (5-60s) since bullet extraction is more intensive
# - Maximum of 6 total attempts (5 retries)
# - Only retries on rate limits (not general API errors)
# - Logs each retry attempt via _log_retry_attempt
# - Re-raises final exception if all retries fail
def extract_raw_data_from_text(
    transcript_text: str,
    target_name: str,
    metadata: Dict[str, Any],
    open_ai_api,
    prompt_type,
    max_bullets: int = 100,
) -> List[Dict[str, Optional[str]]]:
    """
    Extracts structured bullet points from a transcript using the OpenAI API.

    This function sends the transcript, target name, and video metadata to the
    LLM with a specific prompt designed to extract key bullet points in a
    structured, delimited format. It includes retry logic for handling
    transient API errors and rate limits.

    Args:
        transcript_text: The full text of the transcript to analyze.
        target_name: The name of the person or entity to focus the bullet
                     point extraction on.
        metadata: A dictionary containing video metadata, expected to have
                  keys like 'title', 'uploader', 'upload_date', and 'webpage_url'.
                  This metadata is used to provide context to the LLM and
                  potentially populate source/date fields in the extracted bullets.
        max_bullets: The maximum number of bullet points to attempt to extract.
        prompt_type: Direct to either highlight or bullet prompt:
            format_text_bullet_prompt OR format_text_highlight_prompt

    Returns:
        A list of dictionaries, where each dictionary represents a raw bullet
        point with potential keys: 'headline_raw', 'speaker_raw', 'body_raw',
        'source_raw', and 'date_raw'. Returns an empty list if extraction fails,
        no relevant bullets are found, or inputs are invalid.
    """
    logging.info(f"Starting Text Bullet extraction for target: {target_name}")

    # Basic input validation
    if not transcript_text or not transcript_text.strip():
        logging.error("Cannot extract text bullets from an empty transcript.")
        return []
    if not metadata:
        logging.error("Cannot extract text bullets: Metadata is missing.")
        return []

    try:
        client = OpenAI(api_key=open_ai_api)
        # Format the prompt for bullet extraction
        if prompt_type == "format_text_bullet_prompt":
            prompt = format_text_bullet_prompt(
                transcript_text, target_name, metadata, max_bullets
            )
        elif prompt_type == "format_text_highlight_prompt":
            prompt = format_text_highlight_prompt(
                transcript_text, target_name, metadata, max_bullets
            )

        logging.debug("Sending Text Bullet extraction prompt to LLM...")
        # Consider logging the prompt content here for detailed debugging if needed

        # Call the OpenAI Chat Completions API
        response = client.chat.completions.create(
            model=Config.ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # Lower temperature for more structured, predictable output
        )

        # Extract the raw text response from the API
        raw_text_response = response.choices[0].message.content
        logging.info("Text bullet extraction response received.")

        # Check for empty or specific "no bullets found" response
        if not raw_text_response or not raw_text_response.strip():
             logging.warning("LLM returned empty content for text bullet extraction.")
             return []
        if "@@NO BULLETS FOUND@@" in raw_text_response:
             logging.info("LLM indicated no relevant bullets found.")
             return []

        # --- Parse Delimited Text ---
        # The LLM is expected to return data in a specific delimited format.
        # This section parses that format into a list of dictionaries.
        extracted_bullets_raw = []
        # Split the response into individual bullet blocks using the start delimiter
        # Expected format:
        # *** BULLET START ***
        # **Headline:** ... @@DELIM@@ **Speaker:** ... @@DELIM@@ **Body:** ... @@DELIM@@ **Source:** ... @@DELIM@@ **Date:** ...
        # *** BULLET END ***
        bullet_blocks = raw_text_response.split("*** BULLET START ***")

        for block in bullet_blocks:
            block = block.strip()
            # Skip empty blocks or blocks that don't contain the end delimiter
            if not block or "*** BULLET END ***" not in block:
                continue

            # Remove the end delimiter to isolate the content within the block
            content = block.split("*** BULLET END ***")[0].strip()

            # Split the content by the main delimiter (@@DELIM@@) to get individual fields
            parts = content.split("@@DELIM@@")
            bullet_data = {}

            # Extract data for each field based on known prefixes (e.g., **Headline:**)
            for part in parts:
                part = part.strip()
                if part.startswith("**Headline:**"):
                    bullet_data['headline_raw'] = part[len("**Headline:**"):].strip()
                elif part.startswith("**Speaker:**") and prompt_type == "format_text_bullet_prompt":
                    bullet_data['speaker_raw'] = part[len("**Speaker:**"):].strip()
                elif part.startswith("**Body:**") and prompt_type == "format_text_bullet_prompt":
                    bullet_data['body_raw'] = part[len("**Body:**"):].strip()
                elif part.startswith("**Source:**") and prompt_type == "format_text_bullet_prompt":
                    bullet_data['source_raw'] = part[len("**Source:**"):].strip()
                elif part.startswith("**Date:**") and prompt_type == "format_text_bullet_prompt":
                    bullet_data['date_raw'] = part[len("**Date:**"):].strip()
                # elif part.startswith("**Time:**") and prompt_type == "format_text_bullet_prompt":
                    # bullet_data['time_raw'] = part[len("**Time:**"):].strip()

            # Basic validation: check if essential parts (headline, body, speaker) were found
            # Source and date are optional and might be missing.
            if 'headline_raw' in bullet_data and prompt_type == "format_text_highlight_prompt":
                # Append the extracted data as a dictionary to the results list
                extracted_bullets_raw.append({
                    "headline_raw": bullet_data.get('headline_raw')
                })
            elif 'headline_raw' in bullet_data and prompt_type == "format_text_bullet_prompt":
                # Append the extracted data as a dictionary to the results list
                extracted_bullets_raw.append({
                    "headline_raw": bullet_data.get('headline_raw'),
                    "speaker_raw": bullet_data.get('speaker_raw'),
                    "body_raw": bullet_data.get('body_raw'),
                    "source_raw": bullet_data.get('source_raw'), # Will be None if not found
                    "date_raw": bullet_data.get('date_raw')      # Will be None if not found
                    # "time_raw": bullet_data.get('time_raw')
                })
            else:
                # Log a warning if a block could not be fully parsed
                logging.warning(f"Could not parse all required fields from block: {content[:100]}...")

        logging.info(f"Parsed {len(extracted_bullets_raw)} raw bullet data dicts from text.")
        return extracted_bullets_raw

    # Specific error handling for OpenAI API errors.
    # These exceptions are re-raised after logging to be handled by the caller.
    except AuthenticationError:
        logging.error("Authentication error during text bullet extraction.")
        raise
    except RateLimitError:
        logging.error("Rate limit error during text bullet extraction.")
        raise
    except APIError as e:
        logging.error(f"API error during text bullet extraction: {e}")
        raise
    except Exception as e:
        # Catch any other unexpected exceptions during the process
        logging.error(f"Unexpected error during text bullet extraction: {e}", exc_info=True)
        return [] # Return empty list on non-critical failures to allow pipeline to continue

