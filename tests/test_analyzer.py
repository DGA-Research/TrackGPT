# tests/test_analyzer.py
import pytest
from unittest.mock import Mock, patch # Import patch
import openai # For exception types
import sys
import os

# Add the parent directory to sys.path to allow importing modules from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the analyzer module to access analyzer.OpenAI
import analyzer

# Import the analyzer module to access analyzer.OpenAI
import analyzer

# Import the function to test
from analyzer import analyze_transcript, _log_retry_attempt
from config import Config
# Import format_analysis_prompt if you patch it by its source module
# from prompts import format_analysis_prompt # Not needed if patching analyzer.*

# --- Test Cases ---

def test_analyze_transcript_success(mocker):
    """Tests successful analysis via mocked API call."""
    # Arrange
    mock_transcript = "This is the transcript text containing relevant info."
    mock_target = "Test Target"

    # Define the expected analysis string, including the wrapper
    expected_analysis = """```text
## Factual Statements by Test Target

- Test Target stated they like testing.
- Test Target confirmed the transcript was processed.

## Hot-Button Political Issues Assessment

Testing methodology and API mocking were discussed.

**Supporting Quotes:**

*   **Testing Methodology:**
    > "We should always test our code."
*   **API Mocking:**
    > "Mocking APIs is essential for unit tests."
```"""
    expected_prompt = "Placeholder prompt text"

    # --- MOCKING FIX START ---
    # Mock dependencies
    # Patch format_analysis_prompt WHERE IT'S USED (in analyzer.py)
    mock_format_prompt = mocker.patch("analyzer.format_analysis_prompt", return_value=expected_prompt)

    # Patch the OpenAI CONSTRUCTOR to return a mock client instance
    mock_openai_client_instance = Mock()
    mocker.patch("analyzer.OpenAI", return_value=mock_openai_client_instance)

    # Configure the 'create' method ON THE MOCK INSTANCE
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = expected_analysis
    mock_openai_client_instance.chat.completions.create.return_value = mock_response

    # Mock the API Key check inside the function (still good practice)
    mocker.patch("analyzer.Config.OPENAI_API_KEY", "fake_key_for_test")
    # --- MOCKING FIX END ---

    # Act
    result = analyze_transcript(mock_transcript, mock_target)

    # Assert
    assert result == expected_analysis # This should now pass
    # Verify format_analysis_prompt was called correctly
    mock_format_prompt.assert_called_once_with(mock_transcript, mock_target)
    # Verify the API method on the MOCK INSTANCE was called correctly
    mock_openai_client_instance.chat.completions.create.assert_called_once_with(
        model=Config.ANALYSIS_MODEL,
        messages=[{"role": "user", "content": expected_prompt}],
        temperature=0.2
    )
    # Verify the constructor was called (implicitly checks client creation)
    analyzer.OpenAI.assert_called_once_with(api_key="fake_key_for_test")


def test_analyze_transcript_api_error(mocker):
    """Tests handling of an APIError during analysis (reraised)."""
    # Arrange
    mock_transcript = "Some text"
    mock_target = "Target"
    expected_prompt = "Placeholder prompt text"

    # --- MOCKING FIX START ---
    # Mock dependencies
    # Patch format_analysis_prompt WHERE IT'S USED
    mocker.patch("analyzer.format_analysis_prompt", return_value=expected_prompt)

    # Patch the OpenAI CONSTRUCTOR to return a mock client instance
    mock_openai_client_instance = Mock()
    mocker.patch("analyzer.OpenAI", return_value=mock_openai_client_instance)

    # Configure the 'create' method ON THE MOCK INSTANCE to raise the desired error
    mock_openai_client_instance.chat.completions.create.side_effect = openai.APIError(
        "Simulated API Failed", request=None, body=None
    )

    # Mock API key check inside the function
    mocker.patch("analyzer.Config.OPENAI_API_KEY", "fake_key_for_test")
    # --- MOCKING FIX END ---

    # Mock the retry logger (optional but keeps output clean)
    mocker.patch("analyzer._log_retry_attempt")

    # Act & Assert
    with pytest.raises(openai.APIError, match="Simulated API Failed"):
        analyze_transcript(mock_transcript, mock_target)

    # Assert the constructor was called
    # The retry decorator causes the function (including client initialization) to be called multiple times on failure.
    # Assert that the constructor was called the expected number of times (stop_after_attempt=4).
    assert analyzer.OpenAI.call_count == 4
    # Verify the constructor was called with the correct arguments on each attempt
    analyzer.OpenAI.assert_has_calls([mocker.call(api_key="fake_key_for_test")] * 4)
    # Assert the create method was called (multiple times due to retry)
    assert mock_openai_client_instance.chat.completions.create.call_count == 4

# Keep other tests like test_analyze_transcript_empty_transcript as they were
def test_analyze_transcript_empty_transcript(mocker):
    """Tests that analysis returns None for empty transcript input."""
    # Arrange
    mock_log_error = mocker.patch("analyzer.logging.error")
    # Prevent any attempt to call OpenAI constructor if logic short-circuits
    mock_openai_constructor = mocker.patch("analyzer.OpenAI")

    # Act
    result = analyze_transcript("", "Target")

    # Assert
    assert result is None
    mock_log_error.assert_called_with("Cannot analyze an empty transcript.")
    mock_openai_constructor.assert_not_called() # Ensure client wasn't even created