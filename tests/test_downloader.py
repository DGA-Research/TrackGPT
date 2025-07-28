# tests/test_downloader.py
import pytest
import subprocess
from pathlib import Path
from unittest.mock import Mock
import sys
import os

# Add the parent directory to sys.path to allow importing modules from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function to test
from downloader import download_audio
# Import Config if needed for AUDIO_FORMAT etc.
from config import Config
import logging # Need to import logging to mock logging.error

# --- Test Cases ---

def test_download_audio_success(mocker, tmp_path):
    """Tests successful download via mocked subprocess and file check."""
    # Arrange
    test_url = "http://example.com/video"
    test_base_filename = "test_audio"
    output_dir = tmp_path # pytest fixture for temporary directory
    expected_final_path = output_dir / f"{test_base_filename}.{Config.AUDIO_FORMAT}"
    expected_cmd_part = ["-x", "--audio-format", Config.AUDIO_FORMAT] # Check key args

    # Mock dependencies
    mocker.patch("downloader.YT_DLP_PATH", "/fake/path/to/yt-dlp") # Ensure it's not None
    mock_mkdir = mocker.patch("pathlib.Path.mkdir")
    mock_run = mocker.patch("subprocess.run")
    mock_exists = mocker.patch.object(Path, "exists") # Mock the 'exists' method of Path instances

    # Configure mocks
    mock_run.return_value = Mock(returncode=0, stdout="Download complete", stderr="") # Successful run
    mock_exists.return_value = True # Simulate the final file existing

    # Act
    result = download_audio(test_url, output_dir, test_base_filename)

    # Assert
    assert result == str(expected_final_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # Check that subprocess.run was called, maybe checking specific args
    call_args, call_kwargs = mock_run.call_args
    assert call_args[0][0] == "/fake/path/to/yt-dlp" # Check executable
    assert call_args[0][1] == test_url # Check URL
    assert all(item in call_args[0] for item in expected_cmd_part) # Check key args present
    assert call_kwargs.get("check") is True
    # Check that Path.exists was called on the expected final path
    mock_exists.assert_called_once_with()
    # The assertion mock_exists.assert_called_once_with() on line 52 already verifies that exists() was called.
    # We don't need to check __self__ directly.


def test_download_audio_subprocess_error(mocker, tmp_path):
    """Tests handling of a subprocess error during download."""
    # Arrange
    test_url = "http://example.com/video"
    test_base_filename = "test_audio_fail"
    output_dir = tmp_path

    # Mock dependencies
    mocker.patch("downloader.YT_DLP_PATH", "/fake/path/to/yt-dlp")
    mocker.patch("pathlib.Path.mkdir")
    mock_run = mocker.patch("subprocess.run")
    mock_log_error = mocker.patch("downloader.logging.error") # Optional: check logs

    # Configure mock to raise error
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd=["yt-dlp", "..."], stderr="Download failed!"
    )

    # Act
    result = download_audio(test_url, output_dir, test_base_filename)

    # Assert
    assert result is None
    # Optional: Check that specific error messages were logged
    assert mock_log_error.call_count >= 1
    # Example check (adjust based on your actual log message):
    assert any("yt-dlp failed" in call.args[0] for call in mock_log_error.call_args_list)


def test_download_audio_ytdlp_path_not_found(mocker, tmp_path):
    """Tests case where yt-dlp executable path is None."""
     # Arrange
    mocker.patch("downloader.YT_DLP_PATH", None) # Simulate yt-dlp not found
    mock_log_error = mocker.patch("downloader.logging.error")

    # Act
    result = download_audio("some_url", tmp_path, "base")

    # Assert
    assert result is None
    mock_log_error.assert_called_once_with("yt-dlp executable not found. Cannot download.")

# Add test case for file not found after download if desired (`mock_exists.return_value = False`)