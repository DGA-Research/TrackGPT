"""
Command-line helper for running bulk audio transcriptions without Streamlit.

Usage example:

    python bulk_transcribe_cli.py --input path/to/folder --target "Target Name"

The script mirrors the Streamlit bulk ZIP workflow but works directly on local
files or ZIP archives, bypassing Streamlit's upload limits.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv

from output import save_text_file
from transcriber import transcribe_file

SUPPORTED_EXTS = {".mp3", ".m4a", ".mp4", ".wav", ".aac", ".flac", ".ogg", ".webm"}


def _safe_stem(name: str) -> str:
    """Return a filesystem-safe stem without collisions."""
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in Path(name).stem)
    cleaned = cleaned.strip("_")
    return cleaned or "file"


def _iter_directory_audio(
    input_dir: Path, staging_dir: Path
) -> Iterable[Tuple[str, Path, str]]:
    """Yield audio files from a directory, copying into the staging area."""
    used: set[str] = set()
    for source in sorted(input_dir.iterdir()):
        if not source.is_file():
            continue
        ext = source.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue

        stem = _safe_stem(source.name)
        if stem in used:
            index = 1
            while f"{stem}_{index}" in used:
                index += 1
            stem = f"{stem}_{index}"
        used.add(stem)

        dest = staging_dir / f"{stem}{ext}"
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        yield stem, dest, source.name


def _iter_zip_audio(
    archive_path: Path, staging_dir: Path
) -> Iterable[Tuple[str, Path, str]]:
    """Yield audio files from a ZIP archive, extracting into staging area."""
    used: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and Path(member.filename).suffix.lower() in SUPPORTED_EXTS
        ]
        for index, member in enumerate(members, start=1):
            original_name = Path(member.filename).name
            stem = _safe_stem(original_name)
            if stem in used:
                stem = f"{stem}_{index}"
            used.add(stem)

            ext = Path(original_name).suffix.lower() or ".mp3"
            dest = staging_dir / f"{stem}{ext}"
            with dest.open("wb") as handle:
                handle.write(archive.read(member))
            yield stem, dest, original_name


def _transcribe_sources(
    sources: Iterable[Tuple[str, Path, str]],
    target_name: str,
    openai_key: str,
    assemblyai_key: str,
    output_dir: Path,
    aggregate_zip: bool,
) -> Tuple[List[dict], List[str], Path | None]:
    """Run transcription across provided sources."""
    transcripts: List[dict] = []
    errors: List[str] = []
    zip_path: Path | None = None
    zip_handle = None

    if aggregate_zip:
        zip_path = output_dir / "transcripts.zip"
        zip_handle = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

    try:
        for stem, audio_path, display_name in sources:
            logging.info("Transcribing %s", display_name)
            try:
                transcript_text = transcribe_file(
                    str(audio_path),
                    openai_key,
                    assemblyai_key,
                    target_name,
                ).strip()
            except Exception as exc:
                logging.error("Failed to transcribe %s: %s", display_name, exc)
                errors.append(f"{display_name}: {exc}")
                continue

            text_path = output_dir / f"{stem}.txt"
            save_text_file(transcript_text, text_path)

            if zip_handle is not None:
                zip_handle.write(text_path, arcname=f"{stem}.txt")

            transcripts.append(
                {
                    "display_name": display_name,
                    "safe_name": stem,
                    "audio_path": str(audio_path),
                    "text_path": str(text_path),
                }
            )
    finally:
        if zip_handle is not None:
            zip_handle.close()

    return transcripts, errors, zip_path


def _resolve_keys(args) -> Tuple[str, str]:
    """Fetch API keys from CLI args or environment."""
    load_dotenv()
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    assemblyai_key = args.assemblyai_key or os.getenv("ASSEMBLYAI_API_KEY")
    if not openai_key:
        raise SystemExit("Missing OpenAI API key. Provide via --openai-key or OPENAI_API_KEY env var.")
    if not assemblyai_key:
        raise SystemExit(
            "Missing AssemblyAI API key. Provide via --assemblyai-key or ASSEMBLYAI_API_KEY env var."
        )
    return openai_key, assemblyai_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline bulk audio transcription helper.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a directory of audio files, a ZIP archive, or a single audio file.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target name (used for speaker labeling in transcripts).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Destination directory for transcripts (default: ./output).",
    )
    parser.add_argument(
        "--openai-key",
        help="Override OpenAI API key (falls back to OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--assemblyai-key",
        help="Override AssemblyAI API key (falls back to ASSEMBLYAI_API_KEY env var).",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip generating transcripts.zip aggregate.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(message)s",
    )

    openai_key, assemblyai_key = _resolve_keys(args)

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = args.output_dir / f"bulk_{timestamp}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        sources = list(_iter_directory_audio(input_path, staging_dir))
    elif zipfile.is_zipfile(input_path):
        sources = list(_iter_zip_audio(input_path, staging_dir))
    elif input_path.is_file() and input_path.suffix.lower() in SUPPORTED_EXTS:
        # Handle single audio file by copying into staging dir.
        stem = _safe_stem(input_path.name)
        dest = staging_dir / f"{stem}{input_path.suffix.lower()}"
        shutil.copy2(input_path, dest)
        sources = [(stem, dest, input_path.name)]
    else:
        raise SystemExit("Input must be a directory, ZIP archive, or supported audio file.")

    if not sources:
        raise SystemExit("No supported audio files detected.")

    transcripts, errors, zip_path = _transcribe_sources(
        sources,
        args.target,
        openai_key,
        assemblyai_key,
        staging_dir,
        aggregate_zip=not args.no_zip,
    )

    print(f"Processed {len(transcripts)} file(s). Output directory: {staging_dir}")
    if zip_path:
        print(f"Aggregated transcripts archive: {zip_path}")

    if errors:
        print("The following items failed:", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        sys.exit(1 if len(transcripts) == 0 else 0)


if __name__ == "__main__":
    main()
