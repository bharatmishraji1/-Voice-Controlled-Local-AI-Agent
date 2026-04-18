"""
stt_module.py
-------------
Handles audio file processing and microphone transcription using OpenAI Whisper.
Supports both WAV file uploads and live microphone recording.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

import whisper
import numpy as np
import sounddevice as sd
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000       # Whisper expects 16kHz audio
CHANNELS = 1              # Mono audio
DEFAULT_DURATION = 10     # Default recording duration in seconds
WHISPER_MODEL = "base"    # Whisper model size


def load_whisper_model(model_name: str = WHISPER_MODEL) -> whisper.Whisper:
    """
    Load and return the Whisper model.

    Args:
        model_name: Name of the Whisper model to load (e.g., 'base', 'small').

    Returns:
        Loaded Whisper model instance.
    """
    logger.info(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    logger.info("Whisper model loaded successfully.")
    return model


def transcribe_audio_file(file_path: str, model: Optional[whisper.Whisper] = None) -> str:
    """
    Transcribe an audio file using Whisper.

    Args:
        file_path: Path to the audio file (.wav or other supported format).
        model: Pre-loaded Whisper model. Loads 'base' model if None.

    Returns:
        Transcribed text string.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If transcription fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if model is None:
        model = load_whisper_model()

    logger.info(f"Transcribing audio file: {file_path}")
    try:
        result = model.transcribe(str(path), fp16=False)
        transcription = result.get("text", "").strip()
        logger.info(f"Transcription complete: '{transcription}'")
        return transcription
    except Exception as exc:
        logger.error(f"Transcription failed: {exc}")
        raise RuntimeError(f"Transcription failed: {exc}") from exc


def record_from_microphone(duration: int = DEFAULT_DURATION) -> str:
    """
    Record audio from the system microphone and return the transcribed text.

    Args:
        duration: Recording duration in seconds.

    Returns:
        Transcribed text string from the microphone input.

    Raises:
        RuntimeError: If recording or transcription fails.
    """
    logger.info(f"Recording from microphone for {duration} seconds...")
    try:
        audio_data = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()  # Block until recording is complete
        logger.info("Recording complete.")

        # Save to a temporary WAV file for Whisper processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_data, SAMPLE_RATE)

        transcription = transcribe_audio_file(tmp_path)

    except Exception as exc:
        logger.error(f"Microphone recording failed: {exc}")
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc
    finally:
        # Clean up temp file
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return transcription


def transcribe_bytes(audio_bytes: bytes, model: Optional[whisper.Whisper] = None) -> str:
    """
    Transcribe audio from raw bytes (e.g., from Streamlit file uploader).

    Args:
        audio_bytes: Raw audio bytes from an uploaded file.
        model: Pre-loaded Whisper model. Loads 'base' model if None.

    Returns:
        Transcribed text string.

    Raises:
        RuntimeError: If transcription fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        transcription = transcribe_audio_file(tmp_path, model=model)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return transcription
