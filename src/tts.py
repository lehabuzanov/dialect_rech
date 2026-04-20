from __future__ import annotations

import io
import re

import numpy as np
import soundfile as sf
import streamlit as st
from silero import silero_tts


DEFAULT_TTS_MODEL = "v5_ru"
DEFAULT_SPEAKER = "xenia"
DEFAULT_SAMPLE_RATE = 48000


def normalize_text_for_tts(text: str) -> str:
    cleaned = text.replace("\n", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[^0-9А-Яа-яЁё ,.!?;:\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


@st.cache_resource(show_spinner=False)
def load_tts_model(language: str = "ru", speaker_model: str = DEFAULT_TTS_MODEL):
    model, _ = silero_tts(language=language, speaker=speaker_model)
    return model


def synthesize_speech_to_wav_bytes(
    text: str,
    speaker: str = DEFAULT_SPEAKER,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    normalized = normalize_text_for_tts(text)
    if not normalized:
        raise ValueError("Нет текста для озвучивания.")

    model = load_tts_model()
    audio = model.apply_tts(
        text=normalized,
        speaker=speaker,
        sample_rate=sample_rate,
    )
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()

    audio_array = np.asarray(audio, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    return buffer.getvalue()

