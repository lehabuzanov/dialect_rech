from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import streamlit as st
from faster_whisper import WhisperModel


DEFAULT_MODEL_KEY = "balanced"
APP_CACHE_ROOT = Path(tempfile.gettempdir()) / "dialect_rech_cache"
MODEL_CACHE_ROOT = APP_CACHE_ROOT / "models"
HF_CACHE_ROOT = APP_CACHE_ROOT / "huggingface"

HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))


@dataclass(frozen=True, slots=True)
class ModelConfig:
    key: str
    label: str
    model_id: str
    description: str
    beam_size: int
    chunk_length: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "balanced": ModelConfig(
        key="balanced",
        label="Стабильная модель",
        model_id="Systran/faster-whisper-medium",
        description="Основной практичный режим: лучший баланс качества, скорости и стабильности для Streamlit Cloud.",
        beam_size=3,
        chunk_length=24,
    ),
}


def get_model_choices() -> dict[str, str]:
    return {key: config.label for key, config in MODEL_CONFIGS.items()}


def get_model_description(model_key: str) -> str:
    return MODEL_CONFIGS[model_key].description


def get_model_config(model_key: str) -> ModelConfig:
    try:
        return MODEL_CONFIGS[model_key]
    except KeyError as exc:
        raise ValueError(f"Неизвестная модель: {model_key}") from exc


def detect_runtime() -> tuple[str, str]:
    return "cpu", "int8"


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_key: str) -> WhisperModel:
    model_config = get_model_config(model_key)
    device, compute_type = detect_runtime()
    cpu_threads = max(1, min(4, os.cpu_count() or 1))

    return WhisperModel(
        model_config.model_id,
        device=device,
        compute_type=compute_type,
        download_root=str(MODEL_CACHE_ROOT),
        cpu_threads=cpu_threads,
        num_workers=1,
    )


def transcribe_audio(
    audio_path: str,
    model_key: str,
    duration_seconds: float,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, object]:
    if not os.path.exists(audio_path):
        raise FileNotFoundError("Временный аудиофайл не найден.")

    model_config = get_model_config(model_key)
    if progress_callback:
        progress_callback(5, "Загрузка модели распознавания")

    model = load_whisper_model(model_key)
    if progress_callback:
        progress_callback(15, "Модель загружена, начинаю распознавание")

    segments, info = model.transcribe(
        audio_path,
        language="ru",
        task="transcribe",
        beam_size=model_config.beam_size,
        best_of=max(2, model_config.beam_size),
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=False,
        without_timestamps=False,
        chunk_length=model_config.chunk_length,
    )

    segment_items = []
    texts = []
    total_duration = max(duration_seconds, 0.1)

    for segment in segments:
        cleaned = segment.text.strip()
        if cleaned:
            texts.append(cleaned)
            segment_items.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": cleaned,
                }
            )

        if progress_callback:
            ratio = min(1.0, float(segment.end) / total_duration)
            percent = min(95, 15 + int(ratio * 80))
            progress_callback(percent, f"Распознавание: {percent}%")

    if progress_callback:
        progress_callback(100, "Распознавание завершено")

    return {
        "text": " ".join(texts).strip(),
        "segments": segment_items,
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "model_label": model_config.label,
    }
