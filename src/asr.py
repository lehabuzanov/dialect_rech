from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel


DEFAULT_MODEL_KEY = "balanced_medium"
APP_CACHE_ROOT = Path(tempfile.gettempdir()) / "dialect_rech_cache"
MODEL_CACHE_ROOT = APP_CACHE_ROOT / "models"
HF_CACHE_ROOT = APP_CACHE_ROOT / "huggingface"

HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
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
    "turbo_fast": ModelConfig(
        key="turbo_fast",
        label="Turbo: быстро и качественно",
        model_id="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        description="Самый быстрый вариант высокого класса. Хорош для длинных файлов и облачного запуска.",
        beam_size=2,
        chunk_length=24,
    ),
    "quality_large": ModelConfig(
        key="quality_large",
        label="Large v3: максимум качества",
        model_id="Systran/faster-whisper-large-v3",
        description="Максимальное качество распознавания, но требует больше памяти и работает медленнее.",
        beam_size=2,
        chunk_length=20,
    ),
    "balanced_medium": ModelConfig(
        key="balanced_medium",
        label="Medium: баланс скорости и точности",
        model_id="Systran/faster-whisper-medium",
        description="Рекомендуемый режим по умолчанию. Обычно стабильнее всего на Streamlit Cloud.",
        beam_size=3,
        chunk_length=28,
    ),
    "fast_small": ModelConfig(
        key="fast_small",
        label="Small: самый лёгкий режим",
        model_id="Systran/faster-whisper-small",
        description="Подходит для слабых машин и быстрой черновой расшифровки.",
        beam_size=3,
        chunk_length=30,
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
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
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


def transcribe_audio(audio_path: str, model_key: str) -> dict[str, object]:
    if not os.path.exists(audio_path):
        raise FileNotFoundError("Временный аудиофайл не найден.")

    model_config = get_model_config(model_key)
    model = load_whisper_model(model_key)
    segments, info = model.transcribe(
        audio_path,
        language="ru",
        task="transcribe",
        beam_size=model_config.beam_size,
        best_of=max(3, model_config.beam_size),
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=False,
        without_timestamps=False,
        chunk_length=model_config.chunk_length,
    )

    segment_items = []
    texts = []
    for segment in segments:
        cleaned = segment.text.strip()
        if not cleaned:
            continue
        texts.append(cleaned)
        segment_items.append(
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": cleaned,
            }
        )

    return {
        "text": " ".join(texts).strip(),
        "segments": segment_items,
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "model_label": model_config.label,
    }
