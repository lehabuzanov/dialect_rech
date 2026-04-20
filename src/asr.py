from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
    "light": ModelConfig(
        key="light",
        label="Лёгкая: быстро",
        model_id="Systran/faster-whisper-small",
        description="Самый лёгкий и быстрый режим. Подходит для черновой расшифровки и длинных файлов.",
        beam_size=2,
        chunk_length=28,
    ),
    "balanced": ModelConfig(
        key="balanced",
        label="Сбалансированная: качество и стабильность",
        model_id="Systran/faster-whisper-medium",
        description="Рекомендуемый режим по умолчанию. Лучший баланс точности, скорости и стабильности.",
        beam_size=3,
        chunk_length=24,
    ),
    "accurate": ModelConfig(
        key="accurate",
        label="Точная: улучшенное качество",
        model_id="Systran/faster-whisper-large-v2",
        description="Более тяжёлая multilingual-модель для лучшего качества. Медленнее, но стабильнее turbo-вариантов.",
        beam_size=2,
        chunk_length=18,
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
    try:
        model = load_whisper_model(model_key)
        segments, info = _transcribe_with_model(model, audio_path, model_config)
        used_model = model_config
    except Exception:
        if model_key == DEFAULT_MODEL_KEY:
            raise
        fallback_config = get_model_config(DEFAULT_MODEL_KEY)
        model = load_whisper_model(DEFAULT_MODEL_KEY)
        segments, info = _transcribe_with_model(model, audio_path, fallback_config)
        used_model = fallback_config

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
        "model_label": used_model.label,
    }


def _transcribe_with_model(model: WhisperModel, audio_path: str, model_config: ModelConfig):
    return model.transcribe(
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
