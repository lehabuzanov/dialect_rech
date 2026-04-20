from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from ctranslate2.converters import TransformersConverter
from faster_whisper import BatchedInferencePipeline, WhisperModel


DEFAULT_MODEL_KEY = "whisper_large_v3_turbo"
MODEL_CACHE_ROOT = Path.home() / ".cache" / "dialect_rech" / "models"
CT2_CACHE_ROOT = MODEL_CACHE_ROOT / "ctranslate2"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    key: str
    label: str
    source_type: str
    model_id: str
    description: str


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "whisper_large_v3_turbo": ModelConfig(
        key="whisper_large_v3_turbo",
        label="openai/whisper-large-v3-turbo",
        source_type="hf_transformers_convert",
        model_id="openai/whisper-large-v3-turbo",
        description="Максимально сильная модель по умолчанию. При первом запуске локально конвертируется в формат faster-whisper.",
    ),
    "faster_whisper_large_v3": ModelConfig(
        key="faster_whisper_large_v3",
        label="Systran/faster-whisper-large-v3",
        source_type="ct2_direct",
        model_id="Systran/faster-whisper-large-v3",
        description="Тяжёлая альтернативная модель large-v3 без этапа локальной конвертации.",
    ),
    "faster_whisper_medium": ModelConfig(
        key="faster_whisper_medium",
        label="Systran/faster-whisper-medium",
        source_type="ct2_direct",
        model_id="Systran/faster-whisper-medium",
        description="Более лёгкий fallback для слабых ПК.",
    ),
    "faster_whisper_small": ModelConfig(
        key="faster_whisper_small",
        label="Systran/faster-whisper-small",
        source_type="ct2_direct",
        model_id="Systran/faster-whisper-small",
        description="Самый быстрый из встроенных fallback-вариантов, но менее точный.",
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
def load_whisper_pipeline(model_key: str):
    model_config = get_model_config(model_key)
    device, compute_type = detect_runtime()
    model_path = prepare_model_path(model_config)
    whisper_model = WhisperModel(model_path, device=device, compute_type=compute_type)
    return BatchedInferencePipeline(model=whisper_model)


def prepare_model_path(model_config: ModelConfig) -> str:
    if model_config.source_type == "ct2_direct":
        return model_config.model_id
    if model_config.source_type == "hf_transformers_convert":
        return ensure_converted_model(model_config)
    raise ValueError(f"Неподдерживаемый source_type: {model_config.source_type}")


def ensure_converted_model(model_config: ModelConfig) -> str:
    device, _ = detect_runtime()
    quantization = "float16" if device == "cuda" else "int8"
    CT2_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    model_dir = CT2_CACHE_ROOT / f"{model_config.model_id.replace('/', '--')}--{quantization}"
    model_bin = model_dir / "model.bin"
    tokenizer = model_dir / "tokenizer.json"

    if model_bin.exists() and tokenizer.exists():
        return str(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    converter = TransformersConverter(model_name_or_path=model_config.model_id)
    converter.convert(
        output_dir=str(model_dir),
        quantization=quantization,
        copy_files=["tokenizer.json", "preprocessor_config.json"],
        force=False,
    )
    return str(model_dir)


def transcribe_audio(
    audio_path: str,
    model_key: str,
    beam_size: int = 5,
    batch_size: int = 8,
) -> dict[str, object]:
    if not os.path.exists(audio_path):
        raise FileNotFoundError("Временный аудиофайл не найден.")

    pipeline = load_whisper_pipeline(model_key)
    segments, info = pipeline.transcribe(
        audio_path,
        language="ru",
        task="transcribe",
        beam_size=beam_size,
        best_of=max(beam_size, 5),
        temperature=0.0,
        vad_filter=True,
        word_timestamps=True,
        batch_size=batch_size,
        condition_on_previous_text=False,
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

    joined_text = " ".join(texts).strip()
    return {
        "text": joined_text,
        "segments": segment_items,
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
    }
