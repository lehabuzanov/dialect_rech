from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from src.audio_utils import AudioFileInfo, format_duration, format_size


SESSION_DEFAULTS: dict[str, Any] = {
    "status_type": "info",
    "status_message": "Загрузите аудиофайл, затем запустите распознавание.",
    "uploaded_file_name": None,
    "uploaded_file_bytes": None,
    "uploaded_file_path": None,
    "audio_info": None,
    "waveform_data": None,
    "waveform_sr": None,
    "selection_start": 0.0,
    "selection_end": 0.0,
    "orthography_text": "",
    "transcription_text": "",
    "segments": [],
    "selected_model_key": "balanced",
    "transcription_mode": "ru_practical",
    "last_processed_model_key": None,
    "result_timestamp": None,
}


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def set_status(message: str, status_type: str = "info") -> None:
    st.session_state.status_message = message
    st.session_state.status_type = status_type


def render_status_box() -> None:
    status_type = st.session_state.get("status_type", "info")
    message = st.session_state.get("status_message", "")

    if status_type == "success":
        st.success(message)
    elif status_type == "warning":
        st.warning(message)
    elif status_type == "error":
        st.error(message)
    else:
        st.info(message)


def render_audio_info(info: AudioFileInfo) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Файл", info.filename)
    col2.metric("Размер", format_size(info.size_bytes))
    col3.metric("Длительность", format_duration(info.duration_seconds))
    sample_rate = f"{info.sample_rate} Гц" if info.sample_rate else "не определена"
    col4.metric("Частота", sample_rate)

    if info.channels:
        st.caption(f"Каналы: {info.channels}. Формат: {info.suffix.lstrip('.').upper()}.")


def build_export_payload() -> dict[str, Any]:
    audio_info = st.session_state.get("audio_info")
    return {
        "source_file": audio_info.filename if audio_info else None,
        "processed_at": st.session_state.get("result_timestamp") or datetime.now().isoformat(),
        "model_key": st.session_state.get("last_processed_model_key"),
        "orthography": st.session_state.get("orthography_text", ""),
        "transcription": st.session_state.get("transcription_text", ""),
        "segments": st.session_state.get("segments", []),
    }


def clear_results(keep_uploaded_file: bool = True) -> None:
    preserved = {
        "selected_model_key": st.session_state.get("selected_model_key"),
    }
    if keep_uploaded_file:
        for key in ("uploaded_file_name", "uploaded_file_bytes", "uploaded_file_path", "audio_info", "waveform_data", "waveform_sr"):
            preserved[key] = st.session_state.get(key)

    for key, value in SESSION_DEFAULTS.items():
        st.session_state[key] = value

    for key, value in preserved.items():
        st.session_state[key] = value
