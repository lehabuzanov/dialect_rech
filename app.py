from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from src.asr import DEFAULT_MODEL_KEY, get_model_choices, get_model_description, transcribe_audio
from src.audio_utils import (
    build_waveform_figure,
    cleanup_temp_file,
    decode_audio_waveform,
    inspect_audio_bytes,
    persist_uploaded_file,
)
from src.transcription import normalize_orthography, practical_transcription
from src.ui_helpers import (
    build_export_payload,
    clear_results,
    init_session_state,
    render_audio_info,
    render_status_box,
    set_status,
)


st.set_page_config(
    page_title="Диалектная речь: расшифровка и транскрипция",
    page_icon="🎙️",
    layout="wide",
)


def main() -> None:
    init_session_state()
    _render_header()
    _render_sidebar()
    render_status_box()
    _handle_file_upload()
    _render_loaded_audio()
    _render_actions()
    _render_results()


def _render_header() -> None:
    st.title("Диалектная речь: орфография и транскрипция")
    st.caption(
        "Загрузите аудиофайл, получите черновую орфографическую расшифровку и "
        "автоматическую широкую фонетическую транскрипцию, затем при необходимости отредактируйте их вручную."
    )


def _render_sidebar() -> None:
    st.sidebar.header("Настройки распознавания")
    model_choices = get_model_choices()
    model_keys = list(model_choices.keys())
    current_index = model_keys.index(st.session_state.get("selected_model_key", DEFAULT_MODEL_KEY))
    selected_key = st.sidebar.selectbox(
        "Модель распознавания",
        options=model_keys,
        index=current_index,
        format_func=lambda key: model_choices[key],
        help="Выбранная модель действительно используется при нажатии кнопки распознавания.",
    )
    st.session_state.selected_model_key = selected_key
    st.sidebar.caption(get_model_description(selected_key))


def _handle_file_upload() -> None:
    uploaded = st.file_uploader(
        "Загрузите аудиофайл",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        return

    if uploaded.name == st.session_state.get("uploaded_file_name"):
        return

    previous_path = st.session_state.get("uploaded_file_path")
    cleanup_temp_file(previous_path)
    file_bytes = uploaded.getvalue()

    try:
        audio_info = inspect_audio_bytes(file_bytes, uploaded.name)
        waveform, waveform_sr = decode_audio_waveform(file_bytes)
        temp_path = persist_uploaded_file(file_bytes, audio_info.suffix)
    except Exception as exc:
        clear_results(keep_uploaded_file=False)
        set_status(f"Ошибка загрузки файла: {exc}", "error")
        return

    clear_results(keep_uploaded_file=False)
    st.session_state.uploaded_file_name = uploaded.name
    st.session_state.uploaded_file_bytes = file_bytes
    st.session_state.uploaded_file_path = temp_path
    st.session_state.audio_info = audio_info
    st.session_state.waveform_data = waveform
    st.session_state.waveform_sr = waveform_sr
    set_status("Файл загружен и готов к распознаванию.", "success")


def _render_loaded_audio() -> None:
    audio_info = st.session_state.get("audio_info")
    file_bytes = st.session_state.get("uploaded_file_bytes")
    waveform = st.session_state.get("waveform_data")
    waveform_sr = st.session_state.get("waveform_sr")

    if not audio_info or not file_bytes:
        return

    st.subheader("Исходный файл")
    render_audio_info(audio_info)
    st.audio(file_bytes, format=audio_info.mime_type)

    if waveform is not None and waveform_sr:
        figure = build_waveform_figure(waveform, waveform_sr)
        st.plotly_chart(figure, width="stretch")


def _render_actions() -> None:
    if not st.session_state.get("uploaded_file_path"):
        return

    col1, col2 = st.columns([1, 1])
    if col1.button("Распознать", type="primary", width="stretch"):
        _run_recognition()
    if col2.button("Очистить результаты", width="stretch"):
        _clear_text_results()


def _run_recognition() -> None:
    audio_path = st.session_state.get("uploaded_file_path")
    model_key = st.session_state.get("selected_model_key", DEFAULT_MODEL_KEY)
    if not audio_path:
        set_status("Сначала загрузите аудиофайл.", "warning")
        return

    try:
        with st.spinner("Распознаю аудио..."):
            result = transcribe_audio(audio_path, model_key=model_key)
        orthography = normalize_orthography(str(result["text"]))
        transcription = practical_transcription(orthography)

        st.session_state.orthography_text = orthography
        st.session_state.transcription_text = transcription
        st.session_state.segments = result["segments"]
        st.session_state.last_processed_model_key = model_key
        st.session_state.result_timestamp = datetime.now().isoformat()
        set_status(f"Распознавание завершено. Использована модель: {result['model_label']}.", "success")
    except Exception as exc:
        set_status(
            f"Ошибка распознавания: {exc}. Попробуйте более лёгкую модель из списка.",
            "error",
        )


def _clear_text_results() -> None:
    st.session_state.orthography_text = ""
    st.session_state.transcription_text = ""
    st.session_state.segments = []
    st.session_state.result_timestamp = None
    set_status("Результаты очищены. Загруженный файл сохранён.", "info")


def _render_results() -> None:
    if not st.session_state.get("audio_info"):
        return

    st.subheader("Результаты")
    st.text_area(
        "Орфографическая запись",
        height=220,
        key="orthography_text",
    )

    if st.button("Обновить транскрипцию из орфографии", width="stretch"):
        normalized = normalize_orthography(st.session_state.orthography_text)
        st.session_state.orthography_text = normalized
        st.session_state.transcription_text = practical_transcription(normalized)
        set_status("Транскрипция обновлена из орфографической записи.", "success")

    st.text_area(
        "Транскрипция (широкая фонетическая)",
        height=220,
        key="transcription_text",
    )

    segments = st.session_state.get("segments", [])
    if segments:
        with st.expander("Сегменты распознавания"):
            st.dataframe(segments, width="stretch", hide_index=True)

    _render_downloads()


def _render_downloads() -> None:
    orthography = st.session_state.get("orthography_text", "")
    transcription = st.session_state.get("transcription_text", "")
    payload = build_export_payload()

    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Скачать орфографию (.txt)",
        data=orthography.encode("utf-8"),
        file_name="orthography.txt",
        mime="text/plain",
        width="stretch",
    )
    col2.download_button(
        "Скачать транскрипцию (.txt)",
        data=transcription.encode("utf-8"),
        file_name="transcription.txt",
        mime="text/plain",
        width="stretch",
    )
    col3.download_button(
        "Скачать всё (.json)",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="dialect_transcription_result.json",
        mime="application/json",
        width="stretch",
    )


if __name__ == "__main__":
    main()
