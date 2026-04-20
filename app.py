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
    persist_audio_segment_wav,
    persist_uploaded_file,
)
from src.transcription import build_transcription, get_transcription_modes, normalize_orthography
from src.ui_helpers import (
    apply_pending_editor_updates,
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
    apply_pending_editor_updates()
    _apply_theme()
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
        "Загрузите аудиофайл, получите орфографическую расшифровку и отдельную транскрипцию, затем при необходимости отредактируйте результат."
    )
    st.markdown(
        "[Репозиторий проекта](https://github.com/lehabuzanov/dialect_rech)"
    )
    with st.expander("Краткая инструкция"):
        st.write(
            "1. Загрузите аудиофайл.\n"
            "2. При необходимости выберите режим распознавания.\n"
            "3. Нажмите `Распознать`.\n"
            "4. При необходимости переключите формат транскрипции.\n"
            "5. Скачайте результаты."
        )


def _render_sidebar() -> None:
    st.sidebar.header("Настройки распознавания")
    theme_choices = {
        "dark_blue": "Тёмно-синяя",
        "light": "Светлая",
    }
    theme_keys = list(theme_choices.keys())
    theme_index = theme_keys.index(st.session_state.get("theme_mode", "dark_blue"))
    st.session_state.theme_mode = st.sidebar.radio(
        "Тема интерфейса",
        options=theme_keys,
        index=theme_index,
        format_func=lambda key: theme_choices[key],
    )

    model_choices = get_model_choices()
    model_keys = list(model_choices.keys())
    current_index = model_keys.index(st.session_state.get("selected_model_key", DEFAULT_MODEL_KEY))
    selected_key = st.sidebar.selectbox(
        "Режим распознавания",
        options=model_keys,
        index=current_index,
        format_func=lambda key: model_choices[key],
        disabled=True,
        help="Оставлен один наиболее стабильный и практичный режим.",
    )
    st.session_state.selected_model_key = selected_key
    st.sidebar.caption(get_model_description(selected_key))

    transcription_modes = get_transcription_modes()
    mode_keys = list(transcription_modes.keys())
    mode_index = mode_keys.index(st.session_state.get("transcription_mode", "ru_practical"))
    selected_mode = st.sidebar.radio(
        "Формат транскрипции",
        options=mode_keys,
        index=mode_index,
        format_func=lambda key: transcription_modes[key],
    )
    previous_mode = st.session_state.get("transcription_mode")
    st.session_state.transcription_mode = selected_mode
    if previous_mode != selected_mode and st.session_state.get("orthography_text", "").strip():
        st.session_state.transcription_text = build_transcription(
            st.session_state.orthography_text,
            selected_mode,
        )


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
    st.session_state.selection_start = 0.0
    st.session_state.selection_end = float(audio_info.duration_seconds)
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

    st.markdown("**Диапазон распознавания**")
    range_values = st.slider(
        "Выберите фрагмент аудио",
        min_value=0.0,
        max_value=float(audio_info.duration_seconds),
        value=(
            float(st.session_state.get("selection_start", 0.0)),
            float(st.session_state.get("selection_end", audio_info.duration_seconds)),
        ),
        step=0.5,
    )
    st.session_state.selection_start = float(range_values[0])
    st.session_state.selection_end = float(range_values[1])

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
    source_audio_path = st.session_state.get("uploaded_file_path")
    model_key = st.session_state.get("selected_model_key", DEFAULT_MODEL_KEY)
    if not source_audio_path:
        set_status("Сначала загрузите аудиофайл.", "warning")
        return

    selection_start = float(st.session_state.get("selection_start", 0.0))
    selection_end = float(st.session_state.get("selection_end", 0.0))
    waveform = st.session_state.get("waveform_data")
    waveform_sr = st.session_state.get("waveform_sr")
    if waveform is None or waveform_sr is None:
        set_status("Не удалось подготовить аудио для распознавания.", "error")
        return

    progress_bar = st.progress(0, text="Подготовка распознавания")
    progress_status = st.empty()
    segment_path = None

    def update_progress(percent: int, message: str) -> None:
        progress_bar.progress(percent, text=message)
        progress_status.caption(message)

    try:
        update_progress(2, "Подготовка выбранного фрагмента")
        segment_path = persist_audio_segment_wav(
            waveform,
            waveform_sr,
            selection_start,
            selection_end,
        )
        result = transcribe_audio(
            segment_path,
            model_key=model_key,
            duration_seconds=max(0.1, selection_end - selection_start),
            progress_callback=update_progress,
        )
        orthography = normalize_orthography(str(result["text"]))
        transcription = build_transcription(orthography, st.session_state.transcription_mode)

        st.session_state.orthography_text = orthography
        st.session_state.pending_orthography_editor = orthography
        st.session_state.transcription_text = transcription
        st.session_state.pending_transcription_editor = transcription
        st.session_state.segments = result["segments"]
        st.session_state.last_processed_model_key = model_key
        st.session_state.result_timestamp = datetime.now().isoformat()
        progress_bar.progress(100, text="Готово")
        set_status(f"Распознавание завершено. Использована модель: {result['model_label']}.", "success")
        st.rerun()
    except Exception as exc:
        progress_bar.empty()
        progress_status.empty()
        set_status(
            f"Ошибка распознавания: {exc}.",
            "error",
        )
        cleanup_temp_file(segment_path)
        return
    cleanup_temp_file(segment_path)


def _clear_text_results() -> None:
    st.session_state.orthography_text = ""
    st.session_state.pending_orthography_editor = ""
    st.session_state.transcription_text = ""
    st.session_state.pending_transcription_editor = ""
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
        key="orthography_editor",
    )

    if st.button("Обновить транскрипцию из орфографии", width="stretch"):
        normalized = normalize_orthography(st.session_state.orthography_editor)
        st.session_state.orthography_text = normalized
        st.session_state.pending_orthography_editor = normalized
        st.session_state.transcription_text = build_transcription(
            normalized,
            st.session_state.transcription_mode,
        )
        st.session_state.pending_transcription_editor = st.session_state.transcription_text
        set_status("Транскрипция обновлена из орфографической записи.", "success")
        st.rerun()

    st.text_area(
        "Транскрипция",
        height=220,
        key="transcription_editor",
    )

    segments = st.session_state.get("segments", [])
    if segments:
        with st.expander("Сегменты распознавания"):
            st.dataframe(segments, width="stretch", hide_index=True)

    _render_downloads()


def _render_downloads() -> None:
    orthography = st.session_state.get("orthography_editor", "")
    transcription = st.session_state.get("transcription_editor", "")
    st.session_state.orthography_text = orthography
    st.session_state.transcription_text = transcription
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


def _apply_theme() -> None:
    theme_mode = st.session_state.get("theme_mode", "dark_blue")
    if theme_mode == "light":
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
                color: #0f172a;
            }
            [data-testid="stSidebar"] {
                background: #ffffff;
            }
            [data-testid="stTextArea"] textarea {
                background: #e9eef6;
                color: #0f172a;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 20%, rgba(37, 99, 235, 0.18) 0, rgba(37, 99, 235, 0.03) 22%, transparent 23%),
                radial-gradient(circle at 78% 16%, rgba(59, 130, 246, 0.18) 0, rgba(59, 130, 246, 0.03) 24%, transparent 25%),
                linear-gradient(180deg, #020617 0%, #071224 50%, #0b1120 100%);
            color: #e5eefb;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.16;
            background-image:
                url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 900'><path d='M0 160 C120 130 160 210 280 180 S460 120 580 170 760 230 880 190 1040 120 1160 170 1320 230 1440 180' fill='none' stroke='%233b82f6' stroke-width='3'/><path d='M0 280 C90 320 180 210 300 250 S510 330 630 285 810 220 930 265 1110 330 1230 285 1350 220 1440 255' fill='none' stroke='%2360a5fa' stroke-width='2.2'/><path d='M0 420 C110 380 210 470 330 420 S520 365 640 430 820 490 940 440 1120 360 1240 420 1360 500 1440 455' fill='none' stroke='%2393c5fd' stroke-width='2'/></svg>");
            background-repeat: no-repeat;
            background-size: cover;
            mix-blend-mode: screen;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stSidebar"] {
            background: rgba(7, 18, 36, 0.88);
            border-right: 1px solid rgba(96, 165, 250, 0.18);
        }
        [data-testid="stSidebar"] * {
            color: #dbeafe;
        }
        [data-testid="stTextArea"] textarea {
            background: rgba(15, 23, 42, 0.88);
            color: #eff6ff;
            border: 1px solid rgba(96, 165, 250, 0.28);
        }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"], .stMarkdown, .stCaption, label, p, h1, h2, h3 {
            color: #e5eefb !important;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(96, 165, 250, 0.35);
            box-shadow: 0 8px 30px rgba(2, 6, 23, 0.35);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
