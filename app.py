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
from src.transcription import normalize_orthography, practical_transcription, transcription_to_tts_text
from src.tts import synthesize_speech_to_wav_bytes
from src.ui_helpers import (
    build_export_payload,
    clear_results,
    init_session_state,
    render_audio_info,
    render_status_box,
    set_status,
)


st.set_page_config(
    page_title="Диалектная речь: орфография и транскрипция",
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
    st.title("Транскрипция русской диалектной речи")
    st.caption(
        "Локальное Streamlit-приложение для распознавания загруженного аудиофайла, "
        "получения орфографической записи и русской практической транскрипции."
    )


def _render_sidebar() -> None:
    st.sidebar.header("Настройки распознавания")
    model_choices = get_model_choices()
    model_keys = list(model_choices.keys())
    current_index = model_keys.index(st.session_state.get("selected_model_key", DEFAULT_MODEL_KEY))
    selected_key = st.sidebar.selectbox(
        "Модель ASR",
        options=model_keys,
        index=current_index,
        format_func=lambda key: model_choices[key],
        help="Выбранная модель реально используется при нажатии кнопки распознавания.",
    )
    st.session_state.selected_model_key = selected_key
    st.sidebar.caption(get_model_description(selected_key))

    st.sidebar.markdown("**Ограничения**")
    st.sidebar.write(
        "- Только загрузка файла, без микрофона.\n"
        "- Язык распознавания жёстко фиксирован: `ru`.\n"
        "- Перевод отключён, используется только `transcribe`."
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
        st.plotly_chart(figure, use_container_width=True)


def _render_actions() -> None:
    if not st.session_state.get("uploaded_file_path"):
        return

    col1, col2 = st.columns([1, 1])

    if col1.button("Распознать", type="primary", use_container_width=True):
        _run_recognition()

    if col2.button("Очистить результаты", use_container_width=True):
        _clear_text_results()


def _run_recognition() -> None:
    audio_path = st.session_state.get("uploaded_file_path")
    model_key = st.session_state.get("selected_model_key", DEFAULT_MODEL_KEY)
    if not audio_path:
        set_status("Сначала загрузите аудиофайл.", "warning")
        return

    try:
        set_status("Идёт распознавание речи. Это может занять некоторое время.", "info")
        with st.spinner("Распознаю аудио локально через faster-whisper..."):
            result = transcribe_audio(audio_path, model_key=model_key)
        orthography = normalize_orthography(str(result["text"]))
        transcription = practical_transcription(orthography)

        st.session_state.orthography_text = orthography
        st.session_state.transcription_text = transcription
        st.session_state.segments = result["segments"]
        st.session_state.last_processed_model_key = model_key
        st.session_state.result_timestamp = datetime.now().isoformat()
        st.session_state.tts_orthography_bytes = None
        st.session_state.tts_transcription_bytes = None
        set_status("Распознавание завершено.", "success")
    except Exception as exc:
        set_status(f"Ошибка распознавания: {exc}", "error")


def _clear_text_results() -> None:
    st.session_state.orthography_text = ""
    st.session_state.transcription_text = ""
    st.session_state.segments = []
    st.session_state.result_timestamp = None
    st.session_state.tts_orthography_bytes = None
    st.session_state.tts_transcription_bytes = None
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

    col1, col2 = st.columns([1, 1])
    if col1.button("Обновить транскрипцию из орфографии", use_container_width=True):
        normalized = normalize_orthography(st.session_state.orthography_text)
        st.session_state.orthography_text = normalized
        st.session_state.transcription_text = practical_transcription(normalized)
        set_status("Транскрипция обновлена из орфографической записи.", "success")

    if col2.button("Прослушать орфографию", use_container_width=True):
        _synthesize_text("orthography")

    if st.session_state.get("tts_orthography_bytes"):
        st.audio(st.session_state.tts_orthography_bytes, format="audio/wav")

    st.text_area(
        "Транскрипция",
        height=220,
        key="transcription_text",
    )

    col3, col4 = st.columns([1, 1])
    if col3.button("Прослушать транскрипцию", use_container_width=True):
        _synthesize_text("transcription")

    if col4.button("Показать сегменты распознавания", use_container_width=True):
        _render_segments()

    if st.session_state.get("tts_transcription_bytes"):
        st.audio(st.session_state.tts_transcription_bytes, format="audio/wav")

    _render_downloads()


def _render_segments() -> None:
    segments = st.session_state.get("segments", [])
    if not segments:
        st.info("Сегменты пока не доступны.")
        return
    st.json(segments)


def _synthesize_text(kind: str) -> None:
    if kind == "orthography":
        text = st.session_state.get("orthography_text", "")
    else:
        text = transcription_to_tts_text(st.session_state.get("transcription_text", ""))

    if not text.strip():
        set_status("Нет текста для озвучивания.", "warning")
        return

    try:
        set_status("Идёт синтез речи.", "info")
        with st.spinner("Синтезирую речь локально через Silero TTS..."):
            audio_bytes = synthesize_speech_to_wav_bytes(text)
        if kind == "orthography":
            st.session_state.tts_orthography_bytes = audio_bytes
        else:
            st.session_state.tts_transcription_bytes = audio_bytes
        set_status("Синтез речи завершён.", "success")
    except Exception as exc:
        set_status(f"Ошибка синтеза речи: {exc}", "error")


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
        use_container_width=True,
    )
    col2.download_button(
        "Скачать транскрипцию (.txt)",
        data=transcription.encode("utf-8"),
        file_name="transcription.txt",
        mime="text/plain",
        use_container_width=True,
    )
    col3.download_button(
        "Скачать всё (.json)",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="dialect_transcription_result.json",
        mime="application/json",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
