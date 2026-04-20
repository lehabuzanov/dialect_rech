from __future__ import annotations

import io
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import av
import numpy as np
import plotly.graph_objects as go


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass(slots=True)
class AudioFileInfo:
    filename: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int | None
    channels: int | None
    suffix: str
    mime_type: str


def get_mime_type(suffix: str) -> str:
    mapping = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }
    return mapping.get(suffix.lower(), "audio/wav")


def ensure_supported_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Неподдерживаемый формат файла: {suffix or 'без расширения'}. "
            f"Поддерживаются: {supported}."
        )
    return suffix


def inspect_audio_bytes(file_bytes: bytes, filename: str) -> AudioFileInfo:
    if not file_bytes:
        raise ValueError("Файл пустой.")

    suffix = ensure_supported_extension(filename)

    try:
        with av.open(io.BytesIO(file_bytes)) as container:
            audio_stream = _get_audio_stream(container.streams.audio)
            duration_seconds = _resolve_duration_seconds(container, audio_stream)
            sample_rate = getattr(audio_stream.codec_context, "sample_rate", None)
            channels = getattr(audio_stream.codec_context, "channels", None)
    except av.AVError as exc:
        raise ValueError("Не удалось прочитать аудиофайл. Проверьте формат и целостность файла.") from exc

    return AudioFileInfo(
        filename=filename,
        size_bytes=len(file_bytes),
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        suffix=suffix,
        mime_type=get_mime_type(suffix),
    )


def decode_audio_waveform(
    file_bytes: bytes,
    target_sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:
    try:
        with av.open(io.BytesIO(file_bytes)) as container:
            resampler = av.audio.resampler.AudioResampler(
                format="fltp",
                layout="mono",
                rate=target_sample_rate,
            )
            chunks: list[np.ndarray] = []
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                if resampled is None:
                    continue
                frames = resampled if isinstance(resampled, list) else [resampled]
                for current in frames:
                    array = current.to_ndarray()
                    if array.ndim == 2:
                        mono = array[0]
                    else:
                        mono = array
                    chunks.append(mono.astype(np.float32, copy=False))
    except av.AVError as exc:
        raise ValueError("Не удалось декодировать аудио для построения волны.") from exc

    if not chunks:
        raise ValueError("Аудиофайл не содержит доступных для декодирования данных.")

    waveform = np.concatenate(chunks)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform = waveform / peak
    return waveform, target_sample_rate


def downsample_waveform(waveform: np.ndarray, max_points: int = 5000) -> np.ndarray:
    if waveform.size <= max_points:
        return waveform

    step = math.ceil(waveform.size / max_points)
    trimmed = waveform[: waveform.size - (waveform.size % step)]
    if trimmed.size == 0:
        return waveform[:max_points]
    reduced = trimmed.reshape(-1, step)
    return reduced.mean(axis=1)


def build_waveform_figure(
    waveform: np.ndarray,
    sample_rate: int,
    title: str = "Звуковая волна",
) -> go.Figure:
    reduced = downsample_waveform(waveform)
    time_axis = np.linspace(0, waveform.size / sample_rate, num=reduced.size)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=time_axis,
            y=reduced,
            mode="lines",
            line={"color": "#0F766E", "width": 1.5},
            hovertemplate="Время: %{x:.2f} сек<br>Амплитуда: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=260,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="#F8FAFC",
        xaxis_title="Время, сек",
        yaxis_title="Амплитуда",
    )
    return figure


def persist_uploaded_file(file_bytes: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        return tmp_file.name


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} Б"

    units = ["КБ", "МБ", "ГБ"]
    value = float(size_bytes)
    for unit in units:
        value /= 1024
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
    return f"{size_bytes} Б"


def format_duration(duration_seconds: float) -> str:
    total_seconds = int(round(duration_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def cleanup_temp_file(path: str | None) -> None:
    if not path:
        return
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink(missing_ok=True)


def _get_audio_stream(streams: Iterable[av.audio.stream.AudioStream]) -> av.audio.stream.AudioStream:
    try:
        return next(iter(streams))
    except StopIteration as exc:
        raise ValueError("В файле не найден аудиопоток.") from exc


def _resolve_duration_seconds(
    container: av.container.input.InputContainer,
    audio_stream: av.audio.stream.AudioStream,
) -> float:
    if container.duration:
        return float(container.duration / av.time_base)

    if audio_stream.duration and audio_stream.time_base:
        return float(audio_stream.duration * audio_stream.time_base)

    raise ValueError("Не удалось определить длительность аудиофайла.")

