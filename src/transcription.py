from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*|\s+|[^А-Яа-яЁё\s]+")

VOWELS = set("аеёиоуыэюя")
SOFTENERS = set("еёиюяь")
ALWAYS_HARD = set("жшц")
ALWAYS_SOFT = set("йчщ")

CONSONANT_MAP = {
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ж": "ʐ",
    "з": "z",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ф": "f",
    "х": "x",
    "ц": "t͡s",
    "ч": "t͡ɕ",
    "ш": "ʂ",
    "щ": "ɕː",
}

VOWEL_MAP = {
    "а": "a",
    "е": "e",
    "ё": "o",
    "и": "i",
    "о": "o",
    "у": "u",
    "ы": "ɨ",
    "э": "e",
    "ю": "u",
    "я": "a",
}

DEVOICE_MAP = {"b": "p", "v": "f", "g": "k", "d": "t", "z": "s", "ʐ": "ʂ"}
VOICE_MAP = {value: key for key, value in DEVOICE_MAP.items()}
OBSTRUENTS = set(DEVOICE_MAP) | set(DEVOICE_MAP.values()) | {"t͡s", "t͡ɕ", "ɕː", "x"}


def normalize_orthography(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])([^\s\n])", r"\1 \2", text)
    return text.strip()


def practical_transcription(text: str) -> str:
    normalized = normalize_orthography(text)
    parts: list[str] = []
    for token in TOKEN_RE.findall(normalized):
        if not token:
            continue
        if token.isspace():
            parts.append(token)
        elif re.search(r"[А-Яа-яЁё]", token):
            if "-" in token:
                parts.append("-".join(_transcribe_word(part) for part in token.split("-")))
            else:
                parts.append(_transcribe_word(token))
        else:
            parts.append(token)
    return "".join(parts)


def _transcribe_word(word: str) -> str:
    source = _prepare_word(word.lower())
    phonemes: list[str] = []
    letters = list(source)

    for index, char in enumerate(letters):
        prev_char = letters[index - 1] if index > 0 else ""
        next_char = letters[index + 1] if index + 1 < len(letters) else ""

        if char in {"ь", "ъ"}:
            continue

        if char in CONSONANT_MAP:
            base = CONSONANT_MAP[char]
            if char in ALWAYS_SOFT:
                phonemes.append(base)
                continue

            soft = char not in ALWAYS_HARD and next_char in SOFTENERS
            phonemes.append(_join_phoneme(base, soft))
            continue

        if char in VOWELS:
            iotated = char in "еёюя" and (index == 0 or prev_char in VOWELS or prev_char in "ьъ")
            if iotated:
                phonemes.append("j")

            if char == "и" and prev_char in ALWAYS_HARD:
                phonemes.append("ɨ")
            else:
                phonemes.append(VOWEL_MAP[char])
            continue

        phonemes.append(char)

    phonemes = _apply_cluster_assimilation(phonemes)
    phonemes = _apply_final_devoicing(phonemes)
    return "".join(phonemes)


def _prepare_word(word: str) -> str:
    transformed = word
    transformed = re.sub(r"ться\b", "ца", transformed)
    transformed = re.sub(r"тся\b", "ца", transformed)
    transformed = re.sub(r"ого\b", "ова", transformed)
    transformed = re.sub(r"его\b", "ева", transformed)
    transformed = transformed.replace("сч", "щ")
    transformed = transformed.replace("зч", "щ")
    transformed = transformed.replace("тч", "ч")
    transformed = transformed.replace("дч", "ч")
    transformed = transformed.replace("стн", "сн")
    transformed = transformed.replace("здн", "зн")
    return transformed


def _apply_cluster_assimilation(phonemes: list[str]) -> list[str]:
    updated = phonemes[:]
    for index in range(len(updated) - 1):
        current_base, current_soft = _split_phoneme(updated[index])
        next_base, _ = _split_phoneme(updated[index + 1])

        if current_base not in OBSTRUENTS or next_base not in OBSTRUENTS:
            continue

        if next_base in DEVOICE_MAP.values() or next_base in {"t͡s", "t͡ɕ", "ɕː", "x"}:
            current_base = DEVOICE_MAP.get(current_base, current_base)
        elif next_base in DEVOICE_MAP:
            current_base = VOICE_MAP.get(current_base, current_base)

        updated[index] = _join_phoneme(current_base, current_soft)
    return updated


def _apply_final_devoicing(phonemes: list[str]) -> list[str]:
    if not phonemes:
        return phonemes

    base, soft = _split_phoneme(phonemes[-1])
    if base in DEVOICE_MAP:
        phonemes[-1] = _join_phoneme(DEVOICE_MAP[base], soft)
    return phonemes


def _split_phoneme(phoneme: str) -> tuple[str, bool]:
    if phoneme.endswith("ʲ"):
        return phoneme[:-1], True
    return phoneme, False


def _join_phoneme(base: str, soft: bool) -> str:
    if not soft or base in ALWAYS_SOFT_PHONEMES:
        return base
    return f"{base}ʲ"


ALWAYS_SOFT_PHONEMES = {"j", "t͡ɕ", "ɕː"}
