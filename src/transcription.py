from __future__ import annotations

import re


WORD_RE = re.compile(r"[А-Яа-яЁё-]+")

FINAL_DEVOICE_MAP = str.maketrans(
    {
        "б": "п",
        "в": "ф",
        "г": "к",
        "д": "т",
        "ж": "ш",
        "з": "с",
    }
)


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
    return WORD_RE.sub(_transcribe_word_match, normalized)


def transcription_to_tts_text(text: str) -> str:
    normalized = text.lower()
    replacements = {
        "што": "что",
        "штобы": "чтобы",
        "ево": "его",
        "ова": "ого",
        "ца": "тся",
    }
    words = []
    for chunk in normalized.split():
        cleaned = re.sub(r"[^а-яё-]", "", chunk)
        words.append(replacements.get(cleaned, cleaned or chunk))
    return normalize_orthography(" ".join(words))


def _transcribe_word_match(match: re.Match[str]) -> str:
    original = match.group(0)
    lowercase = original.lower()
    transcribed = _transcribe_word(lowercase)

    if original.isupper():
        return transcribed.upper()
    if original[0].isupper():
        return transcribed.capitalize()
    return transcribed


def _transcribe_word(word: str) -> str:
    transformed = word
    transformed = transformed.replace("что", "што")
    transformed = transformed.replace("чтобы", "штобы")
    transformed = transformed.replace("чтоб", "штоб")
    transformed = re.sub(r"(е|о)го\b", "ево", transformed)
    transformed = re.sub(r"(е|о)му\b", "ему", transformed)
    transformed = re.sub(r"ться\b", "ца", transformed)
    transformed = re.sub(r"тся\b", "ца", transformed)
    transformed = re.sub(r"дс", "ц", transformed)
    transformed = re.sub(r"тс", "ц", transformed)
    transformed = re.sub(r"сч", "щ", transformed)
    transformed = re.sub(r"зч", "щ", transformed)
    transformed = re.sub(r"сш", "ш", transformed)
    transformed = re.sub(r"зш", "ш", transformed)
    transformed = re.sub(r"жч", "щ", transformed)
    transformed = _apply_final_devoicing(transformed)
    return transformed


def _apply_final_devoicing(word: str) -> str:
    if not word:
        return word
    final = word[-1]
    devoiced = final.translate(FINAL_DEVOICE_MAP)
    if devoiced == final:
        return word
    return word[:-1] + devoiced

