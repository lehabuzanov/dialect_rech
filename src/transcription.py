from __future__ import annotations

import re

from gruut import sentences


IPA_TO_RU_MAP = [
    ("t͡ɕ", "ч"),
    ("ɕː", "щ"),
    ("ʐ", "ж"),
    ("ʂ", "ш"),
    ("t͡s", "ц"),
    ("rʲ", "р’"),
    ("lʲ", "л’"),
    ("nʲ", "н’"),
    ("mʲ", "м’"),
    ("pʲ", "п’"),
    ("bʲ", "б’"),
    ("fʲ", "ф’"),
    ("vʲ", "в’"),
    ("tʲ", "т’"),
    ("dʲ", "д’"),
    ("sʲ", "с’"),
    ("zʲ", "з’"),
    ("kʲ", "к’"),
    ("gʲ", "г’"),
    ("xʲ", "х’"),
    ("j", "й"),
    ("ɨ", "ы"),
    ("eː", "э̄"),
    ("iː", "ӣ"),
    ("uː", "ӯ"),
    ("oː", "о̄"),
    ("aː", "а̄"),
    ("a", "а"),
    ("b", "б"),
    ("d", "д"),
    ("e", "э"),
    ("f", "ф"),
    ("g", "г"),
    ("i", "и"),
    ("k", "к"),
    ("l", "л"),
    ("m", "м"),
    ("n", "н"),
    ("o", "о"),
    ("p", "п"),
    ("r", "р"),
    ("s", "с"),
    ("t", "т"),
    ("u", "у"),
    ("v", "в"),
    ("x", "х"),
    ("z", "з"),
]


def normalize_orthography(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])([^\s\n])", r"\1 \2", text)
    return text.strip()


def get_transcription_modes() -> dict[str, str]:
    return {
        "ru_practical": "Русская практическая",
        "ipa": "Английские / международные символы (IPA)",
    }


def build_transcription(text: str, mode: str) -> str:
    normalized = normalize_orthography(text)
    ipa = _phonemize_russian(normalized)
    if mode == "ipa":
        return ipa
    if mode == "ru_practical":
        return _ipa_to_russian_practical(ipa)
    raise ValueError(f"Неизвестный режим транскрипции: {mode}")


def _phonemize_russian(text: str) -> str:
    chunks: list[str] = []
    for sentence in sentences(text, lang="ru-ru"):
        words: list[str] = []
        for word in sentence.words:
            if word.is_spoken and word.phonemes:
                words.append("".join(word.phonemes))
            else:
                words.append(word.text)
        chunks.append(" ".join(words))
    return " // ".join(chunk.strip() for chunk in chunks if chunk.strip())


def _ipa_to_russian_practical(ipa_text: str) -> str:
    practical = ipa_text
    for source, target in IPA_TO_RU_MAP:
        practical = practical.replace(source, target)
    practical = practical.replace("ˈ", "ˈ")
    practical = practical.replace(" ", " ")
    return practical
