import re
from typing import Iterable


_URL_RE = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+")
_DIGIT_RE = re.compile(r"\d+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _DIGIT_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_series(texts: Iterable[str]) -> list[str]:

    return [normalize_text(str(t)) for t in texts]


