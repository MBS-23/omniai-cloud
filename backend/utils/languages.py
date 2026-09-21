"""Single source of truth for language codes.

This table previously existed in three places (``backend/main.py``,
``frontend/ui_config.py`` and implicitly in ``translation_ai.py``), which let
them drift apart -- the frontend offered Norwegian, Danish, Finnish, Hebrew and
Greek that the translation layer had no NLLB mapping for. Everything now reads
from here, so a language is offered in the UI only if it can actually be
translated.
"""

from __future__ import annotations

# langdetect code -> (display name, NLLB-200 code)
# A language appears in the UI only when it has an NLLB code.
LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("English", "eng_Latn"),
    "hi": ("Hindi", "hin_Deva"),
    "te": ("Telugu", "tel_Telu"),
    "ta": ("Tamil", "tam_Taml"),
    "ml": ("Malayalam", "mal_Mlym"),
    "kn": ("Kannada", "kan_Knda"),
    "mr": ("Marathi", "mar_Deva"),
    "gu": ("Gujarati", "guj_Gujr"),
    "bn": ("Bengali", "ben_Beng"),
    "pa": ("Punjabi", "pan_Guru"),
    "ur": ("Urdu", "urd_Arab"),
    "ne": ("Nepali", "npi_Deva"),
    "si": ("Sinhala", "sin_Sinh"),
    "es": ("Spanish", "spa_Latn"),
    "fr": ("French", "fra_Latn"),
    "de": ("German", "deu_Latn"),
    "it": ("Italian", "ita_Latn"),
    "pt": ("Portuguese", "por_Latn"),
    "ru": ("Russian", "rus_Cyrl"),
    "uk": ("Ukrainian", "ukr_Cyrl"),
    "pl": ("Polish", "pol_Latn"),
    "nl": ("Dutch", "nld_Latn"),
    "sv": ("Swedish", "swe_Latn"),
    "da": ("Danish", "dan_Latn"),
    "no": ("Norwegian", "nob_Latn"),
    "fi": ("Finnish", "fin_Latn"),
    "el": ("Greek", "ell_Grek"),
    "he": ("Hebrew", "heb_Hebr"),
    "tr": ("Turkish", "tur_Latn"),
    "ar": ("Arabic", "arb_Arab"),
    "fa": ("Persian", "pes_Arab"),
    "sw": ("Swahili", "swh_Latn"),
    "ja": ("Japanese", "jpn_Jpan"),
    "ko": ("Korean", "kor_Hang"),
    "zh-cn": ("Chinese (Simplified)", "zho_Hans"),
    "zh-tw": ("Chinese (Traditional)", "zho_Hant"),
    "th": ("Thai", "tha_Thai"),
    "vi": ("Vietnamese", "vie_Latn"),
    "id": ("Indonesian", "ind_Latn"),
    "ms": ("Malay", "zsm_Latn"),
    "tl": ("Tagalog", "tgl_Latn"),
}

UNKNOWN = "unknown"

# Codes that may appear as a detection result but are not translation targets.
_NON_TARGET = {UNKNOWN, "", "auto"}


def language_name(code: str | None) -> str:
    """Human-readable name for a language code, falling back to the code."""
    if not code:
        return "Unknown"
    normalised = code.strip().lower()
    if normalised in _NON_TARGET:
        return "Unknown"
    entry = LANGUAGES.get(normalised)
    return entry[0] if entry else code


def nllb_code(code: str | None, default: str = "eng_Latn") -> str:
    """NLLB-200 token for a langdetect code."""
    if not code:
        return default
    entry = LANGUAGES.get(code.strip().lower())
    return entry[1] if entry else default


def is_supported(code: str | None) -> bool:
    """True when this language can be used as a translation source or target."""
    if not code:
        return False
    return code.strip().lower() in LANGUAGES


def display_names() -> dict[str, str]:
    """Code -> display name, for templates and the client config blob."""
    names = {code: value[0] for code, value in LANGUAGES.items()}
    names[UNKNOWN] = "Unknown"
    return names


def translation_codes() -> list[str]:
    """Selectable translation languages, English first then alphabetical."""
    rest = sorted(
        (code for code in LANGUAGES if code != "en"),
        key=lambda code: LANGUAGES[code][0],
    )
    return ["en", *rest]
