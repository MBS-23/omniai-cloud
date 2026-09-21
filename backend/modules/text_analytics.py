"""Sentiment and named entities (paper section 4B; Appendix A.5).

The thesis pipeline, unchanged in substance:

1. detect the language with langdetect;
2. if it is not English, translate to English with NLLB-200, because both
   downstream models are English models;
3. extract named entities with spaCy ``en_core_web_sm``;
4. score sentiment with ``cardiffnlp/twitter-roberta-base-sentiment-latest``,
   then apply the thesis's calibration for grief-heavy text, which the base
   model tends to call neutral.

Both models are loaded from the Hugging Face Hub, spaCy's included: its
``spacy/en_core_web_sm`` repository holds the full pipeline directory, which
``spacy.load`` reads directly, so no separate model package is installed.
"""

from __future__ import annotations

from typing import Any

from backend.modules.hub import PAPER_MODELS
from backend.modules.lazy import Lazy
from backend.modules.translation_ai import translate_to_english
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.text")


def _build_ner() -> Any:
    import spacy
    from huggingface_hub import snapshot_download

    # The repository also carries the pip wheel of the same pipeline; the
    # directory is all spacy.load needs.
    path = snapshot_download(PAPER_MODELS["ner"].repo_id, ignore_patterns=["*.whl"])
    return spacy.load(path)


def _build_sentiment() -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo = PAPER_MODELS["sentiment"].repo_id
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo)
    return tokenizer, model.eval()


NER = Lazy("ner", "spaCy en_core_web_sm", _build_ner, resident_mb=80)
SENTIMENT = Lazy("sentiment", "RoBERTa sentiment", _build_sentiment, resident_mb=550)


def _normalize_sentiment_label(raw_label: str) -> str:
    label = (raw_label or "").strip().lower()
    if "negative" in label or label in {"label_0", "neg"}:
        return "SAD"
    if "neutral" in label or label in {"label_1", "neu"}:
        return "NEUTRAL"
    if "positive" in label or label in {"label_2", "pos"}:
        return "POSITIVE"
    return "NEUTRAL"


def _sentiment_scores(text: str) -> dict[str, float]:
    """The model's probability for each of SAD, NEUTRAL and POSITIVE."""
    import torch

    tokenizer, model = SENTIMENT.get()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)[0].tolist()

    scores = {"SAD": 0.0, "NEUTRAL": 0.0, "POSITIVE": 0.0}
    for index, probability in enumerate(probabilities):
        label = _normalize_sentiment_label(model.config.id2label[index])
        scores[label] = max(scores[label], float(probability))
    return scores


def _calibrate_sentiment(
    text: str, model_label: str, model_confidence: float, scores: dict[str, float]
) -> tuple[str, float]:
    """The thesis's correction for poetic or grief-heavy text."""
    lowered = (text or "").lower()
    grief_terms = [
        "grief", "loss", "lost", "alone", "lonely", "broken", "hollow",
        "empty", "wound", "vanished", "missing", "gone", "dark", "tears",
        "sorrow", "sad", "hurt", "pain", "ache", "mourning", "shadow",
    ]
    grief_hits = sum(1 for term in grief_terms if term in lowered)

    sad_score = scores.get("SAD", 0.0)
    neutral_score = scores.get("NEUTRAL", 0.0)
    positive_score = scores.get("POSITIVE", 0.0)

    label, confidence = model_label, model_confidence

    if label == "NEUTRAL" and grief_hits >= 2 and sad_score >= 0.20:
        label = "SAD"
        confidence = max(confidence, min(0.90, sad_score + 0.22))

    if label == "POSITIVE" and grief_hits >= 3 and positive_score < 0.75 and sad_score >= 0.22:
        label = "SAD"
        confidence = max(sad_score, min(0.88, sad_score + 0.18))

    if label != "SAD" and sad_score > neutral_score + 0.08 and sad_score > positive_score + 0.08:
        label = "SAD"
        confidence = max(confidence, sad_score)

    return label, confidence


def detect_language(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect

        # langdetect is randomised; seeding makes the same text always report
        # the same language, which the benchmark and the tests rely on.
        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "unknown"


def analyze_text(text: str) -> dict[str, Any]:
    """Language, sentiment and named entities, with the reasoning spelled out."""
    steps: list[str] = []
    language = detect_language(text)
    steps.append(f"Detected input language as '{language}'.")

    processed = text
    translated: str | None = None
    if language not in {"en", "unknown"}:
        result = translate_to_english(text, source_lang=language)
        candidate = (result.get("translated_text") or "").strip()
        if candidate and candidate != text.strip():
            translated = processed = candidate
            steps.append("Translated to English with NLLB-200, because spaCy and RoBERTa are English models.")
        else:
            steps.append("Translation to English did not change the text, so it was analysed as written.")
    else:
        steps.append("Input is English, so no translation was needed.")

    entities = [{"text": ent.text, "label": ent.label_} for ent in NER.get()(processed).ents]
    steps.append(f"Extracted {len(entities)} named entities with spaCy en_core_web_sm.")

    scores = _sentiment_scores(processed)
    model_label = max(scores, key=scores.get)
    label, confidence = _calibrate_sentiment(processed, model_label, scores[model_label], scores)
    steps.append("Scored sentiment with RoBERTa (negative, neutral, positive) and applied the thesis calibration.")

    return {
        "original_language": language,
        "translated_text": translated,
        "sentiment": label,
        "sentiment_confidence": round(confidence * 100, 2),
        "sentiment_scores": {name: round(value * 100, 2) for name, value in scores.items()},
        "entities": entities,
        "explanation": " ".join(steps),
    }
