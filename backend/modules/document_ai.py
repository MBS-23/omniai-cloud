"""Document intelligence pipeline.

Handles text-based PDFs, scanned PDFs and Word documents. Which path a file
takes is decided by whether it carries a usable text layer, not by what the
user claims it is.

OCR and summarisation are delegated to the active backend, so the same code
serves both the local PaddleOCR/Mistral path and the fully hosted one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.providers.base import Backend
from backend.utils.docx_utils import extract_text_from_docx, validate_docx_file
from backend.utils.logger import get_logger
from backend.utils.pdf_utils import (
    cleanup_render_dir,
    normalize_utf8_text,
    process_pdf,
    validate_pdf_file,
)

logger = get_logger("OmniAI.document")

# Below this, a "summary" would be longer than the source.
MIN_SUMMARY_CHARS = 50

_DOCUMENT_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("resume", "curriculum vitae"), "This document appears to be a resume or CV."),
    (("invoice", "total amount", "amount due"), "This document appears to be an invoice or bill."),
    (("semester", "sgpa", "cgpa", "marksheet"), "This document appears to be an academic record."),
    (("agreement", "hereby agree", "terms and conditions"), "This document appears to be a contract or agreement."),
    (("abstract", "references", "et al."), "This document appears to be an academic paper."),
)


def _derive_insights(text: str) -> list[str]:
    """Cheap keyword heuristics that give the reader a quick orientation."""
    lowered = text.lower()
    insights = [
        message for keywords, message in _DOCUMENT_HINTS
        if any(keyword in lowered for keyword in keywords)
    ]
    return insights or ["This document contains structured informational content."]


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect

        return detect(text)
    except Exception:
        return "unknown"


def process_document(document_path: str, backend: Backend) -> dict[str, Any]:
    """Extract, summarise and characterise a PDF or DOCX."""
    extension = Path(document_path).suffix.lower()
    render_dir = None
    # Set only when OCR actually produced text. The caller used to infer this
    # by searching for "OCR" in extraction_method, which also matched the
    # message saying OCR was unavailable.
    ocr_used = False

    try:
        if extension == ".pdf":
            metadata = validate_pdf_file(document_path)
            page_count = metadata.get("page_count")
            source_type, extracted_text, image_paths, render_dir = process_pdf(document_path)

            if source_type == "scanned_pdf" and image_paths:
                blocks = []
                for image_path in image_paths:
                    ocr_result = backend.extract_text(str(image_path))
                    if ocr_result.get("has_text"):
                        blocks.append(normalize_utf8_text(ocr_result["text"]))
                extracted_text = "\n\n".join(blocks)
                document_type = "Scanned or image-based PDF"
                extraction_method = f"OCR via {backend.name}"
                ocr_used = bool(blocks)
            elif source_type == "scanned_pdf":
                document_type = "Scanned or image-based PDF"
                extraction_method = (
                    "Could not be read: this is a scanned PDF, and Poppler, which "
                    "renders its pages so they can be read, is not installed"
                )
            else:
                document_type = "Text-based PDF"
                extraction_method = "Embedded text layer (no OCR)"

        elif extension == ".docx":
            validate_docx_file(document_path)
            extracted_text = extract_text_from_docx(document_path)
            source_type = "docx"
            page_count = None
            document_type = "Word document (.docx)"
            extraction_method = "Embedded DOCX text"

        else:
            raise ValueError("Unsupported document type. Only PDF and DOCX are allowed")

        extracted_text = normalize_utf8_text(extracted_text or "")
        normalised = extracted_text.strip()

        if not normalised:
            return {
                "document_type": document_type,
                "extraction_method": extraction_method,
                "text_length": 0,
                "page_count": page_count,
                "summary": "",
                "summary_generated": False,
                "no_text_reason": (
                    "No readable text was found in this document."
                    if ocr_used or source_type != "scanned_pdf"
                    else extraction_method
                ),
                "insights": [],
                "language": "unknown",
                "extracted_text": "",
                "ocr_used": ocr_used,
            }

        summary_generated = False
        if len(normalised) < MIN_SUMMARY_CHARS:
            summary = "The document is very short; the extracted text is shown in full above."
        else:
            summary = (backend.summarize(extracted_text, max_length=200) or "").strip()
            summary_generated = bool(summary)

        if not summary:
            summary = "A summary could not be generated, but the extracted text is available above."

        result: dict[str, Any] = {
            "document_type": document_type,
            "extraction_method": extraction_method,
            "text_length": len(extracted_text),
            "page_count": page_count,
            "summary": summary,
            "summary_generated": summary_generated,
            "insights": _derive_insights(extracted_text),
            "language": _detect_language(extracted_text),
            "ocr_used": ocr_used,
        }

        # Raw text is only returned where the reader may need to verify it:
        # OCR output and DOCX. For text PDFs the summary is the deliverable.
        if source_type in {"scanned_pdf", "mixed_pdf", "docx"}:
            result["extracted_text"] = extracted_text

        return result

    finally:
        cleanup_render_dir(render_dir)
