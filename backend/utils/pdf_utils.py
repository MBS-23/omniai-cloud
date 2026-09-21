"""PDF validation, text extraction and rasterisation.

Three problems in the original version are fixed here:

* The Poppler binary directory was hardcoded to one developer's Downloads
  folder, so scanned-PDF support was broken on every other machine. It is now
  read from ``OMNIAI_POPPLER_PATH``, falling back to ``PATH``, which is
  correct on Linux, macOS and inside the container.
* Rasterised pages were written to a single shared
  ``temp_uploads/pdf_images/page_N.png``. Two concurrent uploads overwrote
  each other, so one request could read another request's pages. Each call now
  gets its own temporary directory.
* Those files were never deleted. :func:`process_pdf` now returns the
  directory so the caller can clean up, and :func:`cleanup_render_dir` does it.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pdf2image import convert_from_path
from pypdf import PdfReader

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.pdf")

MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 200
# Pages rendered for OCR. Beyond this the cost is not worth the marginal text.
MAX_RASTER_PAGES = 30
RASTER_DPI = 200


def _poppler_path() -> str | None:
    """Explicit Poppler directory, or None to let pdf2image use PATH."""
    configured = get_settings().poppler_path
    if configured and Path(configured).is_dir():
        return configured
    if configured:
        logger.warning("OMNIAI_POPPLER_PATH is set but not a directory: %s", configured)
    return None


def poppler_available() -> bool:
    """True when pdftoppm can be located, so scanned PDFs can be processed."""
    if _poppler_path():
        return True
    return shutil.which("pdftoppm") is not None


def normalize_utf8_text(text: str) -> str:
    """Enforce a strict UTF-8 round trip, rejecting lone surrogates."""
    if not isinstance(text, str):
        raise TypeError("Document text must be a string")
    return text.encode("utf-8", "strict").decode("utf-8", "strict")


def validate_pdf_file(pdf_path: str) -> dict[str, int]:
    """Reject anything that is not a safe, readable PDF.

    Checks extension, magic bytes, size, encryption and page count before any
    parsing library walks the structure.
    """
    settings = get_settings()
    max_bytes = settings.limits.max_upload_bytes or MAX_PDF_SIZE_BYTES
    max_pages = settings.limits.max_pdf_pages or MAX_PDF_PAGES

    path = Path(pdf_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only .pdf documents are supported")

    if not path.exists() or not path.is_file():
        raise ValueError("Uploaded PDF file not found")

    file_size = path.stat().st_size
    if file_size <= 0:
        raise ValueError("Uploaded PDF is empty")

    if file_size > max_bytes:
        raise ValueError(
            f"PDF exceeds the maximum allowed size of {max_bytes // (1024 * 1024)} MB"
        )

    with open(path, "rb") as file_obj:
        if file_obj.read(5) != b"%PDF-":
            raise ValueError("Invalid PDF signature. Upload a valid PDF document")

    try:
        reader = PdfReader(str(path), strict=False)
    except Exception as error:
        raise ValueError(f"Invalid or corrupted PDF: {error}") from error

    if reader.is_encrypted:
        raise ValueError("Encrypted or password-protected PDFs are not supported")

    page_count = len(reader.pages)
    if page_count <= 0:
        raise ValueError("PDF has no readable pages")

    if page_count > max_pages:
        raise ValueError(f"PDF exceeds the maximum page limit of {max_pages}")

    return {"file_size": file_size, "page_count": page_count}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Pull the embedded text layer out of a PDF, if it has one."""
    reader = PdfReader(pdf_path)
    pages_text = []

    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception as error:
            logger.warning("Page text extraction failed: %s", error)
            continue
        if text:
            pages_text.append(text)

    return normalize_utf8_text("\n".join(pages_text).strip())


def extract_images_from_pdf(pdf_path: str) -> tuple[list[Path], Path]:
    """Render PDF pages to PNGs in a fresh temporary directory.

    Returns the page paths and the directory holding them. The caller owns
    that directory and should pass it to :func:`cleanup_render_dir`.
    """
    render_dir = Path(tempfile.mkdtemp(prefix="omniai_pdf_"))

    try:
        images = convert_from_path(
            pdf_path,
            dpi=RASTER_DPI,
            poppler_path=_poppler_path(),
            first_page=1,
            last_page=MAX_RASTER_PAGES,
        )
    except Exception as error:
        cleanup_render_dir(render_dir)
        if "poppler" in str(error).lower():
            raise ValueError(
                "Scanned PDFs need Poppler, which was not found. Install it and "
                "either add it to PATH or set OMNIAI_POPPLER_PATH in your .env file."
            ) from error
        raise ValueError(f"Could not render PDF pages: {error}") from error

    image_paths: list[Path] = []
    for index, image in enumerate(images, start=1):
        image_path = render_dir / f"page_{index:03d}.png"
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    logger.info("Rendered %s page(s) for OCR", len(image_paths))
    return image_paths, render_dir


def cleanup_render_dir(render_dir: Path | None) -> None:
    """Remove a rasterisation directory, ignoring failures."""
    if render_dir is None:
        return
    shutil.rmtree(render_dir, ignore_errors=True)


def analyze_pdf_type(text: str) -> str:
    """Decide whether a PDF carries a usable text layer."""
    return "text_pdf" if text and len(text) > 100 else "scanned_pdf"


def process_pdf(pdf_path: str) -> tuple[str, str, list[Path], Path | None]:
    """Unified entry point.

    Returns ``(pdf_type, extracted_text, image_paths, render_dir)``. The
    render directory is ``None`` for text PDFs and must be cleaned up by the
    caller otherwise.
    """
    extracted_text = extract_text_from_pdf(pdf_path)
    pdf_type = analyze_pdf_type(extracted_text)

    if pdf_type != "scanned_pdf":
        return pdf_type, extracted_text, [], None

    if not poppler_available():
        logger.warning("Scanned PDF detected but Poppler is unavailable; skipping OCR")
        return pdf_type, extracted_text, [], None

    image_paths, render_dir = extract_images_from_pdf(pdf_path)
    return pdf_type, extracted_text, image_paths, render_dir
