
"""
test_pdf_function.py — Document → text pipeline for the Ollama bill extractor.

This module is the "input stage" of the pipeline. Its job is to turn a PDF or
image file into clean, readable text. Field extraction (account number,
amount due, etc.) is now handled downstream by Ollama in app1.py.

What's in this version:
  - Page classification, OCR, text assembly
  - Multi-DPI OCR with confidence-based selection
  - Smart preprocessing that adapts to lighting variance
  - Image upscaling before OCR for fine print
  - PDF-native text preferred even on image-heavy PDFs (when usable)
  - OCR error correction for digit-like letters in numeric fields
  - Targeted B→8 fix for Con Ed meter numbers
  - PSM 3 / PSM 6 fallback for layouts that drop content
  - NEW: Skew correction (deskew + auto-rotate for sideways scans)
  - NEW: Column-aware text assembly (opt-in via process_pdf flag)
  - NEW: Region-based OCR fallback for low-confidence pages

Public API:
  process_pdf(pdf_path) -> dict with:
    {
      "source_file": str,
      "pages": [{ "page_number", "classification", "full_text" }, ...],
      "pipeline_summary": { ... }
    }

The shape is backward-compatible with the old caller pattern
`result['pages'][-1]['full_text']`, but now also supports joining all pages.
"""

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import fitz
import numpy as np
import pdfplumber
import pytesseract
from pytesseract import Output


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Region:
    """A single detected content region with its bounding box in PDF points."""
    region_id:   int
    bbox:        tuple   # (x0, y0, x1, y1) in PDF points
    region_type: str     # "text_block" | "image_block" | "ocr_block"
    raw_text:    str   = ""
    confidence:  float = 0.0


@dataclass
class PageResult:
    """Extraction result for one PDF page."""
    page_number:    int
    classification: str   # "text_pdf" | "scanned_image"
    full_text:      str = ""
    ocr_dpi:        Optional[int]   = None  # which DPI won (scanned pages only)
    ocr_confidence: Optional[float] = None  # mean confidence of winning OCR
    used_region_ocr: bool = False           # True if region OCR was triggered


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Page Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_page(page: fitz.Page) -> str:
    """
    Classify whether to use native text extraction or OCR.

    Approach: measure how much of the page area is covered by text blocks
    vs image blocks. Native PDFs have text blocks covering most of the
    visible content; scanned PDFs have one big image block.

    Why this beats keyword matching: a bill using unusual wording
    ("Please remit", "Statement balance") gets correctly classified
    as native text. The old keyword check would send it to OCR
    unnecessarily.

    Returns "text_pdf" or "scanned_image".
    """
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return "scanned_image"

    blocks           = page.get_text("blocks")
    text_block_area  = 0
    image_block_area = 0
    text_parts       = []

    for x0, y0, x1, y1, content, _, btype in blocks:
        area = (x1 - x0) * (y1 - y0)
        if btype == 0 and content.strip():
            # Real text block with non-empty content
            text_block_area += area
            text_parts.append(content)
        elif btype == 1:
            # Image block — bounding box but no readable text
            image_block_area += area

    text_coverage  = text_block_area  / page_area
    image_coverage = image_block_area / page_area
    full_text      = " ".join(text_parts)

    # Sanity check — filters out garbled OCR-added text layers
    # where fitz finds blocks but the content is unusable
    word_count = len(full_text.split())
    looks_legitimate = (
        word_count >= 30
        and len(full_text) / max(word_count, 1) >= 3
    )

    # Decision tree
    if text_coverage > 0.30 and looks_legitimate:
        return "text_pdf"
    if image_coverage > 0.50:
        return "scanned_image"
    if text_coverage > 0.10 and looks_legitimate:
        return "text_pdf"
    return "scanned_image"


# ─────────────────────────────────────────────────────────────────────────────
# Step 2a — Text page region detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_text_regions(page: fitz.Page) -> list[Region]:
    """Extract block-level text regions from a native text PDF page."""
    blocks = page.get_text("blocks")
    regions = []
    for i, (x0, y0, x1, y1, content, _, btype) in enumerate(blocks):
        rtype = "text_block" if btype == 0 else "image_block"
        regions.append(Region(
            region_id=i,
            bbox=(round(x0), round(y0), round(x1), round(y1)),
            region_type=rtype,
            raw_text=content.strip() if btype == 0 else "[embedded image]",
            confidence=1.0,
        ))
    return regions



# ─────────────────────────────────────────────────────────────────────────────
# Step 2b — Skew correction (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def rotate_to_upright(img: np.ndarray) -> np.ndarray:
    """
    Detect 90/180/270-degree rotation (sideways or upside-down scans) and
    rotate the image to upright orientation.

    Uses Tesseract's OSD (Orientation and Script Detection) which is fast
    and accurate for major rotations. Only fires when an actual rotation
    is detected — no-ops on already-upright images.

    OSD can fail on very small or text-sparse images; in that case we
    leave the image alone rather than risk a wrong rotation.
    """
    try:
        osd = pytesseract.image_to_osd(img, config="--psm 0")
        match = re.search(r"Rotate: (\d+)", osd)
        if not match:
            return img
        rotation = int(match.group(1))
    except (pytesseract.TesseractError, AttributeError, ValueError):
        return img

    if rotation == 0:
        return img

    rotation_codes = {
        90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }
    code = rotation_codes.get(rotation)
    return cv2.rotate(img, code) if code is not None else img


def deskew(img: np.ndarray) -> np.ndarray:
    """
    Detect and correct small rotational skew in a document image.

    How it works:
      1. Find dark pixels (text) in the image.
      2. Compute the minimum-area rotated rectangle around them.
      3. Extract the rotation angle of that rectangle.
      4. Rotate the image by the negative of that angle to straighten it.

    Skip conditions:
      - Less than 100 dark pixels: image is mostly blank.
      - Angle below 0.5°: probably noise, not real skew.
      - Angle above 45°: image is rotated 90/180/270 (handled by
        rotate_to_upright, not this function).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 100:
        return img

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angles in [-90, 0]; normalize to [-45, 45]
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5 or abs(angle) > 45:
        return img

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

def looks_column_mashed(text: str) -> bool:
    """
    Heuristic to detect when pdfplumber has interleaved two columns
    into a single linear stream of text.

    Signs of column mashup:
      - Lines that contain BOTH a label-like phrase AND a dollar amount
        from a totally different context, e.g. "782 Distribution Demand
        Charge $12.44" — that's a usage value mashed into a charges row.
      - "Total" labels appearing AFTER their values rather than before
      - Header-row words split across multiple lines

    Returns True if the text shows clear column-mashup symptoms.
    Conservative — only fires when we're confident, so simple bills
    keep using pdfplumber as the fast path.
    """
    if not text:
        return False

    # Symptom 1: a label like "Total ... Usage" appears, but the previous
    # line is just a bare number (the usage value got separated from its label)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^\s*Total\s+\w+\s*Usage\s*$", line, re.IGNORECASE):
            if i > 0 and re.match(r"^\s*\d{2,6}\s*$", lines[i - 1]):
                return True

    # Symptom 2: bare unit ("kWh", "CCF", "therms") on its own line
    # immediately after an unrelated dollar amount line — classic sign
    # of a column being interleaved with a charges column
    unit_alone = re.compile(r"^\s*(kWh|CCF|CF|therms?|gallons?)\s*$", re.IGNORECASE)
    for i, line in enumerate(lines):
        if unit_alone.match(line) and i > 0:
            prev = lines[i - 1].strip()
            if re.search(r"\$\d", prev):
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 2c — Smart preprocessing for OCR
# ─────────────────────────────────────────────────────────────────────────────

def smart_preprocess(img_bgr: np.ndarray, do_deskew: bool = True) -> np.ndarray:
    """
    Adaptive preprocessing that picks between plain grayscale denoise and
    adaptive threshold based on lighting variance.

    Optionally runs orientation correction and skew correction first
    (recommended for any scanned input; harmless on clean PDF renders).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if do_deskew:
        # First fix major rotations (sideways scans), then fine-tune skew
        gray = rotate_to_upright(gray)
        gray = deskew(gray)

    if gray.std() > 65:
        # Uneven lighting — adaptive threshold helps
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=15, C=10,
        )

    # Clean printed bill — gentle denoise preserves thin strokes
    return cv2.fastNlMeansDenoising(gray, h=10)


def upscale_for_small_print(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Upscale before OCR. Tesseract was trained on ~300+ DPI text."""
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2d — OCR with multi-DPI selection
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_at_dpi(page: fitz.Page, dpi: int) -> tuple[list[Region], np.ndarray, np.ndarray, float]:
    """Render the page at a given DPI and run Tesseract.

    Returns 4 values now:
      regions, clean_img (preprocessed), raw_gray (NOT preprocessed), score

    raw_gray is what dual-pass OCR needs — adaptive threshold destroys
    white-on-dark text, so the dual-pass fix only works on raw grayscale.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bgr = cv2.imdecode(
        np.frombuffer(pix.tobytes("png"), np.uint8),
        cv2.IMREAD_COLOR,
    )

    # Build the raw grayscale separately — used by dual-pass OCR later.
    # We still apply rotation/skew correction since those are geometric
    # fixes that don't destroy contrast information.
    raw_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    raw_gray = rotate_to_upright(raw_gray)
    raw_gray = deskew(raw_gray)

    clean_img = smart_preprocess(img_bgr)
    upscaled  = upscale_for_small_print(clean_img, factor=1.5)

    data = pytesseract.image_to_data(
        upscaled, config="--psm 6 --oem 3", output_type=Output.DICT,
    )

    scale = (72 / dpi) / 1.5
    regions: list[Region] = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not word or conf < 30:
            continue
        x, y, w, h = (
            data["left"][i], data["top"][i],
            data["width"][i], data["height"][i],
        )
        regions.append(Region(
            region_id=len(regions),
            bbox=(round(x * scale), round(y * scale),
                  round((x + w) * scale), round((y + h) * scale)),
            region_type="ocr_block",
            raw_text=word,
            confidence=conf / 100.0,
        ))

    if regions:
        total_chars = sum(len(r.raw_text) for r in regions)
        score = sum(r.confidence * len(r.raw_text) for r in regions) / total_chars
    else:
        score = 0.0

    return regions, clean_img, raw_gray, score


def detect_ocr_regions_multi_dpi(
    page: fitz.Page,
    dpis: tuple = (300, 200, 400),
    early_exit_score: float = 0.85,
) -> tuple[list[Region], np.ndarray, np.ndarray, int, float]:
    """Multi-DPI OCR; keeps the highest-confidence result.

    Returns: regions, clean_img, raw_gray, winning_dpi, score
    """
    best = (None, None, None, dpis[0], -1.0)

    for dpi in dpis:
        regions, clean_img, raw_gray, score = _ocr_at_dpi(page, dpi)
        if score > best[4]:
            best = (regions, clean_img, raw_gray, dpi, score)
        if score >= early_exit_score:
            break

    return best  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Text assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_full_text(regions: list[Region], y_tolerance: int = 8) -> str:
    """
    Group word-level Region objects that share approximately the same
    Y coordinate into lines, then sort words left-to-right within each
    line. Reconstructs visual line layout from word-level detections.
    """
    if not regions:
        return ""

    clean = [r for r in regions if r.confidence >= 0.25]
    sorted_r = sorted(clean, key=lambda r: r.bbox[1])
    used: set[int] = set()
    lines: list[str] = []

    for i, region in enumerate(sorted_r):
        if i in used:
            continue
        line_y = region.bbox[1]
        line_words = [region]
        used.add(i)

        for j, other in enumerate(sorted_r):
            if j not in used and abs(other.bbox[1] - line_y) <= y_tolerance:
                line_words.append(other)
                used.add(j)

        line_words.sort(key=lambda r: r.bbox[0])
        line_text = " ".join(r.raw_text for r in line_words)
        line_text = re.sub(r"^[^A-Za-z0-9$]+", "", line_text).strip()
        if line_text:
            lines.append(line_text)

    return "\n".join(lines)


def assemble_text_with_columns(
    regions: list[Region],
    y_tolerance: int = 8,
    column_gap_threshold: float = 0.10,
) -> str:
    """
    Column-aware text assembly. Detects vertical gaps in word positions
    and reads column-by-column instead of strict left-to-right per row.

    How column detection works:
      1. Project every word's horizontal range onto the X axis.
      2. Find horizontal positions where NO words appear (vertical gaps).
      3. If a gap is wider than `column_gap_threshold` * page_width,
         treat it as a column boundary.
      4. Read each column independently with assemble_full_text, then
         concatenate with a separator.

    When to use:
      Bills with strong column structure (header rows with 2-3 columns,
      summary panels alongside main content). For single-column bills
      this falls back to behaving exactly like assemble_full_text.

    column_gap_threshold: fraction of page width that constitutes a
    significant gap. 0.10 = 10% of page width. Higher values = stricter.
    """
    if not regions:
        return ""

    clean = [r for r in regions if r.confidence >= 0.25]
    if not clean:
        return ""

    page_width = max(r.bbox[2] for r in clean)
    min_gap = page_width * column_gap_threshold

    # Build a simple occupancy array marking where any word exists
    occupancy = [False] * (int(page_width) + 1)
    for r in clean:
        for x in range(r.bbox[0], min(r.bbox[2] + 1, len(occupancy))):
            occupancy[x] = True

    # Find runs of False (empty horizontal bands) wider than the threshold
    column_boundaries = [0]
    in_gap = False
    gap_start = 0
    for x, occupied in enumerate(occupancy):
        if not occupied and not in_gap:
            in_gap = True
            gap_start = x
        elif occupied and in_gap:
            in_gap = False
            gap_width = x - gap_start
            if gap_width >= min_gap:
                column_boundaries.append((gap_start + x) // 2)
    column_boundaries.append(int(page_width) + 1)

    # If only one column was detected, regular assembly is fine
    if len(column_boundaries) <= 2:
        return assemble_full_text(regions, y_tolerance)

    # Bucket words into columns by their center X
    columns: list[list[Region]] = [[] for _ in range(len(column_boundaries) - 1)]
    for r in clean:
        center_x = (r.bbox[0] + r.bbox[2]) // 2
        for i in range(len(column_boundaries) - 1):
            if column_boundaries[i] <= center_x < column_boundaries[i + 1]:
                columns[i].append(r)
                break

    column_texts = [assemble_full_text(col, y_tolerance) for col in columns if col]
    return "\n\n".join(column_texts)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b — Region-based OCR (NEW, fallback for low-confidence pages)
# ─────────────────────────────────────────────────────────────────────────────

def detect_text_regions_opencv(img: np.ndarray, min_area: int = 200) -> list[tuple]:
    """
    Detect rectangular text-containing regions using OpenCV contours.
    Returns a list of (x, y, w, h) bounding boxes sorted top-to-bottom,
    left-to-right.

    Pipeline:
      1. Threshold to binary (text becomes white on black).
      2. Dilate horizontally to merge characters into words/lines.
      3. Find contours of the resulting blobs — each is a text region.
      4. Filter tiny noise blobs and sort into reading order.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Wider kernel = bigger merged regions (whole lines vs. isolated words)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))

    # Sort top-to-bottom (bucketed into ~20px row bands), then left-to-right
    boxes.sort(key=lambda b: (b[1] // 20, b[0]))
    return boxes


def ocr_by_regions(img: np.ndarray) -> str:
    """
    Box-based OCR: detect text regions, OCR each independently, reassemble.

    Significantly slower than whole-page OCR (3-5×) because every region
    is its own Tesseract call. But each region gets PSM 7 (single line)
    which is more accurate per-region than PSM 3/6 on the whole page.

    Use this when whole-page OCR is dropping content or character-confusing
    on a layout-heavy bill — typically as a fallback for low-confidence pages.
    """
    boxes = detect_text_regions_opencv(img)
    lines = []

    for (x, y, w, h) in boxes:
        # A few pixels of padding so edge characters aren't clipped
        pad = 3
        crop = img[
            max(0, y - pad):y + h + pad,
            max(0, x - pad):x + w + pad,
        ]
        if crop.size == 0:
            continue

        # Upscale aggressively — small regions benefit most
        upscaled = cv2.resize(
            crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
        )

        # PSM 7 = single line of text
        text = pytesseract.image_to_string(
            upscaled, config="--psm 7 --oem 3"
        ).strip()

        if text:
            lines.append(text)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — OCR error correction for numeric fields
# ─────────────────────────────────────────────────────────────────────────────

_DIGIT_LIKE = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
})

_NUMERIC_FIELD_PATTERNS = [
    re.compile(
        r"(account\s*(?:number|no\.?|#)\s*[:\-]?\s*)([A-Za-z0-9\-]{4,20})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(meter\s*(?:number|no\.?|#)\s*[:\-]?\s*)([A-Za-z0-9\-]{4,20})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(amount\s+due\s*[:\-]?\s*\$?\s*)([A-Za-z0-9,\.]{2,15})",
        re.IGNORECASE,
    ),
]


def correct_ocr_in_numeric_fields(text: str) -> str:
    """Targeted OCR correction in tokens after numeric-field labels."""
    def fix_token(match: re.Match) -> str:
        prefix, token = match.group(1), match.group(2)
        if not any(c.isdigit() for c in token):
            return match.group(0)
        fixed = token.translate(_DIGIT_LIKE)
        return prefix + fixed

    for pattern in _NUMERIC_FIELD_PATTERNS:
        text = pattern.sub(fix_token, text)
    return text


def ocr_with_fallback(img: np.ndarray) -> str:
    """Run PSM 3, but if PSM 6 produces 30%+ more text, prefer PSM 6."""
    psm3_text = pytesseract.image_to_string(img, config="--psm 3 --oem 3")
    psm6_text = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
    if len(psm6_text) > len(psm3_text) * 1.3:
        return psm6_text
    return psm3_text


def fix_meter_number_b_confusion(text: str) -> str:
    """
    Tesseract often reads bold 'B' as '8' in meter numbers like 'B0719203'.
    Targeted fix: when "Meter #:" is followed by a token starting with 8
    plus exactly 7 more digits, convert the leading 8 back to B.
    """
    return re.sub(
        r"(meter\s*#?:?\s*)8(\d{7})\b",
        r"\1B\2",
        text,
        flags=re.IGNORECASE,
    )
def _explode_blocks_to_words(regions: list[Region]) -> list[Region]:
    """
    Convert block-level Region objects (from fitz.get_text("blocks"))
    into word-level Region objects so column detection works correctly.

    Block-level regions are too coarse for column detection — a single
    block can span both columns, defeating the gap-detection logic.
    Word-level regions give us the granularity we need.

    Approximates word bounding boxes by distributing the block's bbox
    proportionally across its words. Not pixel-perfect but accurate
    enough for column boundary detection.
    """
    word_regions = []
    rid = 0
    for r in regions:
        if r.region_type != "text_block" or not r.raw_text:
            continue

        x0, y0, x1, y1 = r.bbox
        block_width = x1 - x0
        if block_width <= 0:
            continue

        # Split block into lines, then each line into words
        for line in r.raw_text.split("\n"):
            words = line.split()
            if not words:
                continue

            total_chars = sum(len(w) for w in words) + len(words) - 1
            if total_chars == 0:
                continue

            # Distribute the block width proportionally to char counts
            cursor = x0
            for w in words:
                proportion = (len(w) + 1) / total_chars
                word_w     = block_width * proportion
                word_regions.append(Region(
                    region_id=rid,
                    bbox=(round(cursor), y0, round(cursor + word_w), y1),
                    region_type="text_block",
                    raw_text=w,
                    confidence=1.0,
                ))
                cursor += word_w
                rid += 1

    return word_regions

# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

# Threshold below which we trigger region-based OCR as a fallback.
# Length-weighted mean confidence — 0.70 means "average confidence per
# character is 70%". Tuned conservatively; raise it (e.g., 0.75) to
# trigger region OCR more often, lower it (e.g., 0.60) to trigger less.
LOW_CONFIDENCE_THRESHOLD = 0.70


def process_pdf(
    pdf_path: str,
    verbose: bool = True,
    use_columns: bool = True,
    region_ocr_fallback: bool = False,
) -> dict:
    """
    Run the document → text pipeline on a PDF or image file.

    Routing:
      1. Image files (.png, .jpg, .jpeg) → always OCR. Image files have
         no PDF structure so there's no text layer to inspect.
      2. PDFs with native text → fast path via pdfplumber. Skips OCR
         entirely. ~50ms per page vs ~10s for OCR.
      3. PDFs that are scanned images → full OCR pipeline.

    Args:
      pdf_path: file to process.
      verbose: print per-page progress.
      use_columns: if True, use column-aware text assembly. Helps on bills
                   with strong 2-3-column structure; can hurt on single-
                   column bills.
      region_ocr_fallback: if True, run region-based OCR when whole-page
                   OCR confidence is below LOW_CONFIDENCE_THRESHOLD.

    Returns the standard dict shape (see module docstring).
    """
    path = Path(pdf_path)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Processing: {path.name}")
        print(f"{'=' * 60}")

    output = {
        "source_file": path.name,
        "pages": [],
        "pipeline_summary": {},
    }

    # ── Short-circuit for image files ─────────────────────────────────────
    # PNG/JPG/JPEG files have no PDF structure, so there's no text layer
    # to read and no reason to run classify_page. Force OCR directly.
    # This avoids fitz.open() pretending an image is a 1-page PDF and
    # then running our coverage check (which would always say 0% text
    # coverage anyway).
    is_image_file = path.suffix.lower() in {".png", ".jpg", ".jpeg"}

    all_text_parts: list[str] = []
    doc = fitz.open(str(path))

    for page_num, page in enumerate(doc, start=1):
        if verbose:
            print(f"\n[Page {page_num}]")

        # Image files always go through OCR; PDFs get classified
        if is_image_file:
            classification = "scanned_image"
            if verbose:
                print(f"    → Image file — routing to OCR")
        else:
            classification = classify_page(page)
            if verbose:
                print(f"    → Classification: {classification.upper()}")

        page_result = PageResult(
            page_number=page_num,
            classification=classification,
        )

        # ── Branch 1: Native text PDF — fast path ─────────────────────────
        if classification == "text_pdf":
            # pdfplumber reads characters directly from the PDF text layer.
            # No image rendering, no OCR, no preprocessing — just read what
            # the PDF file actually contains as text data. ~50ms per page.
            full_text = ""
            try:
                with pdfplumber.open(str(path)) as plumber_pdf:
                    plumber_page = plumber_pdf.pages[page_num - 1]
                    full_text = plumber_page.extract_text() or ""
            except Exception as e:
                if verbose:
                    print(f"    → pdfplumber failed ({e}), trying fitz blocks fallback")

            # Fallback chain — only triggers if pdfplumber returned nothing.
            # First try fitz block-level extraction with our line assembly,
            # then plain fitz.get_text() as a last resort.
            if not full_text.strip():
                regions = detect_text_regions(page)
                if regions:
                    full_text = assemble_full_text(regions)
                if not full_text.strip():
                    full_text = page.get_text("text")

            if verbose:
                print(f"    → {len(full_text)} chars from native text layer "
                      f"(skipped OCR)")

        # ── Branch 2: Scanned image — full OCR pipeline ───────────────────
        else:
            if verbose:
                print(f"    → Running multi-DPI OCR (200/300/400)...")
            regions, clean_img, raw_gray, winning_dpi, score = (
                detect_ocr_regions_multi_dpi(page)
            )

            page_result.ocr_dpi = winning_dpi
            page_result.ocr_confidence = round(score, 3)

            if verbose:
                print(f"    → Best DPI: {winning_dpi}, "
                      f"confidence: {score:.2f}, "
                      f"{len(regions)} word blocks")

            # Dual-pass OCR on raw_gray — recovers white-on-dark text
            # like dark navy header bands. Use raw_gray (not clean_img)
            # because adaptive thresholding destroys white-on-dark text.
            from dual_pass_ocr import ocr_with_dual_pass
            psm3_text = ocr_with_dual_pass(raw_gray, config="--psm 6 --oem 3")

            # Layout-aware assembly from per-word regions
            if use_columns:
                assembled_text = (
                    assemble_text_with_columns(regions) if regions else ""
                )
            else:
                assembled_text = assemble_full_text(regions) if regions else ""

            # Pick the longer of the two text candidates — assembled often
            # catches lines that PSM 6 drops on layout-heavy bills
            if len(assembled_text) > len(psm3_text):
                full_text = assembled_text
            else:
                full_text = psm3_text

            # If we got essentially nothing usable, fall back to assembled
            if len(full_text.strip()) < 50 and regions:
                full_text = assemble_full_text(regions)

            # Region-based OCR fallback for low-confidence pages
            if region_ocr_fallback and score < LOW_CONFIDENCE_THRESHOLD:
                if verbose:
                    print(f"    → Confidence below {LOW_CONFIDENCE_THRESHOLD:.2f}, "
                          f"trying region-based OCR fallback...")
                region_text = ocr_by_regions(clean_img)
                if len(region_text) > len(full_text):
                    full_text = region_text
                    page_result.used_region_ocr = True
                    if verbose:
                        print(f"    → Region OCR adopted ({len(region_text)} chars)")

        # ── Post-processing: targeted OCR corrections ─────────────────────
        # These corrections fix common OCR misreads in numeric fields
        # (account numbers, meter numbers, amounts). They're cheap and
        # safe to run on native text too — pdfplumber output is clean
        # enough that these regexes are no-ops on it.
        full_text = correct_ocr_in_numeric_fields(full_text)
        full_text = fix_meter_number_b_confusion(full_text)

        page_result.full_text = full_text
        all_text_parts.append(full_text)
        output["pages"].append(asdict(page_result))

        if verbose:
            snippet = full_text[:100].replace("\n", " ")
            print(f"    → Text snippet: {snippet!r}...")

    doc.close()

    # Combine all pages with clear separators — recommended input for Ollama
    combined_text = "\n\n--- PAGE BREAK ---\n\n".join(
        f"[Page {i+1}]\n{t}" for i, t in enumerate(all_text_parts)
    )

    output["pipeline_summary"] = {
        "total_pages": len(output["pages"]),
        "classifications": [p["classification"] for p in output["pages"]],
        "all_pages_text": combined_text,
    }

    return output
