import fitz
from pathlib import Path


def load_bill_pages(file_path: str, dpi: int = 150) -> list[bytes]:
    """Converts any PDF, JPG, or PNG into a list of PNG byte strings."""
    doc   = fitz.open(file_path)
    pages = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        pages.append(pix.tobytes("png"))
    doc.close()
    print(f"  Loaded {len(pages)} page(s) from {Path(file_path).name}")
    return pages