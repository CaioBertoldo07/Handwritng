"""
01_extract_glyphs.py -- Rasterize PDF pages and extract individual glyph PNGs.

Usage: python 01_extract_glyphs.py
Output: glyphs/<char>.png  (one file per detected character)
"""

import os
import sys
import fitz          # pymupdf
import cv2
import numpy as np

# ── configuration ─────────────────────────────────────────────────────────────
PDF_PATH = "Handwriting.pdf"
OUT_DIR  = "glyphs"
DPI      = 300
PAD      = 10    # px of padding around each cropped glyph

# Per-page settings
#   min_area    : minimum original-ink pixel count to keep a component
#   row_gap     : vertical gap (px) between the bottom of one row and the top
#                 of the next — used to split rows in the reading-order sort
#   dilate_size : (width, height) of dilation kernel applied BEFORE contour
#                 detection to merge sub-components that belong to the same
#                 glyph.  Bounding boxes are always measured on the ORIGINAL
#                 ink image, so row positions are unaffected by dilation size.
#
#   Page 1  — uppercase: H has a 2-px ink gap in its crossbar that splits it
#             into two components.  A wide (25×5) kernel bridges horizontal
#             gaps ≤ 12 px without touching adjacent letters (≥200 px apart).
#   Page 2  — lowercase: i and j have disconnected dots (42 and 67 px²).
#             A tall (5×100) kernel merges dots with their bodies vertically.
#   Page 3  — symbols: colon and semicolon are stacked sub-dots; period is
#             tiny.  A tall (5×120) kernel merges them.
PAGE_SETTINGS = {
    1: dict(min_area=500, row_gap=80,  dilate_size=(25, 5)),
    2: dict(min_area=40,  row_gap=80,  dilate_size=(5, 100)),
    3: dict(min_area=80,  row_gap=100, dilate_size=(5, 120)),
}

# Windows NTFS is case-insensitive: A.png == a.png, so uppercase and lowercase
# glyphs cannot share the bare letter as filename.  We use a prefix scheme:
#   cap_A  ... cap_Z   for uppercase A-Z
#   small_a ... small_z for lowercase a-z
# Digits and symbol names are already case-safe.
PAGE_CHARS = {
    1: [f"cap_{c}"   for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    2: [f"small_{c}" for c in "abcdefghijklmnopqrstuvwxyz"],
    3: [
        "0","1","2","3","4","5","6","7","8","9",
        "period","comma","exclamation","question",
        "colon","semicolon","hyphen","underscore",
        "parenleft","parenright",
    ],
}

# ── helpers ───────────────────────────────────────────────────────────────────

def rasterize_page(doc: fitz.Document, page_index: int, dpi: int) -> np.ndarray:
    """Return a page as a BGR numpy array at the requested DPI."""
    page = doc[page_index]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def detect_glyphs(img_bgr: np.ndarray, min_area: int,
                  row_gap: int, dilate_size: tuple | None) -> list:
    """
    Return tight (x, y, w, h) bounding boxes in reading order.

    Strategy
    --------
    1. Binarise → morphological close to heal tiny ink gaps.
    2. Apply an optional dilation (page-specific kernel) so sub-components
       that belong to the same glyph merge into one connected region.
    3. Label connected components on the detection image.
    4. For each component, map back to ORIGINAL ink pixels and compute the
       tight bounding box there.  This keeps positions accurate regardless
       of dilation size, so row-gap sorting always uses real ink coordinates.
    5. Filter by original ink pixel count; sort into reading order.
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    close_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw_clean = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_k)

    if dilate_size is not None:
        dilate_k  = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_size)
        bw_detect = cv2.dilate(bw_clean, dilate_k)
    else:
        bw_detect = bw_clean

    num_labels, labels = cv2.connectedComponents(bw_detect)

    boxes = []
    for label in range(1, num_labels):
        component_mask = labels == label                          # bool H×W

        # Tight bounding box from ORIGINAL ink inside this merged component
        orig_in_comp = bw_clean & (component_mask.astype(np.uint8) * 255)
        ink_count = int(orig_in_comp.sum()) // 255
        if ink_count < min_area:
            continue

        ys, xs = np.where(orig_in_comp > 0)
        x = int(xs.min());  w = int(xs.max()) - x + 1
        y = int(ys.min());  h = int(ys.max()) - y + 1
        boxes.append((x, y, w, h))

    return _sort_boxes(boxes, row_gap)


def _sort_boxes(boxes: list, row_gap: int) -> list:
    """
    Reading-order sort using gap-based row detection.

    Items are in the same row when there is no gap larger than row_gap between
    the tallest bottom in the current row and the next item's top.  Within
    each row items are sorted left-to-right.
    """
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda b: b[1])   # sort by top-y first

    rows: list[list] = []
    row = [ordered[0]]

    for box in ordered[1:]:
        row_bottom = max(b[1] + b[3] for b in row)
        gap        = box[1] - row_bottom
        if gap > row_gap:
            rows.append(sorted(row, key=lambda b: b[0]))
            row = [box]
        else:
            row.append(box)

    rows.append(sorted(row, key=lambda b: b[0]))
    return [box for row in rows for box in row]


def crop_glyph(img_bgr: np.ndarray, box: tuple, pad: int) -> np.ndarray:
    x, y, w, h = box
    H, W = img_bgr.shape[:2]
    return img_bgr[max(0, y-pad):min(H, y+h+pad),
                   max(0, x-pad):min(W, x+w+pad)]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: '{PDF_PATH}' not found. Run from the repo root.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    print(f"Opened '{PDF_PATH}' -- {total_pages} page(s) found.\n")

    total_saved = 0

    for page_num, chars in PAGE_CHARS.items():
        page_idx = page_num - 1
        if page_idx >= total_pages:
            print(f"[Page {page_num}] SKIP -- PDF only has {total_pages} page(s).")
            continue

        cfg = PAGE_SETTINGS[page_num]
        print(f"[Page {page_num}] Rasterizing at {DPI} DPI ...")
        img = rasterize_page(doc, page_idx, DPI)

        print(f"[Page {page_num}] Detecting glyphs "
              f"(min_area={cfg['min_area']}, row_gap={cfg['row_gap']}, "
              f"dilate_size={cfg['dilate_size']}) ...")
        boxes = detect_glyphs(
            img, cfg["min_area"], cfg["row_gap"], cfg["dilate_size"]
        )

        expected = len(chars)
        found    = len(boxes)
        print(f"[Page {page_num}] Expected {expected}, detected {found}.")

        if found != expected:
            print(f"[Page {page_num}] WARNING: count mismatch -- "
                  f"mapping first {min(found, expected)} box(es) to characters.")

        # Debug: draw tight original-ink bounding boxes on the source image
        debug = img.copy()
        for bx, by, bw_, bh in boxes:
            cv2.rectangle(debug, (bx, by), (bx+bw_, by+bh), (0, 0, 255), 3)
        debug_path = os.path.join(OUT_DIR, f"_debug_page{page_num}.png")
        cv2.imwrite(debug_path, debug)
        print(f"[Page {page_num}] Debug image -> {debug_path}")

        saved = 0
        for char, box in zip(chars, boxes):
            crop = crop_glyph(img, box, PAD)
            cv2.imwrite(os.path.join(OUT_DIR, f"{char}.png"), crop)
            saved += 1

        print(f"[Page {page_num}] Saved {saved} glyph(s).\n")
        total_saved += saved

    doc.close()
    print(f"Done. {total_saved} glyph(s) written to '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()
