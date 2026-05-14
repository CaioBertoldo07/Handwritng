"""
01_extract_glyphs.py -- Rasterize PDF pages and extract individual glyph PNGs.

Usage: python 01_extract_glyphs.py
Output: glyphs/<char>.png  (one file per detected character)
"""

import os
import sys
import argparse
import fitz          # pymupdf
import cv2
import numpy as np

# ── configuration ─────────────────────────────────────────────────────────────
PDF_PATH = "AnnesHandwriting.pdf"
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
#   Page 1  — uppercase: letters on gray paper; adaptive threshold isolates
#             ink without bleeding the gray background.  A modest (5×5) kernel
#             bridges tiny gaps within a stroke.
#   Page 2  — lowercase: i and j have disconnected dots.  A taller (5×50)
#             kernel merges dots with their bodies vertically.
#   Page 3  — symbols: colon and semicolon are stacked sub-dots; period is
#             tiny.  A taller (5×60) kernel merges them.
PAGE_SETTINGS = {
    1: dict(min_area=300, row_gap=80,  dilate_size=(5, 5),  min_bbox_height=80, min_bbox_width=60),
    2: dict(min_area=40,  row_gap=10,  dilate_size=(15, 50), min_bbox_height=50, min_bbox_width=50,
            narrow_aspect_min=3.0, adaptive_c=8),
    3: dict(min_area=80, row_gap=100, dilate_size=(7, 20), min_bbox_height=40, min_bbox_width=40,
        adaptive_c=8),
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
        "1","2","3","4","5","6","7","8","9",
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
                  row_gap: int, dilate_size: tuple | None,
                  min_bbox_height: int = 0,
                  min_bbox_width: int = 0,
                  adaptive_c: int = 10,
                  adaptive_blocksize: int = 51,
                  use_clahe: bool = False,
                  narrow_aspect_min: float = float('inf')) -> list:
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

    # Optional CLAHE to boost local contrast for faint handwriting strokes.
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

    # Adaptive threshold handles gray/textured paper backgrounds where a global
    # Otsu threshold would treat the entire paper as ink.
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=adaptive_blocksize,
        C=adaptive_c,
    )

    close_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw_clean = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_k)

    if dilate_size is not None:
        dilate_k  = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_size)
        # MORPH_CLOSE (dilate then erode) fills intra-glyph gaps (e.g. the
        # b stem-bowl junction or i/j dot gap) without permanently enlarging
        # components, so adjacent letters that are 30+ px apart stay separate.
        bw_detect = cv2.morphologyEx(bw_clean, cv2.MORPH_CLOSE, dilate_k)
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

        if h < min_bbox_height:
            continue

        # Reject boxes that span more than 40 % of the image in either
        # dimension — these are page-edge artefacts, not glyphs.
        img_h, img_w = img_bgr.shape[:2]
        if w > img_w * 0.40 or h > img_h * 0.40:
            continue

        # Accept if EITHER:
        #   a) Width >= min_bbox_width  (normal-width glyph), OR
        #   b) Width < min_bbox_width but aspect ratio h/w >= narrow_aspect_min
        #      (narrow tall letters like i or l).  Only enabled per-page via
        #      PAGE_SETTINGS.  Also blocks detached strokes like the crossbar of
        #      t (w≈45, h/w≈2) which would otherwise bridge two rows.
        if w < min_bbox_width and (h / max(w, 1)) < narrow_aspect_min:
            continue

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


def _merge_x_clusters(boxes: list[tuple], max_gap: int = 70) -> list[tuple]:
    """Merge nearby boxes in x-order into a single enclosing box."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])
    clusters: list[list[tuple]] = [[boxes[0]]]

    for b in boxes[1:]:
        lx, ly, lw, lh = clusters[-1][-1]
        gap = b[0] - (lx + lw)
        if gap <= max_gap:
            clusters[-1].append(b)
        else:
            clusters.append([b])

    merged: list[tuple] = []
    for group in clusters:
        xs = [b[0] for b in group]
        ys = [b[1] for b in group]
        xe = [b[0] + b[2] for b in group]
        ye = [b[1] + b[3] for b in group]
        x0, y0 = min(xs), min(ys)
        merged.append((x0, y0, max(xe) - x0, max(ye) - y0))

    return merged


def reorder_page3_boxes(boxes: list[tuple]) -> list[tuple]:
    """
    Stabilise ordering for the digits/punctuation sheet (page 3).

        Expected visual layout:
            Row 1 (left->right): 1 2 3 4 5 6 7 8 9
            Row 2 (left->right): . , ! ? : ; - _ ( )
    """
    row1, row2, row3 = [], [], []
    for b in boxes:
        _, y, _, h = b
        cy = y + h / 2
        if cy < 450:
            row1.append(b)
        elif cy < 1050:
            row2.append(b)
        else:
            row3.append(b)

    row1 = sorted(row1, key=lambda b: b[0])[:9]
    row2 = _merge_x_clusters(row2, max_gap=70)
    row2 = sorted(row2, key=lambda b: b[0])

    # If the quote was written lower (third row), append its first cluster.
    if len(row2) < 10 and row3:
        row3 = sorted(_merge_x_clusters(row3, max_gap=70), key=lambda b: b[0])
        row2.extend(row3[: 10 - len(row2)])

    return row1 + row2[:10]


def _write_zero_from_small_o(out_dir: str):
    """Create 0.png from small_o.png with smooth scaling (no jagged edges)."""
    src_path = os.path.join(out_dir, "small_o.png")
    dst_path = os.path.join(out_dir, "0.png")
    if not os.path.exists(src_path):
        return

    src = cv2.imread(src_path)
    if src is None:
        return

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(ink > 0)
    if len(xs) == 0:
        cv2.imwrite(dst_path, src)
        return

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    glyph = src[y0:y1+1, x0:x1+1]

    # Slight upscale so 0 sits better with other digits.
    h, w = glyph.shape[:2]
    scale = 1.15
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    glyph = cv2.resize(glyph, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((target_h + 20, target_w + 20, 3), 255, dtype=np.uint8)
    oy = (canvas.shape[0] - target_h) // 2
    ox = (canvas.shape[1] - target_w) // 2
    canvas[oy:oy+target_h, ox:ox+target_w] = glyph
    cv2.imwrite(dst_path, canvas)


def _write_quotedbl_from_page3(out_dir: str, page3_img: np.ndarray,
                               page3_boxes: list[tuple], pad: int):
    """Build quotedbl.png by merging the two left-most bottom quote marks."""
    if page3_img is None or not page3_boxes:
        return

    bottom = []
    for b in page3_boxes:
        _, y, _, h = b
        if y + h / 2 >= 1050:
            bottom.append(b)

    if len(bottom) < 2:
        return

    bottom = sorted(bottom, key=lambda b: b[0])
    left_two = bottom[:2]

    xs = [b[0] for b in left_two]
    ys = [b[1] for b in left_two]
    xe = [b[0] + b[2] for b in left_two]
    ye = [b[1] + b[3] for b in left_two]
    merged = (min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys))

    crop = crop_glyph(page3_img, merged, pad)
    cv2.imwrite(os.path.join(out_dir, "quotedbl.png"), crop)


def write_supplemental_glyphs(out_dir: str, page3_img: np.ndarray,
                              page3_boxes: list[tuple], pad: int):
    _write_zero_from_small_o(out_dir)
    _write_quotedbl_from_page3(out_dir, page3_img, page3_boxes, pad)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rasterize PDF pages and extract individual glyph PNGs."
    )
    parser.add_argument("--pdf", default=PDF_PATH,
                        help="Input PDF path (default: Handwriting.pdf)")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="Output directory for glyph PNGs (default: glyphs)")
    parser.add_argument("--dpi", type=int, default=DPI,
                        help="Rasterization DPI (default: 300)")
    parser.add_argument("--pad", type=int, default=PAD,
                        help="Padding (px) around each glyph crop (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.pdf):
        print(f"ERROR: '{args.pdf}' not found. Run from the repo root.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    doc = fitz.open(args.pdf)
    total_pages = len(doc)
    print(f"Opened '{args.pdf}' -- {total_pages} page(s) found.\n")

    total_saved = 0
    page3_img = None
    page3_raw_boxes = []

    for page_num, chars in PAGE_CHARS.items():
        page_idx = page_num - 1
        if page_idx >= total_pages:
            print(f"[Page {page_num}] SKIP -- PDF only has {total_pages} page(s).")
            continue

        cfg = PAGE_SETTINGS[page_num]
        print(f"[Page {page_num}] Rasterizing at {args.dpi} DPI ...")
        img = rasterize_page(doc, page_idx, args.dpi)

        print(f"[Page {page_num}] Detecting glyphs "
              f"(min_area={cfg['min_area']}, row_gap={cfg['row_gap']}, "
              f"dilate_size={cfg['dilate_size']}) ...")
        boxes = detect_glyphs(
            img, cfg["min_area"], cfg["row_gap"], cfg["dilate_size"],
            min_bbox_height=cfg.get("min_bbox_height", 0),
            min_bbox_width=cfg.get("min_bbox_width", 0),
            adaptive_c=cfg.get("adaptive_c", 10),
            adaptive_blocksize=cfg.get("adaptive_blocksize", 51),
            use_clahe=cfg.get("use_clahe", False),
            narrow_aspect_min=cfg.get("narrow_aspect_min", float("inf")),
        )

        if page_num == 3:
            page3_img = img.copy()
            page3_raw_boxes = list(boxes)
            boxes = reorder_page3_boxes(boxes)

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
        debug_path = os.path.join(args.out_dir, f"_debug_page{page_num}.png")
        cv2.imwrite(debug_path, debug)
        print(f"[Page {page_num}] Debug image -> {debug_path}")

        saved = 0
        for char, box in zip(chars, boxes):
            crop = crop_glyph(img, box, args.pad)
            cv2.imwrite(os.path.join(args.out_dir, f"{char}.png"), crop)
            saved += 1

        print(f"[Page {page_num}] Saved {saved} glyph(s).\n")
        total_saved += saved

    doc.close()

    # Auxiliary glyphs not present in PAGE_CHARS order but used in the font.
    write_supplemental_glyphs(args.out_dir, page3_img, page3_raw_boxes, args.pad)
    print(f"Done. {total_saved} glyph(s) written to '{args.out_dir}/'.")


if __name__ == "__main__":
    main()
