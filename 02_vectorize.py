"""
02_vectorize.py -- Convert glyph PNGs to SVG vector outlines.

Uses vtracer (pure-Python Rust extension) instead of the external potrace
binary, so no system-level install is required.

Usage: python 02_vectorize.py
Input:  glyphs/<name>.png
Output: svgs/<name>.svg
"""

import os
import sys
import argparse
import cv2
import numpy as np
import vtracer

GLYPH_DIR = "glyphs"
SVG_DIR   = "svgs"

# vtracer settings tuned for handwriting glyphs
VTRACER_OPTS = dict(
    colormode        = "binary",   # black-and-white tracing
    mode             = "spline",   # smooth cubic Bezier curves
    filter_speckle   = 4,          # suppress specs smaller than 4 px (like potrace turdsize=2)
    corner_threshold = 60,         # smoothness: higher = fewer sharp corners
    length_threshold = 4.0,        # minimum segment length
    path_precision   = 3,          # decimal places in SVG output
)


def preprocess_png(png_path: str) -> str:
    """
    Binarise the glyph PNG with Otsu thresholding and save a clean copy in
    a temp location so vtracer always gets a crisp black-on-white image.
    Returns the path of the pre-processed PNG (caller must delete it).
    """
    img  = cv2.imread(png_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # vtracer expects the ink to be dark on a white background — ensure that.
    ink_pixels = np.count_nonzero(bw == 0)
    total      = bw.size
    if ink_pixels > total // 2:
        bw = cv2.bitwise_not(bw)

    # Apply extra cleanup only to glyphs that historically showed a black
    # background artefact after tracing.
    risky = {"0.png", "8.png", "hyphen.png", "underscore.png"}
    if os.path.basename(png_path) in risky:
        fg = (bw == 0).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        h, w = fg.shape
        border_labels = set(np.unique(np.concatenate([
            labels[0, :], labels[h - 1, :], labels[:, 0], labels[:, w - 1]
        ])))

        cleaned = np.zeros_like(fg)
        for label in range(1, n_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if label in border_labels:
                continue
            if area < 6:
                continue
            cleaned[labels == label] = 1

        # Only replace when cleanup still leaves meaningful ink.
        if int(cleaned.sum()) >= 20:
            bw = np.full_like(bw, 255)
            bw[cleaned == 1] = 0

    tmp_path = png_path.replace(".png", "_clean.png")
    cv2.imwrite(tmp_path, bw)
    return tmp_path


def vectorize_glyph(png_path: str, svg_path: str) -> bool:
    """Vectorise one PNG glyph to SVG. Returns True on success."""
    tmp = None
    try:
        tmp = preprocess_png(png_path)
        vtracer.convert_image_to_svg_py(tmp, svg_path, **VTRACER_OPTS)

        # Sanity check: SVG must contain at least one <path>
        with open(svg_path) as f:
            content = f.read()
        if '<path' not in content:
            print(f"  WARNING: {os.path.basename(svg_path)} — SVG produced no paths.")
            return False

        return True

    except Exception as exc:
        print(f"  ERROR vectorising {os.path.basename(png_path)}: {exc}")
        return False

    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def main():
    parser = argparse.ArgumentParser(
        description="Convert glyph PNGs to SVG vector outlines."
    )
    parser.add_argument("--glyph-dir", default=GLYPH_DIR,
                        help="Input directory with glyph PNGs (default: glyphs)")
    parser.add_argument("--svg-dir", default=SVG_DIR,
                        help="Output directory for SVGs (default: svgs)")
    args = parser.parse_args()

    if not os.path.isdir(args.glyph_dir):
        print(f"ERROR: '{args.glyph_dir}/' not found. Run 01_extract_glyphs.py first.",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.svg_dir, exist_ok=True)

    # Collect non-debug PNGs
    pngs = sorted(
        f for f in os.listdir(args.glyph_dir)
        if f.endswith(".png") and not f.startswith("_")
    )
    if not pngs:
        print(f"No glyph PNGs found in '{args.glyph_dir}/'.")
        sys.exit(1)

    print(f"Vectorising {len(pngs)} glyph(s) from '{args.glyph_dir}/' -> '{args.svg_dir}/' ...\n")

    ok = fail = 0
    for fname in pngs:
        name     = os.path.splitext(fname)[0]
        png_path = os.path.join(args.glyph_dir, fname)
        svg_path = os.path.join(args.svg_dir,   f"{name}.svg")

        success = vectorize_glyph(png_path, svg_path)
        if success:
            print(f"  [OK]   {fname:25s} -> {name}.svg")
            ok += 1
        else:
            fail += 1

    print(f"\nDone. {ok} succeeded, {fail} failed.")
    if fail:
        print("Failed glyphs will be skipped in 03_build_font.py.")


if __name__ == "__main__":
    main()
