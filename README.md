# Handwriting → TTF Pipeline

Converts a hand-filled PDF template into a usable TrueType font in three automated steps.

---

## Dependencies

### Python packages

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `pymupdf` | Rasterize PDF pages at 300 DPI |
| `opencv-python` | Glyph detection via contour analysis |
| `numpy` | Array operations for image processing |
| `vtracer` | Bitmap-to-vector tracing (pure Python, no system binary needed) |
| `fonttools` | Assemble the TTF font file |

### No system binaries required

This pipeline uses **vtracer** (a Rust-based Python extension) instead of the `potrace` system binary. Everything installs via pip — no Homebrew, apt, or Chocolatey needed.

---

## PDF template layout

| Page | Contents | Layout |
|------|----------|--------|
| 1 | Uppercase A–Z | 5 per row × 5 rows (last row has 6) |
| 2 | Lowercase a–z | 10 per row × 3 rows |
| 3 | Digits 0–9 and symbols | Row 1: digits, Row 2: . , ! ? : ; - \_ ( ) |

The background must be **white** and the ink **dark** (any dark color).

---

## Running the pipeline

Run the three scripts in order from the repo root:

```bash
python 01_extract_glyphs.py
python 02_vectorize.py
python 03_build_font.py
```

### Step 1 — `01_extract_glyphs.py`

Rasterizes each PDF page at 300 DPI, detects glyph bounding boxes using
OpenCV contour analysis, and saves one PNG per character.

**Output:** `glyphs/` folder — 72 PNG files

**Filename convention (Windows-safe):**
- Uppercase A–Z → `cap_A.png` … `cap_Z.png`
- Lowercase a–z → `small_a.png` … `small_z.png`
- Digits and symbols → `0.png`, `period.png`, `comma.png`, etc.

> Windows NTFS is case-insensitive, so `A.png` and `a.png` would collide.
> The `cap_`/`small_` prefix keeps filenames unambiguous on all operating systems.

Also saves `glyphs/_debug_page{N}.png` showing detected bounding boxes for
inspection.

**Expected output:**
```
[Page 1] Expected 26, detected 26.
[Page 2] Expected 26, detected 26.
[Page 3] Expected 20, detected 20.
Done. 72 glyph(s) written to 'glyphs/'.
```

### Step 2 — `02_vectorize.py`

Converts each glyph PNG to an SVG vector outline using vtracer with
spline-mode tracing (smooth cubic Bézier curves).

**Output:** `svgs/` folder — 72 SVG files

**Expected output:**
```
Vectorising 72 glyph(s) from 'glyphs/' -> 'svgs/' ...
  [OK]   cap_A.png  -> cap_A.svg
  ...
Done. 72 succeeded, 0 failed.
```

### Step 3 — `03_build_font.py`

Parses each SVG, converts cubic Bézier curves to TrueType-compatible quadratic
splines (cu2qu), applies Y-axis flip and uniform scaling, and assembles a
complete TTF with all required tables.

**Scale reference:** the height of the uppercase A SVG is used to set
cap-height = 700 font units (within a 1000-unit-per-em square).

**Output:** `MyHandwriting.ttf`

**Expected output:**
```
Found 72 SVG(s).  Building 'MyHandwriting.ttf' ...
Scale reference: 'cap_A'  SVG height=... px  => scale=...
72 glyphs built, 0 failed/missing.
Font saved -> MyHandwriting.ttf
```

---

## Font metrics

| Property | Value |
|----------|-------|
| UPM (units per em) | 1000 |
| Cap height | 700 |
| Ascender | 800 |
| Descender | −200 |
| Family name | MyHandwriting |
| Style | Regular |

---

## Glyph coverage

72 glyphs encoded:

- Uppercase A–Z (Unicode 0x41–0x5A)
- Lowercase a–z (Unicode 0x61–0x7A)
- Digits 0–9 (Unicode 0x30–0x39)
- Symbols: `.` `,` `!` `?` `:` `;` `-` `_` `(` `)`

---

## Troubleshooting

**Detection mismatch ("Expected 26, detected 27")**
A stray ink mark is being picked up. Check the corresponding
`glyphs/_debug_pageN.png` to see which box is extra. Adjust `min_area`
in `PAGE_SETTINGS` inside `01_extract_glyphs.py` to filter it out.

**Glyph count too low on page 3**
Small symbols like `:` or `.` may be below the `min_area` threshold.
Lower `min_area` for page 3 and/or increase `dilate_h` to merge stacked dots.

**Font renders glyphs as hollow/inverted**
Change `reverse_direction=True` → `reverse_direction=False` in both
`Cu2QuPen` calls inside `03_build_font.py` and rebuild.
