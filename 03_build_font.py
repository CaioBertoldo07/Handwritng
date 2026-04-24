"""
03_build_font.py -- Assemble a TTF from the SVG glyph outlines.

Uses fonttools + cu2qu to convert the cubic Bezier curves produced by vtracer
into TrueType-compatible quadratic splines.

Usage: python 03_build_font.py
Input:  svgs/<name>.svg
Output: MyHandwriting.ttf
"""

import os
import sys
import re
import xml.etree.ElementTree as ET

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen

# ── font constants ────────────────────────────────────────────────────────────
UPM         = 1000
CAP_HEIGHT  = 700    # uppercase / digit height target (font units)
ASCENDER    = 800
DESCENDER   = -200
LINE_GAP    = 0
CU2QU_ERR   = 1.0   # max quadratic approximation error (font units)

FAMILY_NAME = "MyHandwriting"
STYLE_NAME  = "Regular"
OUT_FILE    = "MyHandwriting.ttf"
SVG_DIR     = "svgs"

# ── Unicode mapping ───────────────────────────────────────────────────────────
SPECIAL_CODEPOINTS = {
    "period"     : 0x2E,
    "comma"      : 0x2C,
    "exclamation": 0x21,
    "question"   : 0x3F,
    "colon"      : 0x3A,
    "semicolon"  : 0x3B,
    "hyphen"     : 0x2D,
    "underscore" : 0x5F,
    "parenleft"  : 0x28,
    "parenright" : 0x29,
}

def glyph_name_to_codepoint(name: str) -> int | None:
    # cap_A  ... cap_Z   -> ord('A') ... ord('Z')
    if name.startswith("cap_") and len(name) == 5 and name[4].isupper():
        return ord(name[4])
    # small_a ... small_z -> ord('a') ... ord('z')
    if name.startswith("small_") and len(name) == 7 and name[6].islower():
        return ord(name[6])
    if name in SPECIAL_CODEPOINTS:
        return SPECIAL_CODEPOINTS[name]
    # single digit or symbol char
    if len(name) == 1 and name.isascii():
        return ord(name)
    return None

# ── SVG parsing ───────────────────────────────────────────────────────────────

def parse_svg(svg_path: str):
    """
    Parse an SVG produced by vtracer.
    Returns (width, height, path_list) where each path_list entry is
    (tx, ty, d_string).  tx/ty come from a transform="translate(tx,ty)".
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Strip namespace if present
    tag = root.tag
    ns  = ""
    if tag.startswith("{"):
        ns = tag[:tag.index("}") + 1]

    width  = float(root.attrib["width"])
    height = float(root.attrib["height"])

    paths = []
    for elem in root.iter(f"{ns}path"):
        d         = elem.attrib.get("d", "").strip()
        transform = elem.attrib.get("transform", "")
        tx = ty = 0.0
        m = re.match(r"translate\(\s*([^,\s]+)\s*,\s*([^)\s]+)\s*\)", transform)
        if m:
            tx, ty = float(m.group(1)), float(m.group(2))
        if d:
            paths.append((tx, ty, d))

    return width, height, paths


_NUM  = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_TOK  = re.compile(rf"[MmCcLlHhVvZzSsQqTtAa]|{_NUM}")

def _tokenise(d: str):
    return _TOK.findall(d)


def _draw_path(pen, d: str, tx: float, ty: float,
               scale: float, svg_height: float):
    """
    Parse SVG path 'd' and draw it to 'pen' in font coordinate space.

    Coordinate conversion
    ---------------------
    All path coords are absolute in SVG space.  We apply the translate
    (tx, ty) to put them in the SVG viewport, then:

        font_x =  svg_x * scale
        font_y = (svg_height - svg_y) * scale   # Y-flip

    This places the SVG top-left at (0, svg_height*scale) and bottom-left
    at (0, 0), so glyphs sit on the baseline at y=0.
    """
    tokens  = _tokenise(d)
    it      = iter(tokens)
    cur_cmd = "M"

    def next_float():
        return float(next(it))

    def to_font(sx, sy):
        return (sx * scale, (svg_height - sy) * scale)

    def apply_xy(raw_x, raw_y):
        return to_font(raw_x + tx, raw_y + ty)

    try:
        for tok in it:
            if re.match(r"^[a-zA-Z]$", tok):
                cur_cmd = tok
                if cur_cmd.upper() == "Z":
                    pen.closePath()
                continue

            # First numeric token of a new implicit repeat
            first = float(tok)

            if cur_cmd == "M":
                x, y = apply_xy(first, next_float())
                pen.moveTo((x, y))
                cur_cmd = "L"

            elif cur_cmd == "L":
                x, y = apply_xy(first, next_float())
                pen.lineTo((x, y))

            elif cur_cmd == "C":
                x1, y1 = apply_xy(first,       next_float())
                x2, y2 = apply_xy(next_float(), next_float())
                x,  y  = apply_xy(next_float(), next_float())
                pen.curveTo((x1, y1), (x2, y2), (x, y))

            elif cur_cmd == "H":
                # Horizontal line — we don't have a current point here;
                # vtracer doesn't emit H/V but handle it defensively.
                pen.lineTo((apply_xy(first, 0)[0], None))

            elif cur_cmd == "V":
                pen.lineTo((None, apply_xy(0, first)[1]))

            elif cur_cmd == "Z" or cur_cmd == "z":
                pen.closePath()

    except StopIteration:
        pass


# ── notdef glyph ──────────────────────────────────────────────────────────────

def draw_notdef(pen, advance: int = 500):
    """Draw a simple rectangular box for .notdef."""
    m, t = 50, CAP_HEIGHT
    pen.moveTo((m, 0))
    pen.lineTo((advance - m, 0))
    pen.lineTo((advance - m, t))
    pen.lineTo((m, t))
    pen.closePath()
    # inner hole
    b = 80
    pen.moveTo((m + b, b))
    pen.lineTo((m + b, t - b))
    pen.lineTo((advance - m - b, t - b))
    pen.lineTo((advance - m - b, b))
    pen.closePath()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(SVG_DIR):
        print(f"ERROR: '{SVG_DIR}/' not found. Run 02_vectorize.py first.",
              file=sys.stderr)
        sys.exit(1)

    svgs = {
        os.path.splitext(f)[0]: os.path.join(SVG_DIR, f)
        for f in os.listdir(SVG_DIR)
        if f.endswith(".svg")
    }
    if not svgs:
        print(f"No SVG files found in '{SVG_DIR}/'.")
        sys.exit(1)

    print(f"Found {len(svgs)} SVG(s).  Building '{OUT_FILE}' ...\n")

    # ── compute global scale from uppercase-A height ──────────────────────────
    ref_name = next((n for n in ("cap_A", "cap_H", "cap_B") if n in svgs), None)
    if ref_name is None:
        ref_name = next(iter(svgs))
        print(f"WARNING: uppercase 'A' not found; using '{ref_name}' as scale reference.")

    ref_w, ref_h, _ = parse_svg(svgs[ref_name])
    scale = CAP_HEIGHT / ref_h
    print(f"Scale reference: '{ref_name}'  SVG height={ref_h:.1f} px  "
          f"=> scale={scale:.4f}  (1 px = {scale:.2f} font units)\n")

    # ── build glyph order and cmap ────────────────────────────────────────────
    glyph_order = [".notdef"]
    cmap: dict[int, str] = {}

    for name in sorted(svgs):
        cp = glyph_name_to_codepoint(name)
        if cp is None:
            print(f"  SKIP {name} — no Unicode mapping")
            continue
        glyph_order.append(name)
        cmap[cp] = name

    print(f"Glyphs to encode: {len(glyph_order) - 1}  (+ .notdef)\n")

    # ── draw outlines ─────────────────────────────────────────────────────────
    glyf_table: dict[str, object]  = {}
    metrics:    dict[str, tuple]   = {}

    # .notdef
    nd_advance = 500
    tt_pen = TTGlyphPen(None)
    cu_pen = Cu2QuPen(tt_pen, max_err=CU2QU_ERR, reverse_direction=True)
    draw_notdef(cu_pen, nd_advance)
    glyf_table[".notdef"] = tt_pen.glyph()
    metrics[".notdef"]    = (nd_advance, 50)

    ok = fail = 0
    for name in glyph_order[1:]:   # skip .notdef
        svg_path = svgs.get(name)
        if svg_path is None or not os.path.exists(svg_path):
            print(f"  WARN  {name:20s} — SVG missing, using .notdef")
            glyf_table[name] = glyf_table[".notdef"]
            metrics[name]    = metrics[".notdef"]
            fail += 1
            continue

        try:
            svg_w, svg_h, path_list = parse_svg(svg_path)

            advance = max(1, round(svg_w * scale))
            lsb     = 0

            tt_pen = TTGlyphPen(None)
            cu_pen = Cu2QuPen(tt_pen, max_err=CU2QU_ERR, reverse_direction=True)

            for tx, ty, d in path_list:
                _draw_path(cu_pen, d, tx, ty, scale, svg_h)

            glyph = tt_pen.glyph()
            glyf_table[name] = glyph
            metrics[name]    = (advance, lsb)

            print(f"  [OK]  {name:20s}  adv={advance:4d}  paths={len(path_list)}")
            ok += 1

        except Exception as exc:
            print(f"  FAIL  {name:20s} — {exc}")
            glyf_table[name] = glyf_table[".notdef"]
            metrics[name]    = metrics[".notdef"]
            fail += 1

    print(f"\n{ok} glyphs built, {fail} failed/missing.\n")

    # ── assemble the font ─────────────────────────────────────────────────────
    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyf_table)
    fb.setupHorizontalMetrics(metrics)

    fb.setupHorizontalHeader(ascent=ASCENDER, descent=DESCENDER)
    fb.setupNameTable({
        "familyName"        : FAMILY_NAME,
        "styleName"         : STYLE_NAME,
        "uniqueFontIdentifier": f"{FAMILY_NAME}-{STYLE_NAME}",
        "fullName"          : f"{FAMILY_NAME} {STYLE_NAME}",
        "version"           : "Version 1.0",
        "psName"            : f"{FAMILY_NAME}-{STYLE_NAME}",
    })
    fb.setupOS2(
        sTypoAscender   = ASCENDER,
        sTypoDescender  = DESCENDER,
        sTypoLineGap    = LINE_GAP,
        usWinAscent     = ASCENDER,
        usWinDescent    = abs(DESCENDER),
        sxHeight        = round(CAP_HEIGHT * 0.72),
        sCapHeight      = CAP_HEIGHT,
        fsType          = 0,
    )
    fb.setupPost()
    fb.setupHead(unitsPerEm=UPM)

    fb.font.save(OUT_FILE)
    print(f"Font saved -> {OUT_FILE}")
    print(f"  UPM={UPM}, cap_height={CAP_HEIGHT}, scale={scale:.4f}")
    print(f"  Glyphs: {len(glyph_order)} total ({ok} unique outlines)")


if __name__ == "__main__":
    main()
