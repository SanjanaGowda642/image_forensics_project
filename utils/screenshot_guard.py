"""Screenshot / UI guard.

Interfaces, documents and slides contain legitimate exact repeats (icons, list rows,
buttons, letters). Those look identical to a copy-move forgery for a key-point matcher,
so we measure how 'UI-like' (flat, few distinct colours) the image and the matched
regions are, and downgrade the verdict when a detection is only repeated UI.
"""
import cv2
import numpy as np


def _top_colour_coverage(pix_bgr, top=8, shift=3):
    """Fraction of pixels that fall in the `top` most common (quantised) colours."""
    if len(pix_bgr) == 0:
        return 0.0
    q = (pix_bgr.astype(np.int32) >> shift)
    key = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    _, counts = np.unique(key, return_counts=True)
    return float(np.sort(counts)[::-1][:top].sum() / len(key))


def _rectilinear_edge_ratio(gray_u8, run=21):
    """Share of strong edge pixels that sit on long horizontal / vertical straight runs.
    Interfaces (boxes, rows, dividers, tables) have lots of these; photos and drawings of
    trees, hills, faces... almost never do."""
    edges = cv2.Canny(gray_u8, 60, 150)
    n = int((edges > 0).sum())
    if n < 50:
        return 0.0
    hor = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1)))
    ver = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, run)))
    return float(((hor > 0) | (ver > 0)).sum() / n)


def global_ui_score(img_bgr):
    """0 = natural photo / drawing, 1 = flat interface or document."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (5, 5))
    var = cv2.blur(g * g, (5, 5)) - mean * mean
    flat = float((var < 0.3).mean())                      # truly constant areas (was 1.5: smooth sky/grass counted)
    cov = _top_colour_coverage(img_bgr.reshape(-1, 3), top=16, shift=1)   # was shift=3: gradients collapsed into few colours
    rect = _rectilinear_edge_ratio(gray)
    base = 0.5 * min(1.0, flat / 0.5) + 0.5 * min(1.0, cov / 0.7)
    # a screenshot must ALSO have straight UI lines; without them we trust the score only a little
    score = base * (0.35 + 0.65 * min(1.0, rect / 0.25))
    return float(score), flat, cov


def region_ui_score(img_bgr, mask):
    """How UI-like the detected (cloned) pixels are: share covered by a handful of flat colours."""
    pix = img_bgr[mask > 0]
    return _top_colour_coverage(pix, top=8)


def assess(img_bgr, target_mask, source_mask, forged):
    """Returns dict(ui_score, region_score, is_ui_repeat, warning)."""
    g, flat, cov = global_ui_score(img_bgr)
    out = {"ui_score": round(g, 2), "region_score": None, "is_ui_repeat": False, "warning": None, "short": None}
    if forged:
        both = cv2.bitwise_or(target_mask, source_mask)
        r = region_ui_score(img_bgr, both)
        out["region_score"] = round(r, 2)
        if r >= 0.70 or (g >= 0.55 and r >= 0.50):
            out["is_ui_repeat"] = True
            out["warning"] = ("The matched regions are flat, few-colour areas typical of a screenshot "
                              "(repeated icons, buttons, rows or text). This is probably a design "
                              "repeat, not a forgery - verdict downgraded to INCONCLUSIVE.")
            out["short"] = "Repeated UI elements, probably not a forgery."
            return out
    if g >= 0.50:
        out["warning"] = ("Input looks like a screenshot of an interface or document. Edits to text or "
                          "UI are not copy-move forgeries and cannot be detected here.")
        out["short"] = "Screenshot input: low reliability."
    elif g >= 0.32:
        out["warning"] = ("Part of the image looks like screenshot UI; noise-based cues are weaker "
                          "than on an original camera photo.")
        out["short"] = "Screenshot UI present: weaker cues."
    return out

def find_picture_region(img_bgr, block=16, min_colors=22, min_frac=0.10, max_frac=0.92):
    """Locate the photo / illustration inside a screenshot (browser chrome, tabs, margins, text are flat or
    few-colour; a picture is made of blocks with many distinct colours).
    Returns (x, y, w, h) or None when the whole image already is the picture / no picture was found."""
    h, w = img_bgr.shape[:2]
    nby, nbx = h // block, w // block
    if nby < 6 or nbx < 6:
        return None
    q = (img_bgr[:nby * block, :nbx * block] >> 3).astype(np.int32)
    key = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    blocks = key.reshape(nby, block, nbx, block).transpose(0, 2, 1, 3).reshape(nby, nbx, block * block)
    s = np.sort(blocks, axis=2)
    cnt = 1 + (s[..., 1:] != s[..., :-1]).sum(2)
    photo = (cnt >= min_colors).astype(np.uint8) * 255
    k = max(3, int(0.04 * min(nby, nbx)) | 1)
    photo = cv2.morphologyEx(photo, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    photo = cv2.morphologyEx(photo, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(photo)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh = (int(v) for v in stats[i, :4])
    if bw * bh < min_frac * nbx * nby:
        return None
    x0, y0, x1, y1 = bx * block, by * block, (bx + bw) * block, (by + bh) * block
    if (x1 - x0) * (y1 - y0) > max_frac * h * w:                 # already (almost) all picture
        return None
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)
