"""Synthetic copy-move dataset + demo-scene generator (no downloads needed).

* make_landscape()      -> procedural mountain / lake scene with a small house
* make_background()     -> abstract textured scene (training variety)
* paste_region()        -> the actual copy-move operation (rotate / scale / feather)
* make_demo_forgery()   -> landscape whose house has been cloned twice (used by the web demo)
* make_sample()         -> one random sample of ANY class (see CLASSES)
* generate_dataset()    -> writes   images/sample_XXXX.png , masks/sample_XXXX.png , labels.csv

Classes: authentic | copy-move | splicing | removal (inpainting) | retouch (local edits)

Run:  python data/dataset_generator.py --n 500
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- helpers
def _fractal(h, w, rng, octaves=5, base=3, persistence=0.55):
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh = max(2, base * 2 ** o)
        gw = max(2, int(round(base * 2 ** o * w / h)))
        n = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
    out /= total
    return (out - out.min()) / max(float(out.max() - out.min()), 1e-6)


def _ridge(w, rng, base_y, amp, octaves=6):
    xs = np.linspace(0, 1, w)
    y = np.zeros(w, np.float32)
    a, f = 1.0, 2.0
    for _ in range(octaves):
        y += a * np.interp(xs, np.linspace(0, 1, int(f) + 2), rng.random(int(f) + 2))
        a *= 0.5
        f *= 2
    y = (y - y.min()) / max(float(y.max() - y.min()), 1e-6)
    return base_y - amp * y


def blob_mask(h, w, cx, cy, r, rng, n_vert=14):
    ang = np.linspace(0, 2 * np.pi, n_vert, endpoint=False)
    rad = r * rng.uniform(0.82, 1.15, n_vert)
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return cv2.GaussianBlur(m, (0, 0), 1.5) > 127


# --------------------------------------------------------------------------- scenes
def make_landscape(w=1024, h=768, seed=0):
    """Procedural lake-and-mountain scene. Returns (bgr uint8, house_box (x, y, bw, bh))."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    hz = int(0.50 * h)

    # sky + clouds
    t = np.linspace(0, 1, h)[:, None, None]
    top = np.array([200, 140, 70], np.float32) + rng.uniform(-15, 15, 3)
    bot = np.array([245, 225, 190], np.float32)
    img[:] = top * (1 - t) + bot * t
    clouds = _fractal(h, w, rng, 6, 3)
    cl = np.clip((clouds - 0.52) * 3.0, 0, 1)[..., None]
    img = img * (1 - cl * 0.8) + 255 * cl * 0.8

    # two mountain ridges
    for k, (base, amp, col) in enumerate([(0.42, 0.22, (150, 120, 105)), (0.52, 0.18, (95, 85, 70))]):
        ry = _ridge(w, rng, base * h, amp * h)
        yy = np.arange(h)[:, None]
        m = (yy >= ry[None, :]).astype(np.float32)
        tex = 0.65 + 0.7 * _fractal(h, w, rng, 7, 6)
        shade = 0.75 + 0.5 * np.clip((yy - ry[None, :]) / (0.25 * h), 0, 1)
        layer = np.array(col, np.float32)[None, None, :] * (tex * shade)[..., None]
        img = img * (1 - m[..., None]) + layer * m[..., None]

    # meadow / hills
    ground = np.array([55, 120, 60], np.float32)[None, None, :] * (0.6 + 0.8 * _fractal(h, w, rng, 7, 8))[..., None]
    gm = np.zeros((h, w), np.float32)
    gm[hz + int(0.02 * h):] = 1
    gm = cv2.GaussianBlur(gm, (0, 0), 2)
    img = img * (1 - gm[..., None]) + ground * gm[..., None]

    # lake with reflected sky / mountains
    lake = np.zeros((h, w), np.uint8)
    cv2.ellipse(lake, (int(0.45 * w), int(0.64 * h)), (int(0.40 * w), int(0.115 * h)), 0, 0, 360, 255, -1)
    top_y = int(0.64 * h - 0.115 * h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src_y = np.clip(2 * (top_y - 4) - yy, 0, h - 1)
    src_x = np.clip(xx + 7 * np.sin(yy * 0.21 + rng.uniform(0, 6)) + 3 * np.sin(yy * 0.7), 0, w - 1)
    refl = cv2.remap(img.astype(np.float32), src_x, src_y, cv2.INTER_LINEAR)
    refl = refl * np.array([0.82, 0.9, 0.8], np.float32) + np.array([25, 15, 0], np.float32)
    lm = cv2.GaussianBlur(lake, (0, 0), 2).astype(np.float32)[..., None] / 255
    img = img * (1 - lm) + refl * lm

    # trees (thousands of little textured blobs -> plenty of SIFT keypoints)
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    path_poly = np.array([[int(0.60 * w), h], [int(0.88 * w), h], [int(0.72 * w), int(0.66 * h)], [int(0.66 * w), int(0.66 * h)]])
    pathm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(pathm, [path_poly], 255)
    n_tree = int(2800 * w * h / (1024 * 768))
    for _ in range(n_tree):
        x = int(rng.integers(0, w))
        y = int(rng.integers(int(0.48 * h), h))
        if lake[min(y, h - 1), x] or pathm[min(y, h - 1), x]:
            continue
        r = int(rng.integers(3, 8) * (0.6 + (y / h)))
        g = rng.uniform(0.5, 1.2)
        col = (int(30 * g), int(100 * g * rng.uniform(0.8, 1.2)), int(40 * g))
        cv2.ellipse(img8, (x, y), (r, int(r * 1.4)), 0, 0, 360, col, -1, cv2.LINE_AA)
        cv2.circle(img8, (x - r // 3, y - r // 2), max(1, r // 3), (col[0] + 25, col[1] + 45, col[2] + 20), -1, cv2.LINE_AA)
    # path
    pt = cv2.GaussianBlur(pathm, (0, 0), 2)[..., None].astype(np.float32) / 255
    ptex = (np.array([90, 140, 170], np.float32)[None, None, :] * (0.7 + 0.6 * _fractal(h, w, rng, 7, 10))[..., None])
    img8 = np.clip(img8 * (1 - pt) + ptex * pt, 0, 255).astype(np.uint8)

    # house
    bw = int(0.085 * w)
    bh = int(0.065 * w)
    hx, hy = int(0.36 * w), int(0.80 * h)
    body = (hx, hy, bw, bh)
    cv2.rectangle(img8, (hx, hy), (hx + bw, hy + bh), (200, 215, 235), -1)
    for ry in range(hy + 3, hy + bh, 6):                                # clapboard lines
        cv2.line(img8, (hx, ry), (hx + bw, ry), (170, 185, 205), 1, cv2.LINE_AA)
    cv2.fillPoly(img8, [np.array([[hx - 6, hy], [hx + bw + 6, hy], [hx + bw // 2, hy - int(0.6 * bh)]])], (45, 50, 185), cv2.LINE_AA)
    cv2.rectangle(img8, (hx + int(0.14 * bw), hy + int(0.25 * bh)), (hx + int(0.40 * bw), hy + int(0.55 * bh)), (120, 80, 40), -1)
    cv2.rectangle(img8, (hx + int(0.58 * bw), hy + int(0.40 * bh)), (hx + int(0.86 * bw), hy + bh), (40, 60, 100), -1)
    cv2.line(img8, (hx + int(0.58 * bw) + bw // 7, hy + int(0.40 * bh)), (hx + int(0.58 * bw) + bw // 7, hy + bh), (200, 215, 235), 1)
    cv2.circle(img8, (hx + int(0.80 * bw), hy + int(0.72 * bh)), 2, (60, 200, 230), -1)

    img = img8.astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    img += rng.normal(0, 4.0, img.shape).astype(np.float32)              # sensor grain
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, (hx - 6, hy - int(0.6 * bh), bw + 12, bh + int(0.6 * bh))


def make_background(size=256, seed=0):
    """Abstract textured image for training variety."""
    rng = np.random.default_rng(seed)
    h = w = size
    chans = [_fractal(h, w, rng, 6, int(rng.integers(2, 6))) for _ in range(3)]
    base = np.stack(chans, -1) * rng.uniform(120, 230, 3) + rng.uniform(0, 60, 3)
    img = base.astype(np.uint8)
    for _ in range(int(rng.integers(25, 60))):
        c = tuple(int(v) for v in rng.integers(20, 235, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = int(rng.integers(6, size // 6))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(img, (x, y), s, c, -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + s, y + int(s * rng.uniform(0.5, 1.6))), c, -1)
        else:
            pts = np.stack([x + rng.integers(-s, s, 3), y + rng.integers(-s, s, 3)], 1).astype(np.int32)
            cv2.fillPoly(img, [pts], c, cv2.LINE_AA)
    f = img.astype(np.float32) * (0.85 + 0.3 * _fractal(h, w, rng, 7, 16))[..., None]
    f += rng.normal(0, 4.0, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- copy-move
def paste_region(img, region_mask, offset, angle=0.0, scale=1.0, feather=1.2):
    """Copy `region_mask` pixels of `img` to a shifted / rotated / scaled location.
    Returns (forged_bgr, target_mask_uint8)."""
    h, w = img.shape[:2]
    ys, xs = np.nonzero(region_mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[:, 2] += np.asarray(offset, np.float64)
    src = (region_mask.astype(np.uint8) * 255) if region_mask.dtype != np.uint8 else region_mask
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    tmask = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_NEAREST)
    alpha = cv2.GaussianBlur(tmask.astype(np.float32) / 255.0, (0, 0), feather)[..., None]
    out = img.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), (tmask > 127).astype(np.uint8) * 255


def _jpeg_roundtrip(img, quality):
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def make_demo_forgery(seed=7, w=1024, h=768):
    """Landscape whose house was cloned twice. Returns (forged_bgr, ground_truth_mask)."""
    img, (bx, by, bw, bh) = make_landscape(w, h, seed)
    rng = np.random.default_rng(seed + 1)
    cx, cy = bx + bw / 2, by + bh / 2
    r = 0.62 * max(bw, bh)
    src = blob_mask(h, w, cx, cy, r, rng).astype(np.uint8) * 255
    forged, gt = img, np.zeros((h, w), np.uint8)
    for dx, dy, ang, sc in [(-0.30 * w, 0.04 * h, 0, 1.0), (0.46 * w, -0.17 * h, 8, 1.05)]:
        forged, tm = paste_region(forged, src, (dx, dy), ang, sc, feather=1.0)
        gt = np.maximum(gt, tm)
    return _jpeg_roundtrip(forged, 92), gt


def _random_forgery(img, rng, max_ops=2):
    h, w = img.shape[:2]
    gt = np.zeros((h, w), np.uint8)
    out = img
    for _ in range(int(rng.integers(1, max_ops + 1))):
        for _try in range(30):
            r = rng.uniform(0.07, 0.15) * min(h, w)
            sx, sy = rng.uniform(r, w - r), rng.uniform(r, h - r)
            ang = float(rng.uniform(-12, 12)) if rng.random() < 0.3 else 0.0
            sc = float(rng.uniform(0.92, 1.1)) if rng.random() < 0.3 else 1.0
            tx, ty = rng.uniform(1.3 * r, w - 1.3 * r), rng.uniform(1.3 * r, h - 1.3 * r)
            if np.hypot(tx - sx, ty - sy) < 2.6 * r:
                continue
            src = blob_mask(h, w, sx, sy, r, rng).astype(np.uint8) * 255
            out, tm = paste_region(out, src, (tx - sx, ty - sy), ang, sc, feather=float(rng.uniform(0.6, 1.8)))
            gt = np.maximum(gt, tm)
            break
    return out, gt


# --------------------------------------------------------------------------- more tampering types
CLASSES = ["authentic", "copy-move", "splicing", "removal", "retouch"]


def _random_blob(h, w, rng, lo=0.06, hi=0.14):
    r = rng.uniform(lo, hi) * min(h, w)
    cx, cy = rng.uniform(r + 2, w - r - 2), rng.uniform(r + 2, h - r - 2)
    return blob_mask(h, w, cx, cy, r, rng)


def _donor_style(donor, rng):
    """Give the foreign image a different 'camera history' (what makes real splices detectable)."""
    k = int(rng.integers(0, 5))
    d = donor.copy()
    if k == 0:
        d = np.clip(d.astype(np.float32) + rng.normal(0, rng.uniform(3, 9), d.shape), 0, 255).astype(np.uint8)
    elif k == 1:
        d = cv2.GaussianBlur(d, (0, 0), rng.uniform(0.8, 1.8))
    elif k == 2:
        d = _jpeg_roundtrip(d, int(rng.integers(35, 75)))
    elif k == 3:
        s = rng.uniform(0.55, 0.85)
        small = cv2.resize(d, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        d = cv2.resize(small, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        d = np.clip(d.astype(np.float32) * rng.uniform(0.85, 1.15, 3) + rng.uniform(-15, 15, 3), 0, 255).astype(np.uint8)
    return d


def _splice(img, rng):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    gt = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(1, 3))):
        s = int(rng.integers(0, 10 ** 8))
        donor = make_landscape(w, h, s)[0] if rng.random() < 0.5 else make_background(w, s)
        donor = _donor_style(donor, rng)
        m = _random_blob(h, w, rng)
        alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), float(rng.uniform(0.7, 2.2)))[..., None]
        alpha *= float(rng.uniform(0.85, 1.0))
        out = out * (1 - alpha) + donor.astype(np.float32) * alpha
        gt = np.maximum(gt, m.astype(np.uint8) * 255)
    return np.clip(out, 0, 255).astype(np.uint8), gt


def _removal(img, rng):
    """Object removal: the region is deleted and filled by classical inpainting."""
    h, w = img.shape[:2]
    m = _random_blob(h, w, rng).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    method = cv2.INPAINT_TELEA if rng.random() < 0.5 else cv2.INPAINT_NS
    out = cv2.inpaint(img, m, int(rng.integers(3, 8)), method)
    if rng.random() < 0.35:                                # some tools re-add grain
        a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.5)[..., None]
        out = np.clip(out.astype(np.float32) + rng.normal(0, rng.uniform(2, 5), out.shape) * a, 0, 255).astype(np.uint8)
    return out, m


def _retouch(img, rng):
    """Local edits: blur, sharpen, tone, colour, denoise, liquify."""
    h, w = img.shape[:2]
    m = _random_blob(h, w, rng, 0.07, 0.16)
    alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), float(rng.uniform(1.5, 4.0)))[..., None]
    k = int(rng.integers(0, 6))
    f = img.astype(np.float32)
    if k == 0:
        e = cv2.GaussianBlur(f, (0, 0), float(rng.uniform(1.5, 3.5)))
    elif k == 1:
        e = cv2.addWeighted(f, 1.9, cv2.GaussianBlur(f, (0, 0), 2.0), -0.9, 0)
    elif k == 2:
        e = f * float(rng.uniform(0.7, 1.3)) + float(rng.uniform(-35, 35))
    elif k == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-12, 12)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.6, 1.5), 0, 255)
        e = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    elif k == 4:
        e = cv2.bilateralFilter(img, 9, 45, 9).astype(np.float32)
    else:
        dx = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 9) * 60 * float(rng.uniform(0.5, 1.0))
        dy = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 9) * 60 * float(rng.uniform(0.5, 1.0))
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        e = cv2.remap(f, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = np.clip(f * (1 - alpha) + e * alpha, 0, 255).astype(np.uint8)
    return out, (alpha[..., 0] > 0.5).astype(np.uint8) * 255


def make_ui_screenshot(size=256, seed=0):
    """Flat interface with legitimately repeated icons / rows / buttons (always AUTHENTIC)."""
    rng = np.random.default_rng(seed)
    sc = size / 256.0
    img = np.full((size, size, 3), int(rng.integers(232, 255)), np.uint8)
    hc = tuple(int(v) for v in rng.integers(20, 200, 3))
    head = int(44 * sc)
    cv2.rectangle(img, (0, 0), (size, head), hc, -1)
    cv2.putText(img, "Messages", (int(10 * sc), int(30 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * sc, (255, 255, 255), max(1, int(2 * sc)), cv2.LINE_AA)
    n = int(rng.integers(4, 9))
    row_h = (size - head) // n
    ic = tuple(int(v) for v in rng.integers(30, 220, 3))
    bc = tuple(int(v) for v in rng.integers(30, 200, 3))
    t1 = ["New message", "Tap to open", "Reminder", "Download ready"][int(rng.integers(0, 4))]
    for i in range(n):
        y = head + i * row_h
        cv2.rectangle(img, (int(4 * sc), y + 2), (size - int(4 * sc), y + row_h - 2), (255, 255, 255), -1)
        cv2.rectangle(img, (int(4 * sc), y + 2), (size - int(4 * sc), y + row_h - 2), (215, 215, 215), 1)
        cy = y + row_h // 2
        cv2.circle(img, (int(24 * sc), cy), max(4, int(row_h * 0.3)), ic, -1, cv2.LINE_AA)
        cv2.putText(img, t1, (int(54 * sc), cy + int(4 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.rectangle(img, (size - int(70 * sc), cy - int(row_h * 0.25)), (size - int(14 * sc), cy + int(row_h * 0.25)), bc, -1)
    return img


def _screenshot_wrap(img, mask, rng):
    """Put the picture inside a flat app frame (header bar + margins), like a phone screenshot."""
    h, w = img.shape[:2]
    canvas = np.full((h, w, 3), int(rng.integers(235, 255)), np.uint8)
    cmask = np.zeros((h, w), np.uint8)
    head = int(h * 0.16)
    cv2.rectangle(canvas, (0, 0), (w, head), tuple(int(v) for v in rng.integers(20, 150, 3)), -1)
    m = int(w * 0.04)
    tw, th = w - 2 * m, h - head - 2 * m
    canvas[head + m:head + m + th, m:m + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    cmask[head + m:head + m + th, m:m + tw] = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    return canvas, cmask


def _postprocess(img, mask, rng):
    """Real-world life of an image: rescaling, screenshots, (double) JPEG."""
    h, w = img.shape[:2]
    if rng.random() < 0.15:
        s = rng.uniform(0.7, 1.3)
        img = cv2.resize(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.12:
        img, mask = _screenshot_wrap(img, mask, rng)
    u = rng.random()
    if u < 0.20:
        img = _jpeg_roundtrip(_jpeg_roundtrip(img, int(rng.integers(55, 90))), int(rng.integers(70, 97)))
    elif u < 0.70:
        img = _jpeg_roundtrip(img, int(rng.integers(65, 97)))
    return img, mask


def make_sample(rng, size=256):
    """One random training sample. Returns (bgr uint8, mask uint8 0/255, class_index)."""
    if rng.random() < 0.14:
        img = make_ui_screenshot(size, int(rng.integers(0, 10 ** 8)))
        mask, cls = np.zeros((size, size), np.uint8), 0
        if rng.random() < 0.4:
            img = _jpeg_roundtrip(img, int(rng.integers(70, 97)))
        return img, mask, cls
    s = int(rng.integers(0, 10 ** 8))
    base = make_landscape(size, size, s)[0] if rng.random() < 0.55 else make_background(size, s)
    cls = int(rng.choice(5, p=[0.24, 0.19, 0.19, 0.19, 0.19]))
    if cls == 1:
        img, mask = _random_forgery(base, rng)
    elif cls == 2:
        img, mask = _splice(base, rng)
    elif cls == 3:
        img, mask = _removal(base, rng)
    elif cls == 4:
        img, mask = _retouch(base, rng)
    else:
        img, mask = base, np.zeros(base.shape[:2], np.uint8)
    if cls and not mask.any():
        cls = 0
    img, mask = _postprocess(img, mask, rng)
    return img, (mask > 127).astype(np.uint8) * 255, cls


# --------------------------------------------------------------------------- dataset
def generate_dataset(out_dir=None, n=100, size=256, seed=0, verbose=True):
    out_dir = out_dir or os.path.join(ROOT, "data", "synthetic_dataset")
    img_dir, msk_dir = os.path.join(out_dir, "images"), os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = [("name", "cls", "forged")]
    counts = [0] * len(CLASSES)
    for i in range(n):
        img, mask, cls = make_sample(rng, size)
        name = f"sample_{i:05d}"
        cv2.imwrite(os.path.join(img_dir, name + ".png"), img)
        cv2.imwrite(os.path.join(msk_dir, name + ".png"), mask)
        rows.append((name, cls, int(cls > 0)))
        counts[cls] += 1
        if verbose and (i + 1) % 100 == 0:
            print(f"  generated {i + 1}/{n}")
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    if verbose:
        print(f"[dataset] {n} samples -> {out_dir}   " + "  ".join(f"{c}={k}" for c, k in zip(CLASSES, counts)))
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    generate_dataset(a.out, a.n, a.size, a.seed)
