"""Drawing helpers: tampering overlay with arrows, artifact cut-out, and the
'EXPECTED OUTCOME' report panel (original | overlay | cut-out | restored | gauges | label)."""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .srm import srm_residual_map

NAVY = (23, 52, 110)
ORANGE_BGR = (30, 140, 245)


# --------------------------------------------------------------------------- image-space drawings
def draw_overlay(img_bgr, target_mask, source_mask, arrows):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    red = np.zeros_like(out)
    red[:] = (40, 40, 235)
    tm = target_mask > 0
    out[tm] = cv2.addWeighted(out, 0.45, red, 0.55, 0)[tm]
    sm = source_mask > 0
    if sm.any():
        yel = np.zeros_like(out)
        yel[:] = (40, 190, 250)
        out[sm] = cv2.addWeighted(out, 0.65, yel, 0.35, 0)[sm]
    th = max(2, w // 450)
    for m, col in ((target_mask, (30, 30, 200)), (source_mask, (0, 150, 220))):
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, th, cv2.LINE_AA)
    for (s, t) in arrows:
        p, q = (int(s[0]), int(s[1])), (int(t[0]), int(t[1]))
        cv2.arrowedLine(out, p, q, (0, 0, 0), th + 3, cv2.LINE_AA, tipLength=0.05)
        cv2.arrowedLine(out, p, q, (30, 215, 255), th + 1, cv2.LINE_AA, tipLength=0.05)
        cv2.circle(out, p, th + 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p, th + 2, (30, 215, 255), -1, cv2.LINE_AA)
    return out


def draw_overlay_undetermined(img_bgr, masks):
    """Both clones tinted amber and lettered A, B, ... : we know they are copies of each other,
    but not which one is the original."""
    out = img_bgr.copy()
    w = out.shape[1]
    amber = np.zeros_like(out)
    amber[:] = (0, 165, 255)
    th = max(2, w // 450)
    for i, m in enumerate(masks):
        mb = m > 0
        out[mb] = cv2.addWeighted(out, 0.55, amber, 0.45, 0)[mb]
        cnts, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 120, 220), th, cv2.LINE_AA)
        ys, xs = np.nonzero(mb)
        c = (int(xs.mean()), int(ys.mean()))
        fs = max(1.0, w / 600)
        cv2.putText(out, chr(65 + i), (c[0] - int(10 * fs), c[1] + int(10 * fs)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 1.6, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(out, chr(65 + i), (c[0] - int(10 * fs), c[1] + int(10 * fs)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 1.6, (255, 255, 255), th, cv2.LINE_AA)
    return out


def make_cutout(img_bgr, target_mask, size=360):
    """Zoom-in on the noise residual (SRM) around the largest tampered region."""
    res = srm_residual_map(img_bgr)
    h, w = res.shape
    m = (target_mask > 0).astype(np.uint8)
    if m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh = stats[i, :4]
        pad = int(0.25 * max(bw, bh)) + 8
    else:                                           # authentic: centre crop
        bw = bh = min(h, w) // 3
        x, y = (w - bw) // 2, (h - bh) // 2
        pad = 0
    side = max(bw, bh) + 2 * pad
    cx, cy = x + bw // 2, y + bh // 2
    x0, y0 = int(np.clip(cx - side // 2, 0, max(0, w - side))), int(np.clip(cy - side // 2, 0, max(0, h - side)))
    crop = res[y0:y0 + side, x0:x0 + side]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    out = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if m.any():                                      # faint contour of the region inside the crop
        cm = cv2.resize(m[y0:y0 + side, x0:x0 + side] * 255, (size, size), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, ORANGE_BGR, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (size - 1, size - 1), ORANGE_BGR, 5)
    return out


# --------------------------------------------------------------------------- report panel (PIL)
def _font(sz, bold=False):
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def _pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _fit(bgr, bw, bh):
    h, w = bgr.shape[:2]
    s = min(bw / w, bh / h)
    return cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _gauge(d, cx, cy, r, value, title, fonts):
    d.text((cx, cy - r - 30), title, font=fonts["s_b"], fill=(30, 30, 30), anchor="mm")
    box = [cx - r, cy - r, cx + r, cy + r]
    for i, col in enumerate([(70, 175, 90), (240, 160, 40), (215, 70, 50)]):
        d.arc(box, 180 + 60 * i, 180 + 60 * (i + 1), fill=col, width=14)
    a = math.radians(180 + 180 * float(np.clip(value, 0, 1)))
    d.line([cx, cy, cx + (r - 6) * math.cos(a), cy + (r - 6) * math.sin(a)], fill=(25, 25, 25), width=4)
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(25, 25, 25))
    d.text((cx, cy + 20), f"{value:.2f}", font=fonts["m_b"], fill=(20, 20, 20), anchor="mm")
    d.text((cx, cy + 42), f"({value * 100:.0f}%)", font=fonts["s"], fill=(60, 60, 60), anchor="mm")


def render_panel(original, overlay, cutout, restored, info):
    W, H = 1900, 560
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fonts = {"s": _font(15), "s_b": _font(15, True), "m_b": _font(26, True), "t": _font(19, True), "l_b": _font(21, True)}
    d.rounded_rectangle([8, 8, W - 8, H - 8], radius=14, outline=NAVY, width=3)
    d.rounded_rectangle([20, 0, 420, 34], radius=8, fill=NAVY)
    d.text((34, 17), "ANALYSIS RESULT (OUTPUT)", font=fonts["s_b"], fill="white", anchor="lm")

    titles = ["1) ORIGINAL IMAGE", "2) TAMPERING MASK OVERLAY\nWITH SOURCE-TO-TARGET ARROWS",
              "3) ARTIFACT CUTOUT (ZOOM-IN)", "4) INPAINTED / RESTORED RESULT"]
    cells = [original, overlay, cutout, restored]
    cw, ch, x, y0 = 320, 380, 28, 120
    for t, c in zip(titles, cells):
        d.multiline_text((x + cw // 2, 78), t, font=fonts["s_b"], fill=(20, 20, 20), anchor="mm", align="center")
        f = _pil(_fit(c, cw, ch))
        px, py = x + (cw - f.width) // 2, y0 + (ch - f.height) // 2
        img.paste(f, (px, py))
        d.rectangle([px - 1, py - 1, px + f.width, py + f.height], outline=(120, 120, 120), width=2)
        x += cw + 14
    # divider + gauges
    gx = x + 6
    d.line([gx, 60, gx, H - 40], fill=(150, 150, 150), width=2)
    _gauge(d, gx + 120, 230, 82, info["confidence"], "CONFIDENCE SCORE", fonts)
    _gauge(d, gx + 120, 450, 82, info["intensity"], "TAMPERING INTENSITY", fonts)
    # label box
    bx = gx + 250
    d.rounded_rectangle([bx, 62, W - 24, H - 40], radius=10, outline=NAVY, width=3)
    d.rectangle([bx, 62, W - 24, 96], fill=NAVY)
    d.text(((bx + W - 24) // 2, 79), "PREDICTED LABEL", font=fonts["s_b"], fill="white", anchor="mm")
    d.text(((bx + W - 24) // 2, 150), info["label"].replace(" ", "\n", 1), font=fonts["l_b"], fill=(20, 20, 20), anchor="mm", align="center")
    d.line([bx + 12, 210, W - 36, 210], fill=(170, 170, 170), width=2)
    d.text((bx + 16, 232), "DETAILS", font=fonts["s_b"], fill=(20, 20, 20))
    lines = [f"Confidence: {info['confidence']:.2f} ({info['confidence'] * 100:.0f}%)",
             f"Intensity: {info['intensity']:.2f} ({info['intensity'] * 100:.0f}%)",
             f"Manipulated Area: {info['area_pct']:.1f}%",
             f"Source Regions Matched: {info['regions_matched']}",
             f"Image Resolution: {info['resolution']}",
             f"Keypoint inliers: {info['inliers']}"]
    for i, ln in enumerate(lines):
        d.text((bx + 16, 266 + 30 * i), "\u2022 " + ln, font=fonts["s"], fill=(30, 30, 30))
    if info.get("warning_short"):
        import textwrap
        y = 266 + 30 * len(lines) + 4
        for ln in textwrap.wrap(info["warning_short"], 24):
            d.text((bx + 16, y), ln, font=fonts["s_b"], fill=(205, 105, 0))
            y += 20
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
