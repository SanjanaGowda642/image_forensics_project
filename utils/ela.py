"""Error Level Analysis (ELA).

Re-compress the image as JPEG at a known quality and look at the difference.
Regions that were pasted / edited usually react differently to re-compression
than the rest of the picture.
"""
import cv2
import numpy as np


def compute_ela(img_bgr: np.ndarray, quality: int = 90, amplify=None) -> np.ndarray:
    """Return an ELA map (uint8, same size, 3 channels BGR)."""
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG re-compression failed")
    recomp = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img_bgr, recomp).astype(np.float32)
    if amplify is None:                       # auto-scale so the map uses the full range
        peak = float(np.percentile(diff, 99.5))
        amplify = 255.0 / max(peak, 1.0)
    return np.clip(diff * amplify, 0, 255).astype(np.uint8)


def ela_gray(img_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    """Single-channel float32 ELA magnitude (un-normalised, 0..255 scale)."""
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    recomp = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.absdiff(img_bgr, recomp).astype(np.float32).mean(axis=2)
