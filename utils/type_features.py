"""Patch-level forensic features for tamper-TYPE identification (no deep learning, no PyTorch).

Every 32x32 patch (stride 16) of the image is described by traces that different edits leave behind:

  * ELA at two JPEG qualities           -> recompression history (splices, pasted regions)
  * three SRM noise residual energies   -> sensor / processing noise (splice, inpaint, denoise)
  * median-filter residual, Laplacian, gradient, texture energy -> sharpness / blur / smoothness
  * 8x8 JPEG-grid blockiness            -> a different compression grid than the rest of the image
  * Lab colour statistics, saturation   -> colour edits
Each feature is also given (a) relative to the image median and (b) relative to the 8 neighbouring
patches, because tampering shows up as "this patch differs from its surroundings".
"""
import cv2
import numpy as np

from utils.ela import ela_gray
from utils.srm import SRM_KERNELS

PATCH, STRIDE, WORK_MAX = 32, 16, 384
TOPK = 5                  # image score = mean of the strongest TOPK (smoothed) patches

RAW = ["ela90", "ela75", "srm1", "srm2", "srm3", "noise", "lap", "grad", "tex",
       "L", "a", "b", "Lstd", "astd", "bstd", "sat", "blk"]
REL = RAW[:9]                                           # the log-energy features, minus the image median
CTX = ["ela90", "srm1", "srm2", "noise", "lap", "grad", "L", "a", "b", "blk"]   # minus the neighbours' mean
FEATURE_NAMES = RAW + ["rel_" + n for n in REL] + ["ctx_" + n for n in CTX]
N_FEATURES = len(FEATURE_NAMES)


def prepare(img):
    """Work resolution used for training AND inference (so noise statistics stay comparable)."""
    h, w = img.shape[:2]
    s = min(1.0, WORK_MAX / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(PATCH, int(w * s)), max(PATCH, int(h * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if h < PATCH or w < PATCH:
        img = cv2.copyMakeBorder(img, 0, max(0, PATCH - h), 0, max(0, PATCH - w), cv2.BORDER_REFLECT)
    return img


def centers(h, w):
    return np.arange(PATCH // 2, h - PATCH // 2 + 1, STRIDE), np.arange(PATCH // 2, w - PATCH // 2 + 1, STRIDE)


def _box(m):
    return cv2.boxFilter(m, -1, (PATCH, PATCH), normalize=True, borderType=cv2.BORDER_REFLECT)


def _neighbour_mean(grid):
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    k /= 8.0
    return cv2.filter2D(grid.astype(np.float32), -1, k, borderType=cv2.BORDER_REPLICATE)


def patch_features(img_bgr):
    """Returns (features [ny*nx, N_FEATURES] float32, (ny, nx), prepared_image)."""
    img = prepare(img_bgr)
    h, w = img.shape[:2]
    ys, xs = centers(h, w)
    at = lambda m: _box(m)[np.ix_(ys, xs)]
    g8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = g8.astype(np.float32)
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)

    lap = cv2.Laplacian(g, cv2.CV_32F)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    dx, dy = np.zeros_like(g), np.zeros_like(g)
    dx[:, :-1], dy[:-1, :] = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    col = (np.arange(w) % 8 == 7).astype(np.float32)[None, :]
    row = (np.arange(h) % 8 == 7).astype(np.float32)[:, None]
    m_g = at(g)
    f = {
        "ela90": np.log1p(at(ela_gray(img, 90))),
        "ela75": np.log1p(at(ela_gray(img, 75))),
        "noise": np.log1p(at(np.abs(g - cv2.medianBlur(g8, 3).astype(np.float32)))),
        "lap": np.log1p(at(lap * lap)),
        "grad": np.log1p(at(np.sqrt(gx * gx + gy * gy))),
        "tex": np.log1p(np.sqrt(np.maximum(at(g * g) - m_g ** 2, 0))),
        "sat": at(sat),
        "blk": at(dx * col + dy * row) / (at(dx * (1 - col) + dy * (1 - row)) + 1e-3),
    }
    for i, k in enumerate(SRM_KERNELS):
        f[f"srm{i + 1}"] = np.log1p(at(np.abs(cv2.filter2D(g, cv2.CV_32F, k))))
    for i, n in enumerate(("L", "a", "b")):
        ch = lab[..., i]
        m = at(ch)
        f[n], f[n + "std"] = m, np.sqrt(np.maximum(at(ch * ch) - m ** 2, 0))
    f["Lstd"], f["astd"], f["bstd"] = f.pop("Lstd"), f.pop("astd"), f.pop("bstd")

    cols = [f[n] for n in RAW]
    cols += [f[n] - np.median(f[n]) for n in REL]
    cols += [f[n] - _neighbour_mean(f[n]) for n in CTX]
    ny, nx = len(ys), len(xs)
    return np.stack([c.astype(np.float32) for c in cols], -1).reshape(ny * nx, -1), (ny, nx), img


def patch_cover(mask, prepared_shape):
    """Fraction of every patch covered by `mask` (resized to the prepared image) -> flat array."""
    h, w = prepared_shape[:2]
    m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) if mask.shape[:2] != (h, w) else mask
    ys, xs = centers(h, w)
    return _box((m > 127).astype(np.float32))[np.ix_(ys, xs)].ravel()
