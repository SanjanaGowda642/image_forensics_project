"""Steganalysis Rich Model (SRM) high-pass noise residuals.

Three classic SRM kernels (Fridrich & Kodovsky) suppress image content and keep
the sensor / processing noise, which is exactly where splices and clones leave
traces.
"""
import cv2
import numpy as np

_K1 = np.array([[0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0]], np.float32) / 4.0
_K2 = np.array([[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]], np.float32) / 12.0
_K3 = np.array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]], np.float32) / 2.0
SRM_KERNELS = (_K1, _K2, _K3)


def compute_srm(img_bgr: np.ndarray, truncate: float = 2.0) -> np.ndarray:
    """Return float32 HxWx3 residuals, truncated to [-truncate, truncate] (network input)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = [np.clip(cv2.filter2D(gray, cv2.CV_32F, k), -truncate, truncate) for k in SRM_KERNELS]
    return np.stack(out, axis=-1)


def srm_residual_map(img_bgr: np.ndarray) -> np.ndarray:
    """Contrast-enhanced grayscale noise map (uint8) used for the 'artifact cut-out'."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = np.zeros_like(gray)
    for k in SRM_KERNELS:
        res += np.abs(cv2.filter2D(gray, cv2.CV_32F, k))
    res = cv2.GaussianBlur(res, (0, 0), 0.8)
    hi = float(np.percentile(res, 99.0))
    res = np.clip(res / max(hi, 1e-6), 0, 1) ** 0.7
    res8 = (res * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    return clahe.apply(res8)
