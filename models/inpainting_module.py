"""Restoration module: removes the tampered region and rebuilds it from its surroundings.

Exemplar-based inpainting (Criminisi et al., 2004), implemented on OpenCV:
  * the hole is filled patch by patch, strongest edges / best-supported pixels first
  * every patch is copied from the most similar fully-known window of the image
    (cv2.matchTemplate with a mask), so real texture is reused instead of blurred
  * windows that overlap a tampered region or the matched original object are never
    used as a source, so the removed object cannot come back
Falls back to Telea/Navier-Stokes for anything left over.
"""
import cv2
import numpy as np


class Inpainter:
    def __init__(self, patch=9, dilate_px=4, work_side=560, max_iter=6000, locality=40.0):
        self.patch = patch | 1
        self.dilate_px = dilate_px
        self.work_side = work_side
        self.max_iter = max_iter
        self.locality = locality                  # penalty that prefers nearby source patches

    def __call__(self, img_bgr, mask, avoid=None):
        return self.inpaint(img_bgr, mask, avoid)

    # ------------------------------------------------------------------ public
    def inpaint(self, img_bgr, mask, avoid=None):
        """img_bgr uint8, mask: tampered pixels (>0) to remove, avoid: pixels never used as texture source."""
        if mask is None or not np.any(mask):
            return img_bgr.copy()
        h, w = img_bgr.shape[:2]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.dilate_px + 1,) * 2)
        hole_full = cv2.dilate((mask > 0).astype(np.uint8), k) > 0
        forbid_full = hole_full.copy()
        if avoid is not None:
            forbid_full |= cv2.dilate((avoid > 0).astype(np.uint8), k) > 0

        ws = self.work_side if hole_full.mean() <= 0.04 else int(0.7 * self.work_side)   # big holes: coarser, faster
        s = min(1.0, ws / max(h, w))
        if s < 1.0:
            sw, sh = max(16, int(w * s)), max(16, int(h * s))
            small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
            hole = cv2.resize(hole_full.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
            hole = cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            forbid = cv2.resize(forbid_full.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
            forbid |= hole
        else:
            small, hole, forbid = img_bgr.copy(), hole_full, forbid_full

        filled = small.copy()
        n, lab = cv2.connectedComponents(hole.astype(np.uint8))
        for i in range(1, n):
            comp = lab == i
            if comp.sum() < 4:
                continue
            filled = self._fill_component(filled, comp, forbid | hole)
        # anything the exemplar pass could not reach
        left = hole & (np.abs(filled.astype(np.int16) - small.astype(np.int16)).sum(2) == 0)
        if left.any():
            filled = cv2.inpaint(filled, left.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)

        if s < 1.0:
            up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            up = filled
        alpha = cv2.GaussianBlur(hole_full.astype(np.float32), (0, 0), 1.5)[..., None]
        alpha = np.where(hole_full[..., None], np.maximum(alpha, 0.999), alpha)
        out = img_bgr.astype(np.float32) * (1 - alpha) + up.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ one hole
    def _fill_component(self, img_bgr, comp, forbid_all):
        h, w = comp.shape
        ys, xs = np.nonzero(comp)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        m = int(max(36, 0.9 * max(bw, bh)))
        x0, x1 = max(0, xs.min() - m), min(w, xs.max() + 1 + m)
        y0, y1 = max(0, ys.min() - m), min(h, ys.max() + 1 + m)
        roi = img_bgr[y0:y1, x0:x1]
        hole = comp[y0:y1, x0:x1]
        forbid = forbid_all[y0:y1, x0:x1]
        if comp.sum() > 0.12 * h * w or self._ring_is_smooth(roi, hole, forbid):   # huge holes: fast diffusion only
            res = self._diffuse(roi, hole)
        else:
            res = self._criminisi(roi, hole, forbid)
        out = img_bgr.copy()
        out[y0:y1, x0:x1][hole] = res[hole]
        return out

    @staticmethod
    def _ring_is_smooth(roi, hole, forbid, thr=2.3):
        """True when the clean pixels around the hole carry almost no texture (sky, water, walls)."""
        ring = (cv2.dilate(hole.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~forbid
        if ring.sum() < 50:
            return False
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
        return float(hp[ring].std()) < thr

    @staticmethod
    def _diffuse(roi, hole):
        """Diffusion fill + a little grain measured from the surroundings."""
        hm = hole.astype(np.uint8) * 255
        out = cv2.inpaint(roi, hm, 6, cv2.INPAINT_NS)
        ring = (cv2.dilate(hm, np.ones((31, 31), np.uint8)) > 0) & ~hole
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sigma = float((g - cv2.GaussianBlur(g, (0, 0), 2.0))[ring].std()) if ring.sum() > 50 else 1.5
        noise = np.random.default_rng(0).normal(0, sigma, out.shape).astype(np.float32)
        a = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2.0)[..., None]
        return np.clip(out.astype(np.float32) + noise * a, 0, 255).astype(np.uint8)

    @staticmethod
    def _chroma_cost(img, tmpl, known_t, Hp, Wp, ps):
        """Cheap colour term: mean (a,b) of the known part of the template vs. every window."""
        w = known_t[..., None]
        ab_t = (tmpl[..., 1:] * w).sum((0, 1)) / max(float(known_t.sum()), 1.0)
        mean_ab = cv2.boxFilter(img[..., 1:], -1, (ps, ps), normalize=True,
                                anchor=(0, 0), borderType=cv2.BORDER_REFLECT)[:Hp - ps + 1, :Wp - ps + 1]
        return ((mean_ab - ab_t) ** 2).sum(2)

    def _criminisi(self, roi_bgr, hole, forbid):
        ps = self.patch
        r = ps // 2
        H, W = hole.shape
        if min(H, W) < ps + 2:
            return cv2.inpaint(roi_bgr, hole.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)

        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        pad = lambda a, v=0, mode=cv2.BORDER_CONSTANT: cv2.copyMakeBorder(a, r, r, r, r, mode, value=v)
        img = pad(lab, mode=cv2.BORDER_REFLECT)
        unknown = pad(hole.astype(np.uint8), mode=cv2.BORDER_REFLECT) > 0   # reflected hole stays unknown
        conf = (~unknown).astype(np.float32)
        bad = pad(forbid.astype(np.uint8), 1) > 0                # padding can never be a source
        Hp, Wp = img.shape[:2]
        bad_cnt = cv2.boxFilter(bad.astype(np.float32), -1, (ps, ps), normalize=False,
                                anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)[:Hp - ps + 1, :Wp - ps + 1]
        invalid = bad_cnt > 0
        if invalid.all():                                         # nowhere to copy from
            return cv2.inpaint(roi_bgr, hole.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        gy_idx, gx_idx = np.mgrid[0:Hp - ps + 1, 0:Wp - ps + 1].astype(np.float32)
        diag2 = float(H * H + W * W)
        k3 = np.ones((3, 3), np.uint8)

        for _ in range(self.max_iter):
            if not unknown[r:Hp - r, r:Wp - r].any():
                break
            known_u8 = (~unknown).astype(np.uint8)
            lum = np.ascontiguousarray(img[..., 0])
            front = unknown & (cv2.dilate(known_u8, k3) > 0)
            front[:r], front[Hp - r:], front[:, :r], front[:, Wp - r:] = False, False, False, False   # centres stay inside
            fy, fx = np.nonzero(front)
            if len(fy) == 0:
                break
            C = cv2.boxFilter(conf, -1, (ps, ps), normalize=True)
            gray = img[..., 0]                                    # L channel
            gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
            nx = cv2.Scharr(cv2.GaussianBlur(known_u8.astype(np.float32), (0, 0), 1.2), cv2.CV_32F, 1, 0)
            ny = cv2.Scharr(cv2.GaussianBlur(known_u8.astype(np.float32), (0, 0), 1.2), cv2.CV_32F, 0, 1)
            nn = np.sqrt(nx * nx + ny * ny) + 1e-6
            D = np.abs(-gy * nx + gx * ny) / nn / 255.0
            P = C[fy, fx] * (D[fy, fx] + 0.02)
            j = int(np.argmax(P))
            py, px = int(fy[j]), int(fx[j])

            ty0, tx0 = py - r, px - r
            tmpl = img[ty0:ty0 + ps, tx0:tx0 + ps]
            known_t = (~unknown[ty0:ty0 + ps, tx0:tx0 + ps]).astype(np.float32)
            score = cv2.matchTemplate(lum, np.ascontiguousarray(tmpl[..., 0]), cv2.TM_SQDIFF, mask=known_t)
            score = score / max(float(known_t.sum()), 1.0)         # mean squared difference (luminance)
            score += 0.25 * self._chroma_cost(img, tmpl, known_t, Hp, Wp, ps)
            d2 = ((gy_idx - ty0) ** 2 + (gx_idx - tx0) ** 2) / diag2
            score = score + self.locality * d2
            score[invalid] = np.inf
            by, bx = np.unravel_index(int(np.argmin(score)), score.shape)
            if not np.isfinite(score[by, bx]):
                break

            unk_t = unknown[ty0:ty0 + ps, tx0:tx0 + ps]
            src = img[by:by + ps, bx:bx + ps]
            tgt = img[ty0:ty0 + ps, tx0:tx0 + ps]
            tgt[unk_t] = src[unk_t]
            conf[ty0:ty0 + ps, tx0:tx0 + ps][unk_t] = C[py, px]
            unknown[ty0:ty0 + ps, tx0:tx0 + ps] = False

        out_lab = img[r:r + H, r:r + W]
        out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        left = unknown[r:r + H, r:r + W] & hole
        if left.any():
            out = cv2.inpaint(out, left.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        return out
