"""Tamper-type model that needs no PyTorch: patch features (utils/type_features.py) + gradient boosting.

Classes it knows (patch level): authentic | splicing | removal | retouch.
Copy-move is found by the classical detector (utils/forensics.py), so it is not part of this model.

predict(img) -> (probs[5], heat)   same contract as the optional CNN:
    probs = [authentic, copy-move(0), splicing, removal, retouch]
    heat  = per-pixel tamper probability at the size of `img`
"""
import cv2
import numpy as np

from utils.type_features import PATCH, TOPK, centers, patch_features

BLOB_THR, MIN_CELLS = 0.6, 6      # a detection needs a connected cluster of at least ~MIN_CELLS suspicious patches


class FeatureTypeModel:
    def __init__(self, path):
        import joblib
        d = joblib.load(path)
        self.clf = d["clf"]
        self.threshold = float(d.get("threshold", 0.72))      # image-level 'tampered' score needed to report it
        self.meta = d.get("meta", {})

    def patch_probs(self, img):
        F, grid, prep = patch_features(img)
        P = self.clf.predict_proba(F)
        full = np.zeros((len(F), 5), np.float32)
        for col, cid in enumerate(self.clf.classes_):
            full[:, int(cid)] = P[:, col]
        return full, grid, prep

    @staticmethod
    def _best_blob(sm):
        """Strongest connected cluster of suspicious cells: score = mean probability x min(1, cells / MIN_CELLS)."""
        b = (sm >= BLOB_THR).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        best, best_i = 0.0, 0
        for i in range(1, n):
            s = float(sm[lab == i].mean()) * min(1.0, int(st[i, cv2.CC_STAT_AREA]) / MIN_CELLS)
            if s > best:
                best, best_i = s, i
        return best, (lab == best_i) if best_i else None

    def predict(self, img):
        h0, w0 = img.shape[:2]
        full, (ny, nx), prep = self.patch_probs(img)
        tamper = 1.0 - full[:, 0]
        grid = tamper.reshape(ny, nx).astype(np.float32)
        sm = cv2.GaussianBlur(grid, (3, 3), 0.8, borderType=cv2.BORDER_REPLICATE) if min(ny, nx) >= 3 else grid
        score, blob = self._best_blob(sm)
        sel = blob.ravel() if blob is not None else (tamper >= np.sort(tamper)[::-1][min(TOPK, len(tamper)) - 1])
        share = (tamper[sel, None] * full[sel, 1:]).sum(0)
        share = share / max(float(share.sum()), 1e-9)
        probs = np.zeros(5, np.float32)
        probs[0], probs[1:] = 1.0 - score, score * share
        hp, wp = prep.shape[:2]
        ys, xs = centers(hp, wp)
        acc, cnt = np.zeros((hp, wp), np.float32), np.zeros((hp, wp), np.float32)
        k, r = 0, PATCH // 2
        for y in ys:
            for x in xs:
                acc[y - r:y + r, x - r:x + r] += tamper[k]
                cnt[y - r:y + r, x - r:x + r] += 1
                k += 1
        heat = cv2.resize(acc / np.maximum(cnt, 1), (w0, h0), interpolation=cv2.INTER_LINEAR)
        return probs, heat
