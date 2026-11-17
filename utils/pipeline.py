"""End-to-end analysis: detect -> mask -> overlay/arrows -> cut-out -> inpaint -> scores -> panel."""
import json
import os
import time
import zlib

import cv2
import numpy as np

from models.inpainting_module import Inpainter
from utils.ela import ela_gray
from utils.forensics import _centroid, border_cues, detect_copy_move
from utils.screenshot_guard import assess, find_picture_region, global_ui_score
from utils.visualize import draw_overlay, draw_overlay_undetermined, make_cutout, render_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights", "forensic_net.pt")                # optional PyTorch network
TYPE_WEIGHTS = os.path.join(ROOT, "weights", "feature_model.joblib")       # optional scikit-learn tamper-type model
LABEL_FORGED, LABEL_CLEAN, LABEL_UI = "COPY-MOVE FORGERY", "AUTHENTIC", "INCONCLUSIVE"
CNN_CLASSES = ["authentic", "copy-move", "splicing", "removal", "retouch"]
CLASS_LABELS = {1: "COPY-MOVE FORGERY", 2: "SPLICING", 3: "OBJECT REMOVAL", 4: "RETOUCHING"}
AMBIGUITY_GAP = 0.75     # score gap between the two best "original" candidates below which we say "cannot tell"
MAX_REGION_FRAC = 0.25   # a single "clone" larger than this share of the image is not a pasted patch
MAX_TOTAL_FRAC = 0.30    # ... nor are copies that together cover this much (side-by-side figures, collages)
PAGE_UI_SCORE = 0.70     # screenshot-like inputs (web page / PDF / slide): duplicates there are usually figures or UI
MIN_AREA_FRAC = 0.0035   # a duplicate smaller than this share of the image is probably a repeated icon / pattern tile
CROP_PICTURES = True     # screenshots: analyse the picture inside the page, not the browser / app around it
CNN_THRESH = 0.65        # network "tampered" probability needed to report a forgery the classical detector missed


# --------------------------------------------------------------------------- source / copy roles
def merge_regions(det):
    """Distinct regions found by all clone models: [[centroid, mask], ...]."""
    regions = []
    for m in det.models:
        for mk in (m.mask_a, m.mask_b):
            c = _centroid(mk)
            rad = 0.6 * np.sqrt((mk > 0).sum())
            for r in regions:
                if np.hypot(r[0][0] - c[0], r[0][1] - c[1]) < rad:
                    r[1] = np.maximum(r[1], mk)
                    break
            else:
                regions.append([c, mk.copy()])
    return regions


def assign_roles(img, det, pick=0):
    """Decide which region is the genuine original (its role is 'source'); all others are the copies.

    Regions are ranked by how well they blend into their surroundings (best first).
    pick=0 -> the best-blending region is the original (automatic decision);
    pick=1 -> the next one, ...  so the user can step through the alternatives.
    Returns (target_mask, source_mask, arrows, n_regions, gap, regions, source_index).
    gap = score difference between the two best candidates (small -> cannot tell which is the original)."""
    h, w = img.shape[:2]
    regions = merge_regions(det)
    scores = []
    for _, mk in regions:
        cross, ctx = border_cues(img, mk)
        scores.append(cross + 0.3 * ctx)
    order = np.argsort(scores)
    src_i = int(order[pick % len(order)])
    tgt = np.zeros((h, w), np.uint8)
    arrows = []
    for i, (c, mk) in enumerate(regions):
        if i != src_i:
            tgt = np.maximum(tgt, mk)
            arrows.append((regions[src_i][0], c))
    src = cv2.bitwise_and(regions[src_i][1], cv2.bitwise_not(tgt))
    gap = float(scores[order[1]] - scores[order[0]]) if len(order) > 1 else 99.0
    return tgt, src, arrows, len(regions), gap, regions, src_i


def heat_to_mask(heat, min_frac=0.0015, max_frac=0.5, keep=3):
    """Network heat-map -> clean binary mask (largest few blobs). Empty if the result is implausible."""
    h, w = heat.shape
    m = (heat > 0.5).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    order = [i for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1 if stats[i, cv2.CC_STAT_AREA] >= max(60, min_frac * h * w)]
    out = np.zeros((h, w), np.uint8)
    for i in order[:keep]:
        out[lab == i] = 255
    return out if 0 < (out > 0).mean() <= max_frac else np.zeros((h, w), np.uint8)


def _legend(img):
    """Small colour key in the bottom-left corner of the overlay."""
    out = img.copy()
    h, w = out.shape[:2]
    sc = max(0.45, min(1.0, w / 1400.0))
    th = max(1, int(round(sc * 1.6)))
    rows = [((40, 40, 235), "red: suspected copy (removed in restored image)"),
            ((40, 190, 250), "yellow: original / source"),
            ((30, 215, 255), "arrow: source -> copy")]
    (tw, lh), _ = cv2.getTextSize(rows[0][1], cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc * 1.6, th)
    pad = int(8 * sc * 1.6)
    box_w, box_h = tw + 4 * pad + int(14 * sc * 1.6), len(rows) * (lh + pad) + pad
    x0, y0 = pad, h - box_h - pad
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (60, 60, 60), 1)
    for i, (col, txt) in enumerate(rows):
        y = y0 + pad + i * (lh + pad) + lh
        cv2.rectangle(out, (x0 + pad, y - lh), (x0 + pad + int(12 * sc * 1.6), y), col, -1)
        cv2.putText(out, txt, (x0 + 3 * pad + int(8 * sc * 1.6), y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * sc * 1.6, (20, 20, 20), th, cv2.LINE_AA)
    return out


class ForensicsPipeline:
    def __init__(self, weights=WEIGHTS, max_side=1280, type_weights=TYPE_WEIGHTS):
        self.max_side = max_side
        self.inpainter = Inpainter()
        self.net, self.device, self.cnn_error = None, "cpu", None
        self._det_cache = (None, None)               # (key, Detection) -> re-clicking "another source" is cheaper
        if os.path.isfile(weights):
            try:
                import torch
                from models.multi_stream_net import load_model
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.net = load_model(weights, self.device)
            except Exception as e:                       # torch missing / bad checkpoint
                self.cnn_error = str(e)
        self.type_model = None
        if self.net is None and os.path.isfile(type_weights):
            try:
                from utils.type_model import FeatureTypeModel
                self.type_model = FeatureTypeModel(type_weights)
            except Exception as e:                       # scikit-learn missing / bad file
                self.cnn_error = str(e)

    # ------------------------------------------------------------------
    def _detect(self, img):
        key = (img.shape, zlib.crc32(img.tobytes()))
        if self._det_cache[0] == key:
            return self._det_cache[1]
        det = detect_copy_move(img)
        self._det_cache = (key, det)
        return det

    def analyze(self, img_bgr, out_dir=None, name="result", pick=0):
        t0 = time.time()
        oh, ow = img_bgr.shape[:2]
        img = img_bgr
        if max(oh, ow) > self.max_side:
            s = self.max_side / max(oh, ow)
            img = cv2.resize(img_bgr, (int(ow * s), int(oh * s)), interpolation=cv2.INTER_AREA)
        crop_box = None
        if CROP_PICTURES and global_ui_score(img)[0] >= 0.25:       # a screenshot? analyse the picture inside it
            box = find_picture_region(img)
            if box:
                x0, y0, bw, bh = box
                img, crop_box = img[y0:y0 + bh, x0:x0 + bw], box
        h, w = img.shape[:2]

        det = self._detect(img)
        det_forged = det.forged
        n_regions, gap, regions, src_i = 0, 99.0, [], -1
        tiny = False
        if det_forged:
            tmask, smask, arrows, n_regions, gap, regions, src_i = assign_roles(img, det, pick)
            if float((tmask > 0).mean()) < MIN_AREA_FRAC:               # a tiny repeat (icon, logo, tile) is not a forgery
                tiny, det_forged = True, False
                tmask, smask, arrows, n_regions, regions, src_i = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8), [], 0, [], -1
        else:
            tmask = np.zeros((h, w), np.uint8)
            smask, arrows = tmask.copy(), []
        forged = det_forged and bool(tmask.any())

        cnn, thresh = None, CNN_THRESH                      # learned opinion (CNN or feature model), only if trained
        if self.net is not None:
            from models.multi_stream_net import predict
            probs, heat = predict(self.net, img, self.device)
            cnn = {"probs": probs, "heat": heat, "tamper": float(1.0 - probs[0])}
        elif self.type_model is not None:
            probs, heat = self.type_model.predict(img)
            cnn = {"probs": probs, "heat": heat, "tamper": float(1.0 - probs[0])}
            thresh = self.type_model.threshold

        # ---- network-only detection: tamper types the classical detector cannot see
        cnn_cls, unlocalised = None, False
        if not det_forged and cnn is not None and cnn["tamper"] >= thresh:
            m = heat_to_mask(cnn["heat"])
            if m.any():
                tmask, cnn_cls = m, 1 + int(np.argmax(cnn["probs"][1:]))
            else:
                unlocalised = True
        cnn_only = cnn_cls is not None
        forged = (det_forged and bool(tmask.any())) or cnn_only

        guard = assess(img, tmask, smask, det_forged and bool(tmask.any()))
        ui_repeat = guard["is_ui_repeat"] or (cnn_only and guard["ui_score"] >= 0.50)
        warning, warning_short = guard["warning"], guard["short"]
        if unlocalised:
            warning = "The network suspects tampering but could not localise it. Treat as inconclusive."
            warning_short = "Suspected, not localised."

        # ---- sanity checks on a copy-move detection
        fracs = [float((mk > 0).mean()) for _, mk in regions]

        def spans_image(mk):              # a figure panel spans (almost) the whole height or width of the picture
            ys, xs = np.nonzero(mk)
            return (ys.max() - ys.min() + 1 >= 0.9 * h or xs.max() - xs.min() + 1 >= 0.9 * w) and (mk > 0).mean() >= 0.15
        collage = bool(det_forged and bool(tmask.any()) and not cnn_only and not ui_repeat and
                       (max(fracs, default=0.0) > MAX_REGION_FRAC or float((tmask > 0).mean()) > MAX_TOTAL_FRAC
                        or any(spans_image(mk) for _, mk in regions)))
        cls_hit = bool(det_forged and bool(tmask.any()) and not cnn_only and not ui_repeat)
        page_fig = bool(cls_hit and not collage and crop_box is None and guard["ui_score"] >= PAGE_UI_SCORE)
        inconclusive_det = collage or page_fig
        ambiguous = bool(cls_hit and not inconclusive_det and n_regions >= 2 and gap < AMBIGUITY_GAP)
        order_x = sorted(range(len(regions)), key=lambda i: np.nonzero(regions[i][1])[1].mean()) if regions else []
        letter = {i: chr(65 + k) for k, i in enumerate(order_x)}
        if collage:
            warning = ("Two large, almost identical areas were found (about %.0f%% of the image). That usually means a "
                       "side-by-side figure, collage or before/after pair - not a pasted patch. Crop a single picture "
                       "and run it again." % (100 * max(float((tmask > 0).mean()), max(fracs, default=0.0))))
            warning_short = "Large duplicated areas: collage or figure?"
        elif page_fig:
            warning = ("This looks like a screenshot of a web page, document or slide. Duplicated areas there are usually "
                       "repeated figure panels or interface elements, not a pasted patch. Crop the picture itself and "
                       "run it again.")
            warning_short = "Screenshot of a page/figure: crop the picture."
        elif ambiguous:
            removed = ", ".join(letter[i] for i in range(len(regions)) if i != src_i)
            kept = letter[src_i]
            warning = ("Regions %s are identical copies of each other and the image gives no clue which one is the "
                       "original. The restored picture removes region %s as an ASSUMPTION - if %s is the copy, "
                       "click 'Try another source'." % (", ".join(letter[i] for i in order_x), removed, kept))
            warning_short = "Original vs copy undetermined: region %s removed as an assumption." % removed

        # ---- the four visual outputs
        if det_forged and not cnn_only:
            if ambiguous or collage or page_fig:
                overlay = draw_overlay_undetermined(img, [regions[i][1] for i in order_x])
            else:
                overlay = _legend(draw_overlay(img, tmask, smask, arrows))
        elif cnn_only:
            overlay = draw_overlay(img, tmask, np.zeros_like(tmask), [])
        else:
            overlay = img.copy()
        cutout = make_cutout(img, tmask)
        # repeated UI elements are not "damage": do not paint over them
        if forged and not ui_repeat and not inconclusive_det and (not cnn_only or cnn_cls == 2):
            restored = self.inpainter(img, tmask, smask)         # copy-move copy / spliced object removed
        else:
            restored = img.copy()
            if cnn_only and cnn_cls in (3, 4) and not ui_repeat:
                warning = warning or "Removed or retouched content cannot be recovered - nothing to restore."
                warning_short = warning_short or "Original not recoverable."
        binary = (tmask > 0).astype(np.uint8) * 255

        # ---- scores
        area = float((tmask > 0).sum()) / (h * w)
        tamper_p = cnn["tamper"] if cnn else 0.0
        if forged:
            if cnn_only:
                share = float(cnn["probs"][cnn_cls] / max(cnn["probs"][1:].sum(), 1e-6))
                conf = float(np.clip(tamper_p * (0.6 + 0.4 * share), 0.50, 0.99))
                intensity = float(np.clip(0.5 * min(1.0, area / 0.10) + 0.5 * tamper_p, 0, 1))
                label = CLASS_LABELS[cnn_cls]
            else:
                c_cls = 1 - np.exp(-det.n_inliers / 8.0)
                conf = 0.75 * c_cls + 0.25 * tamper_p if cnn else c_cls
                conf = float(np.clip(conf, 0.50, 0.99))
                ela = ela_gray(img)
                inside, outside = ela[tmask > 0].mean(), ela[tmask == 0].mean() + 1e-3
                ela_term = float(np.clip(np.tanh(1.5 * (inside / outside - 1.0)), 0, 1))
                intensity = float(np.clip(0.4 * min(1.0, area / 0.10) + 0.3 * ela_term
                                          + 0.3 * (1 - np.exp(-det.n_inliers / 15.0)), 0, 1))
                label = LABEL_FORGED
            if ui_repeat or inconclusive_det:
                label, conf, intensity = LABEL_UI, 0.40, 0.0
        elif unlocalised:
            conf = float(np.clip(1.0 - tamper_p, 0.3, 0.5))
            intensity, label = 0.0, LABEL_UI
        else:
            conf = float(np.clip(0.97 - 0.30 * tamper_p - 0.004 * det.n_matches, 0.50, 0.99))
            intensity, label = 0.0, LABEL_CLEAN
            if guard["ui_score"] >= 0.50:                  # "no forgery found" means little on a UI screenshot
                conf = min(conf, 0.60)

        info = {
            "label": label, "confidence": round(conf, 3), "intensity": round(intensity, 3),
            "area_pct": round(100 * area, 1), "regions_matched": len(det.models),
            "n_regions": n_regions if not cnn_only else 0, "pick": (pick % n_regions) if (n_regions and not cnn_only) else 0,
            "resolution": f"{ow}x{oh}", "inliers": det.n_inliers, "keypoints": det.n_keypoints,
            "matches": det.n_matches,
            "cnn_probability": None if cnn is None else round(tamper_p, 3),
            "cnn_probs": None if cnn is None else {c: round(float(p), 3) for c, p in zip(CNN_CLASSES, cnn["probs"])},
            "cnn_used": self.net is not None or self.type_model is not None, "type_model": "cnn" if self.net is not None else ("feature" if self.type_model is not None else None), "ambiguous": ambiguous, "collage": collage, "page_figure": page_fig, "tiny_repeat_ignored": tiny, "cropped": crop_box is not None, "crop_box": list(crop_box) if crop_box else None, "detected_by": "network" if cnn_only else ("classical" if det_forged else "none"),
            "ui_score": guard["ui_score"],
            "warning": warning, "warning_short": warning_short, "seconds": round(time.time() - t0, 2),
        }
        panel = render_panel(img, overlay, cutout, restored, info)
        res = {"original": img, "overlay": overlay, "cutout": cutout, "restored": restored,
               "mask": binary, "panel": panel, "info": info}
        if out_dir:
            self.save(res, out_dir, name)
        return res

    @staticmethod
    def save(res, out_dir, name):
        os.makedirs(out_dir, exist_ok=True)
        for k in ("original", "overlay", "cutout", "restored", "mask", "panel"):
            cv2.imwrite(os.path.join(out_dir, f"{name}_{k}.png"), res[k])
        with open(os.path.join(out_dir, f"{name}_report.json"), "w") as f:
            json.dump(res["info"], f, indent=2)
