"""Train the multi-type tampering model (splicing / removal / retouch / authentic) WITHOUT PyTorch.

    python train_features.py                         # synthetic data only, ~10 minutes on a laptop CPU
    python train_features.py --n 6000 --val 800      # more data
    python train_features.py --real D:\\data\\casia_converted --real D:\\data\\columbia_converted

Real datasets: folder with  images/NAME.jpg   masks/NAME.png (white = tampered, missing = authentic)
and an optional labels.csv (name,cls with cls 0=authentic 2=splicing 3=removal 4=retouch). Without
labels.csv every tampered real image is treated as splicing.

Output: weights/feature_model.joblib   (the app picks it up automatically)
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "weights", "feature_model.joblib")
CLASS_NAMES = {0: "authentic", 2: "splicing", 3: "removal", 4: "retouch"}


def sample_rows(img, mask, cls, from_real=False):
    """Patch features + patch labels for one image. Patches half-covered by an edit are skipped."""
    from utils.type_features import patch_cover, patch_features
    F, grid, prep = patch_features(img)
    cov = patch_cover(mask, prep.shape)
    if cls <= 0:
        y = np.zeros(len(F), int)
        keep = cov == 0
    else:
        y = np.where(cov >= 0.35, cls, 0)
        keep = (cov >= 0.35) | (cov == 0)
        if cls == 1:                                       # copy-move: patch statistics are identical, skip the edit
            keep = cov == 0
    return F[keep], y[keep]


def collect_synthetic(n, seed, cache=None, keep_images=False, max_seconds=None, chunk=250):
    """Synthetic training rows, generated in chunks of `chunk` images.
    With `cache` (a folder) every finished chunk is saved, so an interrupted run resumes where it stopped."""
    from data.dataset_generator import make_sample
    X, Y, imgs, t0 = [], [], [], time.time()
    if cache:
        os.makedirs(cache, exist_ok=True)
    for k in range((n + chunk - 1) // chunk):
        count = min(chunk, n - k * chunk)
        path = os.path.join(cache, f"syn_s{seed}_c{k}_n{count}.npz") if cache else None
        if path and os.path.isfile(path) and not keep_images:
            d = np.load(path)
            X.append(d["X"]), Y.append(d["Y"])
            continue
        if max_seconds is not None and time.time() - t0 > max_seconds:
            print(f"    time budget reached after {k * chunk} images - run the same command again to continue", flush=True)
            return None, None, None
        rng = np.random.default_rng([seed, k])
        cx, cy = [], []
        for _ in range(count):
            img, mask, cls = make_sample(rng, 256)
            f, y = sample_rows(img, mask, cls)
            cx.append(f), cy.append(y)
            if keep_images:
                imgs.append((img, mask, cls))
        cx, cy = np.concatenate(cx), np.concatenate(cy)
        if path:
            np.savez_compressed(path, X=cx, Y=cy)
        X.append(cx), Y.append(cy)
        print(f"    {min((k + 1) * chunk, n)}/{n} images", flush=True)
    return np.concatenate(X), np.concatenate(Y), imgs


def load_real(folder, max_images=4000):
    labels = {}
    lp = os.path.join(folder, "labels.csv")
    if os.path.isfile(lp):
        with open(lp, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("cls", "") != "":
                    labels[r["name"]] = int(r["cls"])
    out = []
    idir, mdir = os.path.join(folder, "images"), os.path.join(folder, "masks")
    for fn in sorted(os.listdir(idir))[:max_images]:
        name, ext = os.path.splitext(fn)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
            continue
        img = cv2.imread(os.path.join(idir, fn))
        mp = os.path.join(mdir, name + ".png")
        mask = cv2.imread(mp, 0) if os.path.isfile(mp) else None
        if img is None:
            continue
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        elif mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cls = labels.get(name, 2 if (mask > 127).any() else 0)
        if cls == 1:
            cls = 2
        out.append((img, mask, cls))
    return out


def evaluate(model, items, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Image-level evaluation with the exact aggregation the app uses."""
    from utils.pipeline import heat_to_mask
    rows = []
    for img, mask, cls in items:
        if cls == 1:
            continue
        probs, heat = model.predict(img)
        rows.append((cls, 1 - probs[0], int(np.argmax(probs[1:])) + 1, heat, mask))
    print(f"  images evaluated: {len(rows)}   (authentic {sum(r[0] == 0 for r in rows)}, tampered {sum(r[0] > 0 for r in rows)})")
    print("  thr   false-alarm   recall   type-correct   mean mask IoU (detected)")
    for thr in thresholds:
        au = [r for r in rows if r[0] == 0]
        tp = [r for r in rows if r[0] > 0]
        fa = np.mean([r[1] >= thr for r in au]) if au else float("nan")
        det = [r for r in tp if r[1] >= thr]
        rec = len(det) / max(len(tp), 1)
        typ = np.mean([r[2] == r[0] for r in det]) if det else float("nan")
        ious = []
        for r in det:
            m = heat_to_mask(r[3]) > 0
            g = r[4] > 127
            if m.shape != g.shape:
                g = cv2.resize(g.astype(np.uint8), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            ious.append((m & g).sum() / max((m | g).sum(), 1))
        print(f"  {thr:.1f}   {fa:9.3f}   {rec:7.3f}   {typ:11.3f}   {np.mean(ious) if ious else float('nan'):.3f}")
    print("  per-type recall @0.7:")
    for cid, name in CLASS_NAMES.items():
        if cid == 0:
            continue
        sel = [r for r in rows if r[0] == cid]
        if sel:
            print(f"     {name:10s} n={len(sel):4d}  detected {np.mean([r[1] >= 0.7 for r in sel]):.2f}   "
                  f"named correctly {np.mean([r[1] >= 0.7 and r[2] == cid for r in sel]):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="synthetic training images")
    ap.add_argument("--val", type=int, default=500, help="synthetic validation images")
    ap.add_argument("--real", action="append", help="real dataset folder (images/ + masks/)")
    ap.add_argument("--threshold", type=float, default=0.72, help="image-level score needed to report a forgery")
    ap.add_argument("--real-weight", type=float, default=3.0, help="importance of one real patch vs one synthetic patch")
    ap.add_argument("--auth-keep", type=float, default=0.3, help="share of authentic patches kept (speed)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "feature_cache"), help="folder for resumable feature chunks")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop collecting after this long (re-run to continue)")
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    from utils.type_model import FeatureTypeModel

    t0 = time.time()
    print(f"[1/4] synthetic training data ({a.n} images)")
    X, Y, _ = collect_synthetic(a.n, seed=1, cache=a.cache, max_seconds=a.max_seconds)
    if X is None:
        return
    keep = (Y > 0) | (np.random.default_rng(0).random(len(Y)) < a.auth_keep)      # most authentic patches are redundant
    X, Y = X[keep], Y[keep]
    W = np.ones(len(Y), np.float32)
    real_val = []
    if a.real:
        for folder in a.real:
            items = load_real(folder)
            k = int(0.85 * len(items))
            print(f"    real data {folder}: {k} train / {len(items) - k} val images")
            for img, mask, cls in items[:k]:
                f, y = sample_rows(img, mask, cls, True)
                X, Y = np.concatenate([X, f]), np.concatenate([Y, y])
                W = np.concatenate([W, np.full(len(y), a.real_weight, np.float32)])
            real_val += items[k:]
    print(f"[2/4] fitting on {len(X)} patches  " + "  ".join(f"{CLASS_NAMES[c]}={int((Y == c).sum())}" for c in CLASS_NAMES))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                         class_weight="balanced", random_state=0)
    clf.fit(X, Y, sample_weight=W)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump({"clf": clf, "threshold": a.threshold, "meta": {"n_train_images": a.n, "real": a.real or []}}, a.out)
    model = FeatureTypeModel(a.out)
    print(f"[3/4] held-out SYNTHETIC validation ({a.val} new images)")
    _, _, val = collect_synthetic(a.val, seed=777, keep_images=True)
    evaluate(model, val)
    if real_val:
        print("[4/4] held-out REAL validation")
        evaluate(model, real_val)
    print(f"saved {a.out}   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
