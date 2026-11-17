"""Benchmark the detector on a real dataset (or on your own images) and report honest numbers.

It measures, per image and overall:
  * image level : detection rate on tampered images, false-alarm rate on untampered ones,
                  precision / recall / F1 / accuracy, AUC, how often the answer is INCONCLUSIVE
  * pixel level : IoU and F1 of the predicted mask against the ground-truth mask, split by
                  - AUTO     (what the app shows by default)
                  - ORACLE   (best of the 'try another source' choices -> separates localisation errors
                              from 'picked the wrong region as the copy' errors)
                  - UNION    (both clone regions together, for datasets whose masks mark source + copy)
  * a breakdown by tampered-area size, the time per image, and the worst failures as pictures.

----------------------------------------------------------------------------------------------
QUICK START
  python benchmark.py --self-test                          # proves the script works (makes its own images)

  # false alarms only: a folder of normal, untampered photos
  python benchmark.py --authentic D:\\photos\\normal

  # a folder converted to our layout:  DIR/images/NAME.jpg  DIR/masks/NAME.png  (no mask = untampered)
  python benchmark.py --root D:\\data\\my_set

  # any layout: tell it where the images are and how the mask file is named
  python benchmark.py --images D:\\data\\set\\forged --masks D:\\data\\set\\masks --mask-suffix _gt
  python benchmark.py --images D:\\data\\comofod --image-glob "*_F.png" --stem-replace _F _B     # CoMoFoD-style names
  python benchmark.py --images D:\\data\\set\\forged --masks D:\\data\\set\\gt --authentic D:\\data\\set\\originals

  ALWAYS check the 'first pairings' lines it prints: each image must be paired with ITS mask.
  Useful extras:  --max 200   --no-network   --save-failures 12   --invert-mask   --out results_folder
----------------------------------------------------------------------------------------------
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import tempfile
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
FORGERY_LABELS = {"COPY-MOVE FORGERY": 1, "SPLICING": 2, "OBJECT REMOVAL": 3, "RETOUCHING": 4}
CLASS_NAMES = ["authentic", "copy-move", "splicing", "removal", "retouch"]


# --------------------------------------------------------------------------- dataset discovery
def list_images(folder, pattern="*"):
    files = []
    for p in sorted(glob.glob(os.path.join(folder, pattern))):
        if os.path.isfile(p) and p.lower().endswith(IMG_EXT):
            files.append(p)
    return files


def find_mask(mask_dirs, stem, suffix):
    for d in mask_dirs:
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            p = os.path.join(d, stem + suffix + ext)
            if os.path.isfile(p):
                return p
    return None


def discover(a):
    """Returns a list of dicts: name, image, mask (or None), kind ('tampered' | 'authentic'), cls."""
    items, missing = [], 0
    labels = {}
    if a.root:
        img_dir, msk_dir = os.path.join(a.root, "images"), os.path.join(a.root, "masks")
        lp = os.path.join(a.root, "labels.csv")
        if os.path.isfile(lp):
            with open(lp, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("cls", "") != "":
                        labels[r["name"]] = int(r["cls"])
        images, masks_dirs = list_images(img_dir, a.image_glob), [msk_dir]
    elif a.images:
        images = list_images(a.images, a.image_glob)
        masks_dirs = [a.masks or a.images]
    else:
        images, masks_dirs = [], []
    for p in images:
        stem = os.path.splitext(os.path.basename(p))[0]
        mstem = stem.replace(a.stem_replace[0], a.stem_replace[1]) if a.stem_replace else stem
        m = find_mask(masks_dirs, mstem, a.mask_suffix)
        if m is None:
            if a.root or a.missing_mask == "authentic":
                items.append(dict(name=stem, image=p, mask=None, kind="authentic", cls=0))
            else:
                missing += 1
            continue
        items.append(dict(name=stem, image=p, mask=m, kind="tampered", cls=labels.get(stem, -1)))
    if a.authentic:
        for p in list_images(a.authentic):
            items.append(dict(name="AUTH_" + os.path.splitext(os.path.basename(p))[0], image=p, mask=None, kind="authentic", cls=0))
    if missing:
        print(f"[warning] {missing} images had no matching mask and were skipped "
              f"(use --missing-mask authentic to treat them as untampered)")
    return items


# --------------------------------------------------------------------------- metrics
def mask_metrics(pred, gt):
    pred, gt = pred > 0, gt > 0
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    tp, fp, fn = inter, int((pred & ~gt).sum()), int((~pred & gt).sum())
    iou = inter / union if union else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return iou, f1


def to_gt_size(mask, shape):
    if mask.shape[:2] != shape[:2]:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs, i = np.asarray(x)[order], 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    r = ranks(s)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def score_of(info):
    """Continuous 'tampered' score for the AUC: confidence for forgeries, 1-confidence for AUTHENTIC."""
    lab, c = info["label"], info["confidence"]
    if lab in FORGERY_LABELS:
        return c
    return 0.5 if lab == "INCONCLUSIVE" else 1.0 - c


# --------------------------------------------------------------------------- main run
class NoInpaint:
    """The benchmark does not need restored images, so skip that (slow) step."""
    def __call__(self, img, mask, avoid=None):
        return img.copy()


def run(items, a):
    import utils.pipeline as PL
    PL.CROP_PICTURES = False                       # benchmark images are plain pictures, not screenshots
    pipe = PL.ForensicsPipeline()
    if a.no_network:
        pipe.net, pipe.type_model = None, None
    pipe.inpainter = NoInpaint()
    used = "CNN" if pipe.net is not None else ("feature model" if pipe.type_model is not None else "classical detector only")
    print(f"\nPipeline: {used}\n")

    rows, fails = [], []
    t_start = time.time()
    for k, it in enumerate(items):
        img = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] cannot read {it['image']}")
            continue
        t0 = time.time()
        res = pipe.analyze(img, pick=0)
        info = res["info"]
        row = dict(name=it["name"], kind=it["kind"], gt_cls=it["cls"], label=info["label"], confidence=info["confidence"],
                   score=score_of(info), flagged=int(info["label"] in FORGERY_LABELS), inconclusive=int(info["label"] == "INCONCLUSIVE"),
                   pred_area_pct=info["area_pct"], n_regions=info["n_regions"], gt_area_pct=0.0,
                   iou_auto=np.nan, iou_oracle=np.nan, iou_union=np.nan, f1_auto=np.nan, f1_oracle=np.nan)
        if it["kind"] == "tampered":
            gt = cv2.imread(it["mask"], cv2.IMREAD_GRAYSCALE)
            gt = (gt > 127).astype(np.uint8)
            if a.invert_mask:
                gt = 1 - gt
            gt = to_gt_size(gt, img.shape)
            row["gt_area_pct"] = round(100 * float(gt.mean()), 2)
            if gt.mean() > 0.5:
                row["warn"] = "mask covers >50% of image (inverted?)"
            masks = [to_gt_size(res["mask"], img.shape) > 0]
            for pk in range(1, min(info["n_regions"], 5)):            # the other 'try another source' choices
                masks.append(to_gt_size(pipe.analyze(img, pick=pk)["mask"], img.shape) > 0)
            m_iou = [mask_metrics(m, gt) for m in masks]
            row["iou_auto"], row["f1_auto"] = m_iou[0]
            best = max(m_iou, key=lambda t: t[0])
            row["iou_oracle"], row["f1_oracle"] = best
            row["iou_union"] = mask_metrics(np.any(masks, axis=0), gt)[0]
            if row["iou_oracle"] < 0.5 and len(fails) < 300:
                th = 260 / img.shape[1]
                small = lambda x: cv2.resize(x, (260, max(1, int(img.shape[0] * th))))
                g = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                ov = to_gt_size(res["overlay"], img.shape) if res["overlay"].shape[:2] != img.shape[:2] else res["overlay"]
                fails.append((row["iou_oracle"], it["name"], np.hstack([small(img), small(ov), small(g)])))
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        if (k + 1) % 20 == 0 or k + 1 == len(items):
            print(f"  {k + 1}/{len(items)} images   ({(time.time() - t_start) / (k + 1):.1f}s per image)", flush=True)
    return rows, fails, used


def pct(x):
    return "n/a" if x != x else f"{100 * x:5.1f}%"


def summarize(rows, a):
    tp_rows = [r for r in rows if r["kind"] == "tampered"]
    au_rows = [r for r in rows if r["kind"] == "authentic"]
    S = {"n_tampered": len(tp_rows), "n_authentic": len(au_rows)}
    print("\n" + "=" * 66 + "\nRESULTS\n" + "=" * 66)
    if tp_rows:
        S["detection_rate"] = float(np.mean([r["flagged"] for r in tp_rows]))
        S["tampered_inconclusive"] = float(np.mean([r["inconclusive"] for r in tp_rows]))
        print(f"Tampered images: {len(tp_rows)}   detected {pct(S['detection_rate'])}   inconclusive {pct(S['tampered_inconclusive'])}"
              f"   missed {pct(1 - S['detection_rate'] - S['tampered_inconclusive'])}")
    if au_rows:
        S["false_alarm_rate"] = float(np.mean([r["flagged"] for r in au_rows]))
        S["authentic_inconclusive"] = float(np.mean([r["inconclusive"] for r in au_rows]))
        print(f"Untampered images: {len(au_rows)}   false alarms {pct(S['false_alarm_rate'])}   inconclusive {pct(S['authentic_inconclusive'])}")
    if tp_rows and au_rows:
        tp = sum(r["flagged"] for r in tp_rows)
        fp = sum(r["flagged"] for r in au_rows)
        fn = len(tp_rows) - tp
        tn = len(au_rows) - fp
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if prec == prec and prec + rec else float("nan")
        S.update(precision=prec, recall=rec, f1=f1, accuracy=(tp + tn) / (len(rows)),
                 auc=auc(np.array([r["score"] for r in tp_rows]), np.array([r["score"] for r in au_rows])))
        print(f"Image level: precision {pct(prec)}  recall {pct(rec)}  F1 {pct(f1)}  accuracy {pct(S['accuracy'])}  AUC {S['auc']:.3f}")
        print("   (INCONCLUSIVE counts as 'not flagged' here)")
    if tp_rows:
        for key, label in (("iou_auto", "AUTO   (what the app shows)"), ("iou_oracle", "ORACLE (best 'try another source')"),
                           ("iou_union", "UNION  (all clone regions together)")):
            v = np.array([r[key] for r in tp_rows])
            det = np.array([r["flagged"] for r in tp_rows], bool)
            S["mean_" + key] = float(np.nanmean(v))
            print(f"Mask IoU {label}: mean {np.nanmean(v):.3f}   median {np.nanmedian(v):.3f}"
                  f"   among detected {np.nanmean(v[det]) if det.any() else float('nan'):.3f}   >=0.5 in {pct(np.mean(v >= 0.5))}")
        f = np.array([r["f1_auto"] for r in tp_rows])
        S["mean_pixel_f1_auto"] = float(np.nanmean(f))
        print(f"Pixel F1 (AUTO): mean {np.nanmean(f):.3f}")
        wrong = [r for r in tp_rows if r["iou_oracle"] > 0.3 and r["iou_auto"] < 0.1]
        S["wrong_side"] = len(wrong)
        print(f"'Picked the wrong region as the copy' (auto IoU<0.1 but a swap fixes it): {len(wrong)} of {len(tp_rows)}")
        print("By tampered-area size (share of the image):")
        for lo, hi, nm in ((0, 1, "small  <1%"), (1, 5, "medium 1-5%"), (5, 101, "large  >5%")):
            sel = [r for r in tp_rows if lo <= r["gt_area_pct"] < hi]
            if sel:
                print(f"   {nm:12s} n={len(sel):4d}  detected {pct(np.mean([r['flagged'] for r in sel]))}   "
                      f"mean IoU auto {np.nanmean([r['iou_auto'] for r in sel]):.3f}  oracle {np.nanmean([r['iou_oracle'] for r in sel]):.3f}")
        warn = [r for r in tp_rows if r.get("warn")]
        if warn:
            print(f"[warning] {len(warn)} masks cover more than half of the image - are the masks inverted? try --invert-mask")
    typed = [r for r in tp_rows if r["gt_cls"] > 0 and r["flagged"]]
    if typed:
        conf = np.zeros((5, 5), int)
        for r in typed:
            conf[r["gt_cls"], FORGERY_LABELS.get(r["label"], 0)] += 1
        print("Type confusion among detected images (rows = truth, columns = predicted):  " + " | ".join(c[:6] for c in CLASS_NAMES))
        for i, c in enumerate(CLASS_NAMES[1:], 1):
            if conf[i].sum():
                print(f"   {c:10s} {conf[i].tolist()}")
    S["seconds_per_image"] = float(np.mean([r["seconds"] for r in rows]))
    print(f"Time: {S['seconds_per_image']:.1f}s per image")
    return S


def save_outputs(rows, fails, S, used, a):
    out = a.out or os.path.join(ROOT, "benchmark_results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "per_image.csv"), "w", newline="") as f:
        keys = ["name", "kind", "gt_cls", "label", "confidence", "flagged", "inconclusive", "gt_area_pct", "pred_area_pct",
                "n_regions", "iou_auto", "iou_oracle", "iou_union", "f1_auto", "f1_oracle", "seconds"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    S = dict(S, pipeline=used, command=" ".join(sys.argv))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(S, f, indent=2, default=lambda o: None if o != o else o)
    if a.save_failures and fails:
        fails.sort(key=lambda t: t[0])
        for iou, name, pic in fails[:a.save_failures]:
            cv2.imwrite(os.path.join(out, f"fail_iou{iou:.2f}_{name[:40]}.png"), pic)
        print(f"Saved the {min(a.save_failures, len(fails))} worst failures (original | overlay | ground truth) to {out}")
    print(f"Saved per-image CSV and summary.json to {out}")


# --------------------------------------------------------------------------- self test
def make_selftest(folder, n_tampered=10, n_authentic=6):
    from data.dataset_generator import blob_mask, make_landscape, paste_region
    os.makedirs(os.path.join(folder, "forged"), exist_ok=True)
    os.makedirs(os.path.join(folder, "gt"), exist_ok=True)
    os.makedirs(os.path.join(folder, "originals"), exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n_tampered):
        base = make_landscape(640, 480, 100 + i)[0]
        h, w = base.shape[:2]
        for _ in range(40):
            r = rng.uniform(0.07, 0.10) * min(h, w)
            sx, sy, tx, ty = rng.uniform(r, w - r), rng.uniform(r, h - r), rng.uniform(r, w - r), rng.uniform(r, h - r)
            if np.hypot(tx - sx, ty - sy) > 2.6 * r:
                break
        src = blob_mask(h, w, sx, sy, r, rng).astype(np.uint8) * 255
        forged, gt = paste_region(base, src, (tx - sx, ty - sy), 0, 1.0, feather=1.2)
        cv2.imwrite(os.path.join(folder, "forged", f"set{i:02d}_F.png"), forged)
        cv2.imwrite(os.path.join(folder, "gt", f"set{i:02d}_B.png"), gt)
    for i in range(n_authentic):
        cv2.imwrite(os.path.join(folder, "originals", f"photo{i:02d}.png"), make_landscape(640, 480, 500 + i)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="folder in our layout: images/ masks/ [labels.csv]")
    ap.add_argument("--images", help="folder with the (tampered) images")
    ap.add_argument("--masks", help="folder with the ground-truth masks (default: same as --images)")
    ap.add_argument("--image-glob", default="*", help='which image files to use, e.g. "*_F.png"')
    ap.add_argument("--mask-suffix", default="", help="mask file = image name + this suffix, e.g. _gt")
    ap.add_argument("--stem-replace", nargs=2, metavar=("OLD", "NEW"), help="mask name = image name with OLD replaced by NEW")
    ap.add_argument("--authentic", help="folder of untampered images (for the false-alarm rate)")
    ap.add_argument("--missing-mask", choices=["skip", "authentic"], default="skip")
    ap.add_argument("--invert-mask", action="store_true")
    ap.add_argument("--max", type=int, help="use at most this many images (random sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-network", action="store_true", help="ignore any trained model, test the classical detector only")
    ap.add_argument("--save-failures", type=int, default=10, help="save pictures of the N worst failures")
    ap.add_argument("--out", help="folder for the CSV / JSON / failure pictures")
    ap.add_argument("--self-test", action="store_true", help="generate a small dataset and run on it")
    a = ap.parse_args()

    tmp = None
    if a.self_test:
        tmp = tempfile.mkdtemp(prefix="bench_selftest_")
        make_selftest(tmp)
        a.images, a.masks, a.image_glob, a.stem_replace = os.path.join(tmp, "forged"), os.path.join(tmp, "gt"), "*_F.png", ["_F", "_B"]
        a.authentic = os.path.join(tmp, "originals")
        print(f"[self-test] made 10 forged + 6 untampered images in {tmp}")
    if not (a.root or a.images or a.authentic):
        sys.exit("Give --root DIR, or --images DIR, or --authentic DIR (see the examples at the top of this file), or --self-test.")
    items = discover(a)
    if not items:
        sys.exit("No images found. Check the folder, --image-glob and the mask naming options.")
    if a.max and len(items) > a.max:
        random.Random(a.seed).shuffle(items)
        items = items[:a.max]
    print(f"Dataset: {sum(i['kind'] == 'tampered' for i in items)} tampered + {sum(i['kind'] == 'authentic' for i in items)} untampered images")
    print("First pairings (CHECK that each image goes with its own mask):")
    for it in [i for i in items if i["kind"] == "tampered"][:5]:
        print(f"   {os.path.basename(it['image'])}   <->   {os.path.basename(it['mask'])}")
    rows, fails, used = run(items, a)
    S = summarize(rows, a)
    save_outputs(rows, fails, S, used, a)


if __name__ == "__main__":
    main()