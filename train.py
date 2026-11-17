"""Train the multi-stream, multi-class forensic network.

    python train.py --preset quick                    # ~minutes on CPU, smoke test
    python train.py --preset standard                 # recommended (use a GPU)
    python train.py --preset hard                     # long run (GPU)
    python train.py --preset standard --real D:\\data\\casia --real D:\\data\\columbia
    python train.py --eval-dir D:\\data\\some_real_dataset     # honest accuracy of the saved weights

Synthetic samples (copy-move, splicing, removal, retouching, authentic, UI screenshots) are generated
on the fly, so every epoch sees new images. Real datasets (converted to images/ + masks/ folders, see
data/train.py) are mixed in with --real; model selection then uses the REAL validation split.
"""
import argparse
import os
import random
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(ROOT, "weights", "forensic_net.pt")

PRESETS = {
    "quick":    dict(epochs=6,  per_epoch=480,  batch=16, chs="small", lr=2e-3),
    "standard": dict(epochs=30, per_epoch=2400, batch=16, chs="base",  lr=2e-3),
    "hard":     dict(epochs=80, per_epoch=6400, batch=24, chs="base",  lr=1.5e-3),
}


# --------------------------------------------------------------------------- evaluation
def evaluate(net, loader, dev, num_classes=5):
    import torch
    net.eval()
    conf = np.zeros((num_classes, num_classes), int)
    ious, bin_ok, n = [], 0, 0
    tp = fp = fn = 0
    with torch.no_grad():
        for rgb, ela, srm, m, y in loader:
            seg, cl = net(rgb.to(dev), ela.to(dev), srm.to(dev))
            pr = torch.softmax(cl.float(), 1).cpu().numpy()
            pm = (torch.sigmoid(seg.float()) > 0.5).cpu().numpy()[:, 0]
            gm, y = m.numpy()[:, 0] > 0.5, y.numpy()
            for i in range(len(y)):
                truth_t = bool(gm[i].any()) if y[i] < 0 else bool(y[i] > 0)
                pred_t = (1.0 - pr[i, 0]) > 0.5
                bin_ok += int(truth_t == pred_t)
                n += 1
                if y[i] >= 0:
                    conf[y[i], int(pr[i].argmax())] += 1
                inter, union = (pm[i] & gm[i]).sum(), (pm[i] | gm[i]).sum()
                tp += inter
                fp += (pm[i] & ~gm[i]).sum()
                fn += (~pm[i] & gm[i]).sum()
                if gm[i].any():
                    ious.append(inter / max(union, 1))
    known = conf.sum()
    return {
        "n": n,
        "tamper_acc": bin_ok / max(n, 1),
        "class_acc": float(np.trace(conf) / known) if known else float("nan"),
        "miou": float(np.mean(ious)) if ious else 0.0,
        "pixel_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "confusion": conf,
    }


def print_report(name, r, classes):
    print(f"  [{name}] images={r['n']}  tamper-vs-authentic acc={r['tamper_acc']:.3f}  "
          f"class acc={r['class_acc']:.3f}  mask IoU={r['miou']:.3f}  pixel F1={r['pixel_f1']:.3f}")
    conf = r["confusion"]
    if conf.sum():
        print("     confusion (rows = truth, cols = predicted): " + " | ".join(c[:6] for c in classes))
        for i, c in enumerate(classes):
            tot = conf[i].sum()
            print(f"     {c:10s} {conf[i].tolist()}  recall={conf[i, i] / tot:.2f}" if tot else f"     {c:10s} (none)")


# --------------------------------------------------------------------------- loss
def compute_loss(seg, cls, m, y, ce_w):
    import torch
    import torch.nn.functional as F
    seg, cls = seg.float(), cls.float()
    bce = F.binary_cross_entropy_with_logits(seg, m, pos_weight=torch.tensor(6.0, device=seg.device))
    has = m.flatten(1).sum(1) > 0
    if has.any():
        p, t = torch.sigmoid(seg[has]), m[has]
        inter = (p * t).sum((1, 2, 3))
        dice = 1 - ((2 * inter + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)).mean()
    else:
        dice = seg.sum() * 0
    known = y >= 0
    ce = seg.sum() * 0
    if known.any():
        ce = F.cross_entropy(cls[known], y[known], weight=ce_w, label_smoothing=0.05)
    if (~known).any():                                  # tampered, type unknown: just "not authentic"
        p_auth = torch.softmax(cls[~known], 1)[:, 0]
        ce = ce + (-torch.log((1 - p_auth).clamp_min(1e-6))).mean()
    return bce + dice + ce


# --------------------------------------------------------------------------- training
def train(preset="standard", epochs=None, per_epoch=None, batch=None, lr=None, real=None, real_ratio=0.5,
          workers=None, out=WEIGHTS, resume=False, seed=0, val_size=300):
    try:
        import torch
        from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
    except ImportError:
        print("[train] PyTorch is not installed -> skipping CNN training.\n"
              "        The copy-move detector still works. Install torch from https://pytorch.org to train.")
        return None
    from data.train import FolderDataset, OnTheFlyDataset
    from models.multi_stream_net import CHS_BASE, CHS_SMALL, CLASSES, NUM_CLASSES, MultiStreamForensicNet

    cfg = dict(PRESETS[preset])
    for k, v in (("epochs", epochs), ("per_epoch", per_epoch), ("batch", batch), ("lr", lr)):
        if v is not None:
            cfg[k] = v
    chs = CHS_SMALL if cfg["chs"] == "small" else CHS_BASE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = dev == "cuda"
    if workers is None:
        workers = 2 if (os.cpu_count() or 1) >= 4 else 0
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

    syn = OnTheFlyDataset(cfg["per_epoch"], seed=seed)
    val_syn = OnTheFlyDataset(val_size, seed=999983, fixed=True)
    train_ds, sampler, real_val = syn, None, None
    if real:
        paths = real if isinstance(real, (list, tuple)) else [real]
        tr, va = [], []
        for p in paths:
            names = FolderDataset.list_names(p)
            random.Random(0).shuffle(names)
            k = int(0.85 * len(names))
            tr.append(FolderDataset(p, names[:k], train=True))
            va.append(FolderDataset(p, names[k:], train=False))
        real_tr, real_val = ConcatDataset(tr), ConcatDataset(va)
        train_ds = ConcatDataset([syn, real_tr])
        w = [(1 - real_ratio) / len(syn)] * len(syn) + [real_ratio / max(len(real_tr), 1)] * len(real_tr)
        sampler = WeightedRandomSampler(w, num_samples=cfg["per_epoch"], replacement=True)
        print(f"[train] real data: {len(real_tr)} train / {len(real_val)} val images, mixed at {real_ratio:.0%}")

    loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=sampler is None, sampler=sampler,
                        num_workers=workers, drop_last=True, pin_memory=use_amp)
    val_loader = DataLoader(val_syn, batch_size=32, num_workers=0)
    real_loader = DataLoader(real_val, batch_size=16, num_workers=0) if real_val is not None and len(real_val) else None

    net = MultiStreamForensicNet(chs, NUM_CLASSES).to(dev)
    if resume and os.path.isfile(out):
        ck = torch.load(out, map_location=dev)
        net.load_state_dict(ck["state_dict"])
        print(f"[train] resumed from {out}")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * len(loader), pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce_w = torch.tensor([0.8, 1.0, 1.0, 1.0, 1.0], device=dev)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[train] preset={preset} device={dev} amp={use_amp} params={n_par / 1e6:.2f}M  epochs={cfg['epochs']}  "
          f"samples/epoch={cfg['per_epoch']}  batch={cfg['batch']}  workers={workers}")
    best, t_start = -1.0, time.time()
    try:
        for ep in range(1, cfg["epochs"] + 1):
            syn.set_epoch(ep)
            net.train()
            t0, tot = time.time(), 0.0
            for rgb, ela, srm, m, y in loader:
                rgb, ela, srm, m, y = [t.to(dev, non_blocking=True) for t in (rgb, ela, srm, m, y)]
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    seg, cl = net(rgb, ela, srm)
                loss = compute_loss(seg, cl, m, y, ce_w)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += float(loss)
            r_syn = evaluate(net, val_loader, dev)
            r_real = evaluate(net, real_loader, dev) if real_loader else None
            sel = r_real if r_real else r_syn
            score = sel["tamper_acc"] + sel["miou"]
            eta = (time.time() - t_start) / ep * (cfg["epochs"] - ep)
            print(f"  epoch {ep:02d}/{cfg['epochs']}  loss {tot / len(loader):.3f}  "
                  f"syn: acc {r_syn['tamper_acc']:.3f} cls {r_syn['class_acc']:.3f} IoU {r_syn['miou']:.3f}"
                  + (f"  REAL: acc {r_real['tamper_acc']:.3f} IoU {r_real['miou']:.3f}" if r_real else "")
                  + f"  ({time.time() - t0:.0f}s, ETA {eta / 60:.0f} min)")
            if score > best:
                best = score
                torch.save({"state_dict": net.state_dict(), "chs": list(chs), "num_classes": NUM_CLASSES,
                            "classes": CLASSES, "epoch": ep, "val_tamper_acc": sel["tamper_acc"],
                            "val_iou": sel["miou"]}, out)
    except KeyboardInterrupt:
        print("\n[train] stopped by user - best checkpoint so far is kept")

    if os.path.isfile(out):
        print(f"[train] best checkpoint -> {out}\n[train] final report of the best weights:")
        evaluate_weights(out, None, dev, val_syn, real_loader)
    return out


def evaluate_weights(weights, folder, dev=None, val_syn=None, real_loader=None):
    import torch
    from torch.utils.data import DataLoader
    from data.train import FolderDataset
    from models.multi_stream_net import CLASSES, load_model
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(weights, dev)
    if val_syn is not None:
        print_report("synthetic validation", evaluate(net, DataLoader(val_syn, batch_size=32), dev), CLASSES)
    if real_loader is not None:
        print_report("real validation", evaluate(net, real_loader, dev), CLASSES)
    if folder:
        ds = FolderDataset(folder, train=False)
        print_report(f"folder {folder}", evaluate(net, DataLoader(ds, batch_size=16), dev), CLASSES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="standard")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--per-epoch", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--real", action="append", help="real dataset folder (images/ + masks/); can be repeated")
    ap.add_argument("--real-ratio", type=float, default=0.5)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-dir", help="only evaluate the saved weights on this dataset folder")
    ap.add_argument("--out", default=WEIGHTS)
    a = ap.parse_args()
    if a.eval_dir:
        evaluate_weights(a.out, a.eval_dir)
    else:
        train(a.preset, a.epochs, a.per_epoch, a.batch, a.lr, a.real, a.real_ratio, a.workers, a.out, a.resume)
