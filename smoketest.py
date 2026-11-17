"""Step-by-step self-check of the PyTorch part. Run it BEFORE training:

    python smoke_test.py

Every stage prints PASS or FAIL. If a stage fails, copy the WHOLE output and send it back - the
traceback shows exactly which line broke. At the end it estimates how long each training preset
would take on THIS machine.
"""
import os
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
S = {}                       # state shared between stages
RESULTS = []


def run(name, fn, fatal=False):
    t0 = time.time()
    try:
        note = fn() or ""
        RESULTS.append((name, True))
        print(f"[PASS] {name}  {note}  ({time.time() - t0:.1f}s)", flush=True)
        return True
    except Exception:
        RESULTS.append((name, False))
        print(f"[FAIL] {name}", flush=True)
        traceback.print_exc()
        if fatal:
            print("\nCannot continue without this stage.")
            summary()
            sys.exit(1)
        return False


def summary():
    print("\n" + "=" * 60)
    for n, ok in RESULTS:
        print(("  PASS  " if ok else "  FAIL  ") + n)
    bad = [n for n, ok in RESULTS if not ok]
    if bad:
        print(f"\n{len(bad)} stage(s) failed - send this whole output back.")
    else:
        print("\nAll stages passed. Next:  python force_train.py --preset quick --workers 0")


# ------------------------------------------------------------------ 1. environment
def stage_env():
    import cv2
    import numpy as np
    import torch
    S["torch"] = torch
    S["dev"] = "cuda" if torch.cuda.is_available() else "cpu"
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    v = sys.version_info
    warn = "" if v.major == 3 and 9 <= v.minor <= 13 else "  WARNING: use Python 3.11 or 3.12"
    return (f"python {v.major}.{v.minor}  torch {torch.__version__}  numpy {np.__version__}  opencv {cv2.__version__}  "
            f"device={S['dev']}" + (f" ({torch.cuda.get_device_name(0)})" if S["dev"] == "cuda" else "")
            + ("  [Intel XPU is available but not used by the scripts]" if xpu else "") + warn)


# ------------------------------------------------------------------ 2. data
def stage_data():
    from data.train import OnTheFlyDataset
    ds = OnTheFlyDataset(16, seed=0)
    t0 = time.time()
    rgb, ela, srm, m, y = ds[0]
    S["t_sample"] = (time.time() - t0)
    assert tuple(rgb.shape) == (3, 256, 256) and tuple(ela.shape) == (3, 256, 256) and tuple(srm.shape) == (3, 256, 256), "bad input shape"
    assert tuple(m.shape) == (1, 256, 256), "bad mask shape"
    assert int(y) in range(5), "bad label"
    t0 = time.time()
    for i in range(1, 9):
        ds[i]
    S["t_sample"] = (time.time() - t0) / 8
    S["ds"] = ds
    return f"sample shapes ok, {S['t_sample'] * 1000:.0f} ms per sample"


# ------------------------------------------------------------------ 3. model
def stage_model():
    torch = S["torch"]
    from models.multi_stream_net import CHS_BASE, CHS_SMALL, MultiStreamForensicNet
    out = []
    for tag, chs in (("small", CHS_SMALL), ("base", CHS_BASE)):
        net = MultiStreamForensicNet(chs).to(S["dev"]).eval()
        n_par = sum(p.numel() for p in net.parameters())
        x = [torch.randn(2, 3, 256, 256, device=S["dev"]) for _ in range(3)]
        with torch.no_grad():
            seg, cls = net(*x)
        assert tuple(seg.shape) == (2, 1, 256, 256), f"seg shape {tuple(seg.shape)}"
        assert tuple(cls.shape) == (2, 5), f"cls shape {tuple(cls.shape)}"
        assert torch.isfinite(seg).all() and torch.isfinite(cls).all(), "non-finite outputs"
        out.append(f"{tag}: {n_par / 1e6:.2f}M params")
    return "; ".join(out)


# ------------------------------------------------------------------ 4. training steps
def stage_train():
    torch = S["torch"]
    from torch.utils.data import DataLoader
    from models.multi_stream_net import CHS_BASE, CHS_SMALL, MultiStreamForensicNet
    from train import compute_loss
    dev = S["dev"]
    batch = next(iter(DataLoader(S["ds"], batch_size=8, shuffle=False, num_workers=0)))
    rgb, ela, srm, m, y = [t.to(dev) for t in batch]
    ce_w = torch.tensor([0.8, 1.0, 1.0, 1.0, 1.0], device=dev)
    notes, S["t_step"] = [], {}
    for tag, chs in (("small", CHS_SMALL), ("base", CHS_BASE)):
        net = MultiStreamForensicNet(chs).to(dev).train()
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
        losses, times = [], []
        for _ in range(6):
            t0 = time.time()
            seg, cl = net(rgb, ela, srm)
            loss = compute_loss(seg, cl, m, y, ce_w)
            assert torch.isfinite(loss), "loss is not finite"
            opt.zero_grad()
            loss.backward()
            opt.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)
            losses.append(float(loss))
        S["t_step"][tag] = float(sum(times[2:]) / len(times[2:]))            # skip warm-up steps
        notes.append(f"{tag}: loss {losses[0]:.2f} -> {losses[-1]:.2f}, {S['t_step'][tag]:.2f}s per batch of 8")
        if tag == "small":
            S["net"] = net
        assert losses[-1] < losses[0], f"{tag}: loss did not go down on a repeated batch ({losses})"
    return "; ".join(notes)


# ------------------------------------------------------------------ 5. evaluation
def stage_eval():
    from torch.utils.data import DataLoader
    from train import evaluate, print_report
    val = [S["ds"][i] for i in range(16)]
    r = evaluate(S["net"], DataLoader(val, batch_size=8), S["dev"])
    assert {"tamper_acc", "class_acc", "miou", "pixel_f1", "confusion"} <= set(r), "missing metrics"
    return f"images={r['n']} tamper-acc={r['tamper_acc']:.2f} (untrained network, any value is fine)"


# ------------------------------------------------------------------ 6. checkpoint + predict
def stage_checkpoint():
    import numpy as np
    torch = S["torch"]
    from models.multi_stream_net import CHS_SMALL, CLASSES, NUM_CLASSES, load_model, predict
    path = os.path.join(tempfile.gettempdir(), "smoke_forensic_net.pt")
    torch.save({"state_dict": S["net"].state_dict(), "chs": list(CHS_SMALL), "num_classes": NUM_CLASSES, "classes": CLASSES}, path)
    net = load_model(path, S["dev"])
    S["ckpt"] = path
    img = (np.random.default_rng(0).random((301, 417, 3)) * 255).astype(np.uint8)       # not a multiple of 8 on purpose
    probs, heat = predict(net, img, S["dev"])
    assert probs.shape == (5,) and abs(float(probs.sum()) - 1.0) < 1e-3, f"bad probabilities {probs}"
    assert heat.shape == img.shape[:2], f"heat-map {heat.shape} != image {img.shape[:2]}"
    return f"save/load ok, probs sum to 1, heat-map {heat.shape}"


# ------------------------------------------------------------------ 7. whole pipeline
def stage_pipeline():
    from data.dataset_generator import make_demo_forgery
    from utils.pipeline import ForensicsPipeline
    p = ForensicsPipeline(weights=S["ckpt"])
    assert p.net is not None, f"network was not loaded: {p.cnn_error}"
    img = make_demo_forgery(w=640, h=480)[0]
    info = p.analyze(img)["info"]
    assert info["cnn_used"] and info["cnn_probs"], "network output missing from the report"
    return f"label={info['label']}  network probabilities={info['cnn_probs']}"


# ------------------------------------------------------------------ time estimate
def estimate():
    from train import PRESETS
    cpus = os.cpu_count() or 1
    workers = 2 if cpus >= 4 else 1
    print(f"\nRough training-time estimate on THIS machine ({S['dev']}, {cpus} CPU threads, {workers} data workers).")
    print("(uses the speed measured above; real runs add validation time and, on GPU, run faster with mixed precision)")
    for name, cfg in PRESETS.items():
        step = S["t_step"][cfg["chs"]] * cfg["batch"] / 8
        per_epoch = cfg["per_epoch"] / cfg["batch"] * step + cfg["per_epoch"] * S["t_sample"] / workers
        total = per_epoch * cfg["epochs"]
        print(f"   {name:9s} {cfg['epochs']:3d} epochs x {cfg['per_epoch']:5d} samples  ->  about {total / 3600:6.1f} hours  ({per_epoch / 60:.1f} min per epoch)")


if __name__ == "__main__":
    print("Image-forensics PyTorch self-check\n")
    run("1 environment (python, torch, device)", stage_env, fatal=True)
    ok = run("2 synthetic data sample", stage_data)
    ok = run("3 model builds and runs a forward pass", stage_model) and ok
    if ok:
        run("4 training steps (loss must go down)", stage_train)
    if "net" in S:
        run("5 evaluation and metrics", stage_eval)
        run("6 save / load checkpoint and predict()", stage_checkpoint)
    if "ckpt" in S:
        run("7 full pipeline with the network", stage_pipeline)
    if "t_step" in S:
        try:
            estimate()
        except Exception:
            traceback.print_exc()
    summary()