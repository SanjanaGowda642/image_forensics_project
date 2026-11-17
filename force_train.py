"""Force a clean re-train from scratch (deletes the old weights first).

    python force_train.py --preset quick                 # smoke test
    python force_train.py --preset standard              # recommended (GPU)
    python force_train.py --preset hard --real D:\\data\\casia_converted
"""
import argparse
import os

from train import PRESETS, WEIGHTS, train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="standard")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--real", action="append", help="real dataset folder (images/ + masks/); can be repeated")
    ap.add_argument("--real-ratio", type=float, default=0.5)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    if os.path.isfile(WEIGHTS):
        os.remove(WEIGHTS)
        print("[force_train] removed old weights")
    train(a.preset, epochs=a.epochs, real=a.real, real_ratio=a.real_ratio, workers=a.workers)
