"""Launcher.

    python run.py                    start the web app (http://127.0.0.1:5000)
    python run.py --image my.jpg     analyse one image from the command line
    python run.py --demo             run the built-in demo from the command line
    python run.py --train            (re)train the optional CNN first
Results are written to the outputs/ folder.
"""
import argparse
import importlib
import os
import sys
import threading
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "outputs")

REQUIRED = {"numpy": "numpy", "cv2": "opencv-python", "PIL": "pillow", "flask": "flask"}


def check_dependencies():
    missing = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        sys.exit(
            "\nMissing packages: " + ", ".join(missing)
            + "\n\nInstall them into THIS Python with:\n\n"
            + f'  "{sys.executable}" -m pip install ' + " ".join(missing)
            + "\n\n(or:  pip install -r requirements.txt  after activating your .venv)\n"
        )


def cli(img, name):
    from utils.pipeline import ForensicsPipeline
    res = ForensicsPipeline().analyze(img, OUT, name)
    i = res["info"]
    print(f"\n  Label       : {i['label']}\n  Confidence  : {i['confidence']:.2f}\n  Intensity   : {i['intensity']:.2f}"
          f"\n  Area        : {i['area_pct']}%\n  Regions     : {i['regions_matched']}\n  Resolution  : {i['resolution']}")
    if i.get("warning"):
        print(f"\n  Note        : {i['warning']}")
    print(f"\n  Saved to {OUT}:  {name}_original / _overlay / _mask / _cutout / _restored / _panel .png")


if __name__ == "__main__":
    check_dependencies()
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    if a.train:
        from train import train
        train()
    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            sys.exit(f"cannot read {a.image}")
        cli(img, os.path.splitext(os.path.basename(a.image))[0])
    elif a.demo:
        from data.dataset_generator import make_demo_forgery
        cli(make_demo_forgery()[0], "demo")
    else:
        from app import app
        url = f"http://{a.host}:{a.port}"
        print(f"\n  Image Forensics running at {url}   (Ctrl+C to stop)\n")
        if not a.no_browser:
            threading.Timer(1.2, lambda: webbrowser.open(url)).start()
        app.run(host=a.host, port=a.port, debug=False, threaded=True)
