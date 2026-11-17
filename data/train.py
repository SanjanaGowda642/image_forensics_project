"""Datasets for train.py / force_train.py.

OnTheFlyDataset : endless synthetic samples (all tamper types), new ones every epoch.
FolderDataset   : any REAL dataset converted to   <root>/images/NAME.ext
                                                  <root>/masks/NAME.png   (white = tampered; missing = authentic)
                                                  <root>/labels.csv       (optional: name,cls   cls = 0..4)
If labels.csv is missing, images with an empty mask are 'authentic' and the rest get
cls = -1 ('tampered, type unknown') which still trains localisation and tamper/authentic.
"""
import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset_generator import make_sample
from models.multi_stream_net import INPUT_SIZE, build_inputs

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _to_tensors(img, mask, cls, size):
    rgb, ela, srm = build_inputs(img, size)
    m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) if mask.shape[:2] != (size, size) else mask
    m = torch.from_numpy((m > 127).astype(np.float32))[None]
    return rgb, ela, srm, m, torch.tensor(int(cls), dtype=torch.long)


class OnTheFlyDataset(Dataset):
    def __init__(self, length, seed=0, size=INPUT_SIZE, fixed=False):
        self.length, self.seed, self.size, self.fixed, self.epoch = length, seed, size, fixed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, 0 if self.fixed else self.epoch, i])
        img, mask, cls = make_sample(rng, self.size)
        return _to_tensors(img, mask, cls, self.size)


class FolderDataset(Dataset):
    def __init__(self, root, names=None, train=True, size=INPUT_SIZE):
        self.root, self.train, self.size = root, train, size
        self.img_dir, self.msk_dir = os.path.join(root, "images"), os.path.join(root, "masks")
        self.files = {os.path.splitext(f)[0]: f for f in os.listdir(self.img_dir) if f.lower().endswith(IMG_EXT)}
        self.names = names if names is not None else sorted(self.files)
        self.cls = {}
        lab = os.path.join(root, "labels.csv")
        if os.path.isfile(lab):
            with open(lab, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("cls", "") != "":
                        self.cls[row["name"]] = int(row["cls"])

    @staticmethod
    def list_names(root):
        d = os.path.join(root, "images")
        return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(IMG_EXT))

    def __len__(self):
        return len(self.names)

    def _crop(self, img, mask, rng):
        s = self.size
        ph, pw = max(0, s - img.shape[0]), max(0, s - img.shape[1])
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        h, w = img.shape[:2]
        ys, xs = np.nonzero(mask)
        if len(ys) and (not self.train or rng.random() < 0.7):      # keep the tampered area in view
            jit = int(s * 0.25) if self.train else 0
            cy, cx = int(ys.mean()) + int(rng.integers(-jit, jit + 1)), int(xs.mean()) + int(rng.integers(-jit, jit + 1))
            y0, x0 = int(np.clip(cy - s // 2, 0, h - s)), int(np.clip(cx - s // 2, 0, w - s))
        else:
            y0, x0 = int(rng.integers(0, h - s + 1)), int(rng.integers(0, w - s + 1))
        return img[y0:y0 + s, x0:x0 + s], mask[y0:y0 + s, x0:x0 + s]

    def __getitem__(self, i):
        n = self.names[i]
        rng = np.random.default_rng(None if self.train else i)
        img = cv2.imread(os.path.join(self.img_dir, self.files[n]), cv2.IMREAD_COLOR)
        mp = os.path.join(self.msk_dir, n + ".png")
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mp) else None
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        elif mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        tampered = bool((mask > 127).any())                    # decided on the FULL mask, before cropping
        img, mask = self._crop(img, mask, rng)
        if self.train:
            if rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        cls = self.cls.get(n, -1 if tampered else 0)
        return _to_tensors(img, mask, cls, self.size)


if __name__ == "__main__":
    print("This module only defines datasets. Train with:  python train.py --preset quick")
